import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor,KNeighborsClassifier
from sklearn.tree import DecisionTreeRegressor,DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.impute import SimpleImputer
import warnings
import json
import re
warnings.filterwarnings("ignore")
import seaborn  as sns
from sklearn.datasets import *
import requests
import io
import tempfile

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
import uvicorn


# ─────────────────────────────────────────────
# FastAPI App
# ─────────────────────────────────────────────

app = FastAPI(
    title="Data Cleaner API",
    description="Upload a CSV file and get a cleaned version back, powered by LLM-driven type inference.",
    version="1.0.0",
)


# ══════════════════════════════════════════════
#  ORIGINAL FUNCTIONS  (unchanged)
# ══════════════════════════════════════════════

def load_csv(path_to):
    df=pd.read_csv(path_to)
    return "loaded successfully"


def call_llm(prompt):
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "gemma3:4b",
                "prompt": prompt,
                "stream": False
            },
            timeout=300
        )
        
        # Check HTTP status
        if response.status_code != 200:
            print(f"Error: HTTP {response.status_code}")
            print(f"Response: {response.text}")
            return None
            
        response_json = response.json()
        
        # Check if response has the expected key
        if "response" in response_json:
            return response_json["response"]
        else:
            print(f"Error: 'response' key not found in API response")
            print(f"Available keys: {response_json.keys()}")
            print(f"Full response: {response_json}")
            return None
            
    except requests.exceptions.ConnectionError:
        print("Error: Cannot connect to Ollama at http://localhost:11434")
        print("Make sure Ollama is running: ollama serve")
        return None
    except Exception as e:
        print(f"Error: {type(e).__name__}: {e}")
        return None


def take_llm_decision_for_column(df):
    samples_list = {}
    column_name_list = []
    colum_before_dtype=[]
    n_samples = min(5, len(df))  # Fix: guard against datasets with <5 rows
    for col in df.columns:
        column_name_list.append(col)
        samples = (df[col].astype(str).sample(n_samples).tolist())
        samples_list[col] = samples
        colum_before_dtype.append(str(df[col].dtype))

    prompt_prefix = (
        f"Column names: {column_name_list}\n"
        f"Sample values from each column (JSON): {json.dumps(samples_list)}\n"
        f"Previous  dtype for the column: {json.dumps(colum_before_dtype)}\n"
    )

    prompt_body = (
        "You are a senior data analyst with 10 years of experience.\n"
        "Based on column names, sample values,previous data type for the column infer the most appropriate datatype.\n"
        "Do not assume the data is cleaned.\n"
        "Dates may appear as strings but should be classified as datetime.\n\n"
        "Return ONLY a valid JSON array.\n"
        "Each object must contain:\n"
        "- columnName (string)\n"
        "- datatype (one of: int, float, string, boolean, datetime, category)\n"
        "- confidence_level (number between 0 and 1)\n"
    )
    prompt_example = ('[{"columnName": "order_date", "datatype": "datetime", "confidence_level": 0.95}]')
    prompt_ = (prompt_prefix+ prompt_body+ "Example output format:\n"+ prompt_example)
    result = call_llm(prompt_)

    try:
        match = re.search(r"\[.*\]", result, re.S)
        if not match:
            return None

        data = json.loads(match.group())
        return data

    except Exception:
        return None


def type_casting(df, colums_dtype_list):
    for col in colums_dtype_list:
        if col not in df.columns:
            continue
        try:
            if colums_dtype_list[col] in ("float", "integer", "int", "float64", "int64"):
                # Safety: don't convert if it would destroy >20% of non-null values
                converted = pd.to_numeric(df[col], errors='coerce')
                non_null_before = df[col].notna().sum()
                non_null_after = converted.notna().sum()
                if non_null_before > 0 and (non_null_after / non_null_before) < 0.8:
                    print(f"Skipping numeric cast for '{col}': would lose {non_null_before - non_null_after} values")
                    continue
                df[col] = converted
            elif colums_dtype_list[col] == "datetime":
                df[col]=df[col].astype(str).str.replace("-","/")
                df[col]=pd.to_datetime(df[col],dayfirst=True,format='mixed')
            elif colums_dtype_list[col] == 'string':
                # Don't convert to string — it turns NaN into literal "nan"
                # Just ensure it's object dtype
                df[col] = df[col].astype('object')
        except Exception as e:
            print(f"Warning: type_casting failed for column '{col}': {e}")
    return df


def fill_null_values(df, numerical_columns, categorical_columns, datetime_columns=None):
    """
    EDA-ready null imputation:
      - ID columns (>90% unique)       → SKIP (leave NaN as-is)
      - Numerical                      → median
      - Low-cardinality categorical    → mode
      - High-cardinality categorical   → SKIP (leave NaN as-is)
      - Datetime                       → linear interpolation
    """
    if datetime_columns is None:
        datetime_columns = []

    # --- Numerical columns ---
    for col in numerical_columns:
        if df[col].isna().sum() == 0:
            continue
        # Skip ID-like numerical columns (e.g., PassengerId)
        non_null = df[col].dropna()
        if len(non_null) > 0 and df[col].nunique() / len(non_null) > 0.9:
            print(f"Skipping imputation for ID column: '{col}'")
            continue
        median_val = df[col].median()
        if pd.isna(median_val):
            continue  # All-null — leave as-is
        df[col] = df[col].fillna(median_val)

    # --- Categorical columns ---
    for col in categorical_columns:
        if df[col].isna().sum() == 0:
            continue
        non_null = df[col].dropna()
        if len(non_null) == 0:
            continue  # All-null — leave as-is
        unique_ratio = df[col].nunique() / len(non_null)
        # High-cardinality columns (Ticket, Name, etc.) → don't impute
        if unique_ratio > 0.5:
            print(f"Skipping imputation for high-cardinality column: '{col}'")
            continue
        mode_series = df[col].mode()
        if len(mode_series) == 0:
            continue
        mode_val = mode_series[0]
        if isinstance(mode_val, str) and ' ' in mode_val:
            mode_val = mode_val.split()[0]
        df[col] = df[col].fillna(mode_val)

    # --- Datetime columns ---
    for col in datetime_columns:
        if df[col].isna().sum() == 0:
            continue
        try:
            s = pd.to_datetime(df[col], errors='coerce')
            ints = s.map(lambda x: x.value if pd.notnull(x) else np.nan).astype('float')
            ints = pd.Series(ints, index=df.index)
            ints_interp = ints.interpolate(method='linear', limit_direction='both')
            result = pd.Series([pd.NaT] * len(df), index=df.index, dtype='datetime64[ns]')
            mask = ints_interp.notna()
            if mask.any():
                result.loc[mask] = pd.to_datetime(ints_interp[mask].astype('int64'), unit='ns')
            df[col] = result
        except Exception as e:
            print(f"Warning: datetime imputation failed for '{col}': {e}")

    return df


def standardizevalues(df,categorical_columns):
    for col in categorical_columns:
        # Only apply string ops to actual string/object columns
        if df[col].dtype == 'object':
            try:
                df[col] = df[col].str.strip()
                df[col] = df[col].str.replace(r'\s+', ' ', regex=True)  # Normalize whitespace
            except Exception:
                pass
    return df


def remove_duplicates(df):
    df=df.drop_duplicates()
    return df


# ══════════════════════════════════════════════
#  PIPELINE HELPER FUNCTIONS  (new)
# ══════════════════════════════════════════════

def parse_llm_result(result):
    """
    Parse the raw LLM result into a list of dicts.
    Handles both JSON-list returns and plain-text returns.
    """
    if result is None:
        return None

    # If take_llm_decision_for_column already returned a parsed list, use it directly
    if isinstance(result, list):
        return result

    # Otherwise, parse the plain-text format
    if isinstance(result, str):
        sections = result.split('columnName:')[1:]
        data = []

        for section in sections:
            obj = {"columnName": ""}
            lines = section.strip().split('\n')

            for line in lines:
                if line.startswith('datatype:'):
                    obj['datatype'] = line.replace('datatype:', '').strip()
                elif line.startswith('confidence level:'):
                    try:
                        obj['confidence_level'] = float(line.replace('confidence level:', '').strip())
                    except Exception:
                        obj['confidence_level'] = line.replace('confidence level:', '').strip()
                elif obj['columnName'] == '':
                    obj['columnName'] = line.strip().split()[0]
            data.append(obj)
        return data

    return None


def build_dtype_map(data):
    """
    Build a {column_name: datatype} mapping from the parsed LLM result.
    """
    colums_dtype_list = {}
    for col in data:
        colums_dtype_list[col['columnName']] = col['datatype']
    return colums_dtype_list


def classify_columns(df):
    """
    Split DataFrame columns into numerical, categorical, and datetime lists.
    """
    categorical_columns = []
    numerical_columns = []
    datetime_columns = []
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            datetime_columns.append(col)
        elif pd.api.types.is_numeric_dtype(df[col]):
            numerical_columns.append(col)
        else:
            categorical_columns.append(col)
    return numerical_columns, categorical_columns, datetime_columns


# ══════════════════════════════════════════════
#  CLEANING PIPELINE  (orchestrator)
# ══════════════════════════════════════════════

def infer_dtypes_heuristic(df):
    """
    Fallback: infer column dtypes from pandas when LLM is unavailable.
    """
    DATE_KEYWORDS = ['date', 'time', 'timestamp', 'created', 'updated', 'dob', 'birth']
    result = []
    for col in df.columns:
        dtype = str(df[col].dtype)
        if pd.api.types.is_integer_dtype(df[col]):
            inferred = 'int'
        elif pd.api.types.is_float_dtype(df[col]):
            inferred = 'float'
        elif pd.api.types.is_bool_dtype(df[col]):
            inferred = 'boolean'
        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            inferred = 'datetime'
        elif any(kw in col.lower() for kw in DATE_KEYWORDS):
            # Try to parse as datetime
            try:
                pd.to_datetime(df[col].dropna().head(10), format='mixed')
                inferred = 'datetime'
            except Exception:
                inferred = 'string'
        else:
            # Check if numeric-convertible
            converted = pd.to_numeric(df[col], errors='coerce')
            non_null_original = df[col].dropna()
            non_null_converted = converted.dropna()
            if len(non_null_original) > 0 and len(non_null_converted) / len(non_null_original) > 0.8:
                if (non_null_converted == non_null_converted.astype(int)).all():
                    inferred = 'int'
                else:
                    inferred = 'float'
            else:
                inferred = 'string'
        result.append({'columnName': col, 'datatype': inferred, 'confidence_level': 0.7})
    return result


def replace_nan_strings(df):
    """
    Replace literal 'nan' strings back to actual NaN across the DataFrame.
    This fixes damage from .astype(str) converting NaN → 'nan'.
    """
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].replace({'nan': np.nan, 'NaN': np.nan, 'None': np.nan, '': np.nan})
    return df


def drop_high_null_columns(df, threshold=0.7):
    """
    Drop columns where >threshold of values are null.
    These columns add noise, not signal, for EDA.
    """
    null_pct = df.isnull().mean()
    cols_to_drop = null_pct[null_pct > threshold].index.tolist()
    if cols_to_drop:
        print(f"Dropping columns with >{threshold*100:.0f}% nulls: {cols_to_drop}")
        df = df.drop(columns=cols_to_drop)
    return df


def run_cleaning_pipeline(df):
    """
    EDA-ready data cleaning pipeline:
      1. Infer column dtypes (LLM or heuristic fallback)
      2. Type-cast columns (with data-loss protection)
      3. Replace 'nan' strings with actual NaN
      4. Drop columns with >70% null values
      5. Classify columns (numerical / categorical / datetime)
      6. Impute nulls (skip ID & high-cardinality columns)
      7. Standardize string values
      8. Remove duplicates
    Returns the cleaned DataFrame.
    """

    # Step 1 – Infer dtypes (LLM with heuristic fallback)
    result = take_llm_decision_for_column(df)
    data = parse_llm_result(result)
    if data is None:
        print("LLM unavailable — using heuristic dtype inference.")
        data = infer_dtypes_heuristic(df)

    # Step 2 – Build dtype map & cast (protected against data loss)
    colums_dtype_list = build_dtype_map(data)
    df = type_casting(df, colums_dtype_list)

    # Step 3 – Fix 'nan' strings from astype(str)
    df = replace_nan_strings(df)

    # Step 4 – Drop columns with too many nulls (useless for EDA)
    df = drop_high_null_columns(df, threshold=0.7)

    # Step 5 – Classify columns
    numerical_columns, categorical_columns, datetime_columns = classify_columns(df)

    # Step 6 – Impute nulls (only where it makes sense)
    df = fill_null_values(df, numerical_columns, categorical_columns, datetime_columns)

    # Step 7 – Standardize string values
    if categorical_columns:
        df = standardizevalues(df, categorical_columns)

    # Step 8 – Remove duplicates
    df = remove_duplicates(df)

    return df


# ══════════════════════════════════════════════
#  FASTAPI ENDPOINTS
# ══════════════════════════════════════════════

@app.get("/health")
def health_check():
    """Simple health-check endpoint."""
    return {"status": "healthy"}


@app.post("/clean")
async def clean_csv(file: UploadFile = File(...)):
    """
    Upload a CSV file → run the cleaning pipeline → return the cleaned CSV.
    """
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are accepted.")

    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read CSV: {e}")

    try:
        cleaned_df = run_cleaning_pipeline(df)
    except ValueError as e:
        raise HTTPException(status_code=502, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline error: {e}")

    # Stream the cleaned CSV back to the client
    output = io.StringIO()
    cleaned_df.to_csv(output, index=False)
    output.seek(0)

    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=cleaned_{file.filename}"},
    )


@app.post("/ask")
async def ask_question(file: UploadFile = File(...), question: str = "Describe this dataset"):
    """
    Upload a CSV + ask a natural language question → get an AI-powered answer.
    Runs: clean pipeline → AI engine → structured response.
    """
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are accepted.")

    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read CSV: {e}")

    # Clean the data
    try:
        cleaned_df = run_cleaning_pipeline(df)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cleaning error: {e}")

    # Run AI engine
    try:
        from ai_engine import AIEngine
        engine = AIEngine(cleaned_df)
        result = engine.ask(question)
    except ImportError:
        raise HTTPException(status_code=500, detail="ai_engine.py not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI engine error: {e}")

    return result


# ─────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run("data_cleaner_api:app", host="127.0.0.1", port=8000, reload=True)
