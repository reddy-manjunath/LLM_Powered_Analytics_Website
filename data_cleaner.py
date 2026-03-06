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
    for col in df.columns:
        column_name_list.append(col)
        samples = (df[col].astype(str).sample(5).tolist())
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




result = take_llm_decision(df)

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
                except:
                    obj['confidence_level'] = line.replace('confidence level:', '').strip()
            elif obj['columnName'] == '':
                obj['columnName'] = line.strip().split()[0] 
        data.append(obj)
else:
    data = result


colums_dtype_list={}
for col in data:
    colums_dtype_list[col['columnName']]=col['datatype']


def type_casting(df, colums_dtype_list):
    for col in colums_dtype_list:
        if colums_dtype_list[col] in ("float", "integer", "int", "float64", "int64"):
            df[col] = pd.to_numeric(df[col], errors='coerce')
        elif colums_dtype_list[col] == "datetime":
            df[col]=df[col].astype(str).str.replace("-","/")
            df[col]=pd.to_datetime(df[col],dayfirst=True,format='mixed')
        elif colums_dtype_list[col] == 'string':
            df[col] = df[col].astype('object').astype(str)
    return df

categorical_columns=[]
numerical_columns=[]
for col in df.columns:
    if pd.api.types.is_numeric_dtype(df[col]):
        numerical_columns.append(col)
    else:
        categorical_columns.append(col)

def fill_null_values_KNN(df,numerical_columns,categorical_columns):
    
    imputer = IterativeImputer(estimator=RandomForestRegressor(),max_iter=10,random_state=42)
    df[numerical_columns] = imputer.fit_transform(df[numerical_columns])
    imputer = SimpleImputer(strategy='most_frequent')
    df[categorical_columns] = imputer.fit_transform(df[categorical_columns])
    for col in categorical_columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            s = pd.to_datetime(df[col], errors='coerce')
            ints = s.map(lambda x: x.value if pd.notnull(x) else np.nan).astype('float')
            ints = pd.Series(ints, index=df.index)
            ints_interp = ints.interpolate(method='linear', limit_direction='both')
            result = pd.Series([pd.NaT] * len(df), index=df.index, dtype='datetime64[ns]')
            mask = ints_interp.notna()
            if mask.any():
                result.loc[mask] = pd.to_datetime(ints_interp[mask].astype('int64'), unit='ns')
            df[col] = result
    return df

def standardizevalues(df,categorical_columns):
    for col in categorical_columns:
        if  not df[col].dtype=='datetime64[ns]':
            df[col] = df[col].str.strip()
            df[col] = df[col].str.replace(r'\s+', '', regex=True)
    return df

def remove_duplicates(df):
    df=df.drop_duplicates()
    return df