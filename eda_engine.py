"""
eda_engine.py — Automated Exploratory Data Analysis Engine
==========================================================
Performs univariate, bivariate, and multivariate analysis on any dataset.
Generates statistical summaries, insights, and visualizations.

Usage:
    from eda_engine import EDAEngine

    engine = EDAEngine(df)
    insights = engine.run_full_eda()
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gaussian_kde, chi2_contingency
import seaborn.objects as so
import itertools
import requests
import json
import ast
import re
import logging
from typing import Dict, List, Optional, Any
from pathlib import Path


# ─────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("eda_engine")

import uuid
CHARTS_DIR = Path("eda_charts")
CHARTS_DIR.mkdir(exist_ok=True)
saved_charts = []

def custom_show():
    fig = plt.gcf()
    # Check if figure is empty
    if not fig.get_axes():
        plt.close(fig)
        return
        
    uid = str(uuid.uuid4())[:8]
    filename = f"eda_chart_{uid}.png"
    filepath = CHARTS_DIR / filename
    
    # Save it and add to tracking list
    try:
        fig.savefig(filepath, bbox_inches="tight")
        saved_charts.append(filename)
    except Exception as e:
        logger.error(f"Failed to save chart: {e}")
    finally:
        plt.close(fig)

# Override matplotlib's show function so all existing plot functions save instead
plt.show = custom_show


# ══════════════════════════════════════════════
#  Column Classification
# ══════════════════════════════════════════════

def classify_columns(df):
    """Split DataFrame columns into numerical, categorical, and time lists."""
    numerical_columns = []
    categorical_columns = []
    time_column = []
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            numerical_columns.append(col)
        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            time_column.append(col)
        else:
            categorical_columns.append(col)
    return numerical_columns, categorical_columns, time_column


# ══════════════════════════════════════════════
#  Statistical Analysis Functions
# ══════════════════════════════════════════════

def remove_outliers(df):
    for col in df.columns.tolist():
        q1=df[col].quantile(0.25)
        q3=df[col].quantile(0.75)
        iqr=q3-q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        df = df[(df[col] > lower_bound) | (df[col] < upper_bound)]
    return df

def dataset_shape(df,i):
    return df.shape[i]

def column_types_raw(df):
    types={}
    for col in df.columns:
        types[col]=df[col].dtype
    return types

def missing_value_summary(df):
    null={}
    for col in df.columns:
        value=((df[col].isnull().sum())/df.shape[0])*100
        null[col]=value
    return null

def univariate_numerical_summary(df,numerical_columns):
    df_na=df[numerical_columns]
    return df_na.describe()

def univariate_numeric(df,numerical_columns):
    output=[]
    if len(df) < 1:
        return {
            "distribution_type": "insufficient_data",
            "description": "Too few data points to determine distribution reliably."
        }
    for col in numerical_columns:
        mean=df[col].mean()
        median=df[col].median()
        std=df[col].std()
        if std < 0.5 * mean:
            variance_level = "low"
        elif std < mean:
            variance_level = "moderate"
        else:
            variance_level = "high"

        min_val=df[col].min()
        max_val=df[col].max()
        skewness=df[col].skew()
        kurt=df[col].kurt()
        q1,q3,q95,q99 = df[col].quantile([0.25, 0.75,0.95,0.99])
        iqr = q3-q1
        value_range = df[col].max()-df[col].min()
        concentrated = iqr < 0.25 * value_range

        kde = gaussian_kde(df[col])
        xs = np.linspace(df[col].min(), df[col].max(), 200)
        density = kde(xs)
        peaks = ((density[1:-1] > density[:-2]) & (density[1:-1] > density[2:])).sum()

        if np.abs(skewness) < 0.5:
            skew_desc = "approximately symmetric"
        elif skewness > 0:
            skew_desc = "right-skewed"
        else:
            skew_desc = "left-skewed"


        if kurt > 1:
            tail_desc = "heavy-tailed"
        elif kurt < -1:
            tail_desc = "light-tailed"
        else:
            tail_desc = "moderate tails"

        
        if peaks >= 2:
            modality = "bimodal or multimodal"
        else:
            modality = "unimodal"
            
        line=(f'''Column : {col} ->[ mean: {round(mean,2)} | median : {round(median,2)} | std: {round(std,2)}| variance_level: {variance_level} | skewness : {skew_desc} | tail_behaviour : {tail_desc} | modality : {modality} | min_value : {min_val} | max_value : {max_val} | p25 : {q1} | p75 : {q3}] | p95 : {q95} | p99 : {q99}''')
        output.append(line)
    return output

def univariate_time(df,time_column):
    list1=[]
    for col in time_column:
        count=df[col].value_counts()
        dates=count.index
        value_count=count.values
        line=f"column_name -> {col} | sample_ values : {df[col].sample(5)} | Respective_Value _counts : {value_count}"
        list1.append(line)
    return list1

def univariate_category(df,categorical_columns):
    list1=[]
    for col in categorical_columns:
        unique_count = df[col].nunique()

        if unique_count <= 10:
            cardinality = "low"
        elif unique_count <= 50:
            cardinality = "medium"
        else:
            cardinality = "high"

        count=df[col].value_counts()
        dates=count.index
        value_count=count.values
        dominant=df[col].value_counts().idxmax()
        value_counts = df[col].value_counts(normalize=True)
        top_category = value_counts.index[0]
        top_share = round(value_counts.iloc[0] * 100, 2)
        distribution = "imbalanced" if top_share > 60 else "balanced"

        # Rare categories
        rare_presence = "yes" if (value_counts < 0.05).any() else "no"

        line = f"column -> {col} | unique_count: {unique_count} | cardinality_level: {cardinality} | top_category: {top_category} | top_share: {top_share}% | distribution: {distribution} | are_categories: {rare_presence} | "
    list1.append(line)
    return list1
    
def bivariate_numeric_vs_category(df,numerical_columns,categorical_columns):
    results=[]
    for cat in categorical_columns:
        for num in numerical_columns:
            summary = df.groupby(cat)[num].agg(['sum', 'mean', 'median', 'std', 'count'])
            max_mean = summary['mean'].max()
            min_mean = summary['mean'].min()
            mean_ratio = round(max_mean / min_mean, 2) if min_mean != 0 else None

            dominant_category = summary['count'].idxmax()
            dominant_share = round(summary['count'].max() / summary['count'].sum(), 2)
            variability = np.std(summary['mean'])
            
            if mean_ratio and mean_ratio > 1.5:
                effect = "strong"
            elif mean_ratio and mean_ratio > 1.2:
                effect = "moderate"
            else:
                effect = "weak"

            line=f"pair : {num} vs {cat} | categories : {len(summary)} | mean_range : {round(min_mean,2)}-{round(max_mean,2)} | mean_ratio : {mean_ratio} | dominant_category  {dominant_category} | dominant_share : {dominant_share} | effect_strength : {effect} "
            results.append(line)
    return results

def bivariate_numeric_vs_numeric(df, numerical_columns):
    lines = []

    for num1, num2 in itertools.combinations(numerical_columns, 2):
        corr = df[num1].corr(df[num2])

        if corr is None:
            continue

        corr = round(corr, 3)
        direction = "positive" if corr > 0 else "negative" if corr < 0 else "none"

        abs_corr = np.abs(corr)
        if abs_corr >= 0.7:
            strength = "strong"
        elif abs_corr >= 0.4:
            strength = "moderate"
        elif abs_corr >= 0.2:
            strength = "weak"
        else:
            strength = "negligible"

        line = (
            f"pair: {num1} vs {num2} | "
            f"correlation: {corr} | "
            f"direction: {direction} | "
            f"strength: {strength} | "
            f"relationship: linear_association | "
            f"causation: not_established"
        )

        lines.append(line)

    return lines

def bivariate_category_vs_category(df, categorical_columns):
    lines = []

    for col1, col2 in itertools.combinations(categorical_columns, 2):
        contingency = pd.crosstab(df[col1], df[col2])

        if contingency.shape[0] < 2 or contingency.shape[1] < 2:
            continue

        chi2, p, dof, _ = chi2_contingency(contingency)

        # Association strength (Cramér's V)
        n = contingency.values.sum()
        r, k = contingency.shape
        cramers_v = (chi2 / (n * (min(r - 1, k - 1)))) ** 0.5
        cramers_v = round(cramers_v, 3)

        if cramers_v >= 0.5:
            strength = "strong"
        elif cramers_v >= 0.3:
            strength = "moderate"
        elif cramers_v >= 0.1:
            strength = "weak"
        else:
            strength = "negligible"

        dependency = "dependent" if p < 0.05 else "independent"

        line = (
            f"pair: {col1} vs {col2} | "
            f"test: chi_square | "
            f"p_value: {round(p,4)} | "
            f"cramers_v: {cramers_v} | "
            f"association_strength: {strength} | "
            f"dependency: {dependency}"
        )

        lines.append(line)

    return lines


def bivariate_time_vs_numeric(df, time_column, numerical_columns, agg="mean"):
    lines = []

    # --- VALIDATION ---
    if isinstance(time_column, list):
        raise ValueError("time_column must be a single column name (string), not a list")

    if time_column not in df.columns:
        raise ValueError(f"{time_column} not found in DataFrame")

    df = df.copy()

    # Drop rows where time could not be parsed
    df = df.dropna(subset=[time_column])

    for num in numerical_columns:
        if num not in df.columns:
            continue

        ts = (
            df
            .groupby(df[time_column].dt.to_period("M"))[num]
            .agg(agg)
            .sort_index()
        )

        if len(ts) < 3:
            continue

        # Trend detection
        x = np.arange(len(ts))
        y = ts.values
        slope = np.polyfit(x, y, 1)[0]

        # Direction
        if slope > 0:
            direction = "increasing"
        elif slope < 0:
            direction = "decreasing"
        else:
            direction = "flat"

        # Strength
        rel_change = np.abs((y[-1] - y[0]) / y[0]) if y[0] != 0 else 0

        if rel_change > 0.3:
            strength = "strong"
        elif rel_change > 0.1:
            strength = "moderate"
        else:
            strength = "weak"

        # Volatility
        cv = np.std(y) / np.mean(y) if np.mean(y) != 0 else 0

        if cv > 0.5:
            volatility ="high"
        elif cv > 0.2:
            volatility ="moderate"
        else:
            volatility ="low"

        line = f"pair: {time_column} vs {num} | aggregation: {agg} | granularity: monthly | trend: {direction} | trend_strength: {strength} | volatility: {volatility}"

        lines.append(line)

    return lines


def multivariate_numeric(df):
    lines = []

    corr = df.corr(numeric_only=True)

    numeric_cols = corr.columns

    for i, j in itertools.combinations(numeric_cols, 2):
        value = corr.loc[i, j]

        if pd.isna(value):
            continue

        abs_val = np.abs(value)
        threshold_strong=0.7
        threshold_moderate=0.4
        if abs_val >= threshold_strong:
            level = "strong"
        elif abs_val >= threshold_moderate:
            level = "moderate"
        else:
            level="weak"

        direction = "positive" if value > 0 else "negative"

        redundancy = "likely" if abs_val >= threshold_strong else "possible"

        line = (
            f"multivariate_pair: {i} & {j} | "
            f"correlation: {round(value,3)} | "
            f"direction: {direction} | "
            f"strength: {level} | "
            f"redundancy: {redundancy} | "
            f"causation: not_established"
        )

        lines.append(line)

    if not lines:
        lines.append(
            "multivariate_summary: no strong or moderate correlations detected among numeric features"
        )

    return lines


# ══════════════════════════════════════════════
#  Plotting Functions
# ══════════════════════════════════════════════

def plot_univariate_numeric(df,numerical_column):
        plt.figure(figsize=(10,6))
        sns.histplot(df[numerical_column],kde=True)
        plt.title(f'Distribution of {numerical_column}', fontsize=15)
        plt.show()

def plot_univariate_time(df,col):
    plt.figure(figsize=(10,6))
    time_series = df[col].value_counts().sort_index()
    sns.lineplot(x=time_series.index,y=time_series.values,marker='o')
    plt.title(f'Distribution of {col}', fontsize=15)
    plt.show()


def plot_bivariate_category_vs_numeric(df,categorical_column,numerical_column,plot):
    '''for cat in categorical_columns:
        for num in numerical_columns:
            plt.figure(figsize=(10,6))
            sns.boxplot(data=df, x=cat, y=num, showmeans=True, 
            palette='coolwarm', hue=cat,meanprops={"marker":"o", "markerfacecolor":"white", "markeredgecolor":"black"})
            plt.xlabel(cat)
            plt.ylabel(num)
            plt.xticks(rotation=45,ha='right')
            plt.tight_layout()
            plt.show()

            plt.figure(figsize=(10,6))
            sns.violinplot(data=df,x=cat,y=num,palette='coolwarm', hue=cat)
            plt.xlabel(cat)
            plt.ylabel(num)
            plt.xticks(rotation=45,ha='right')
            plt.tight_layout()
            plt.show()
            
            plt.figure(figsize=(10,6))
            sns.barplot(data=df,x=cat,y=num,palette='coolwarm', hue=cat)
            plt.xlabel(cat)
            plt.ylabel(num)
            plt.xticks(rotation=45,ha='right')
            plt.tight_layout()
            plt.show()'''
    if plot in("box_plot","boxplot"):
        plt.figure(figsize=(10,6))
        sns.boxplot(data=df, x=categorical_column, y=numerical_column, #showmeans=True, 
        palette='coolwarm', hue=categorical_column,meanprops={"marker":"o", "markerfacecolor":"white", "markeredgecolor":"black"})
        plt.xlabel(categorical_column)
        plt.ylabel(numerical_column)
        plt.xticks(rotation=45,ha='right')
        plt.title(f"{categorical_column} vs {numerical_column} Analysis")
        plt.tight_layout()
        plt.show()

    if plot in("violin_plot","violinplot"):
        plt.figure(figsize=(10,6))
        sns.violinplot(data=df,x=categorical_column,y=numerical_column,palette='coolwarm', hue=categorical_column)
        plt.xlabel(categorical_column)
        plt.ylabel(numerical_column)
        plt.xticks(rotation=45,ha='right')
        plt.title(f"{categorical_column} vs {numerical_column} Analysis")
        plt.tight_layout()
        plt.show()

    if plot in("bar_plot","barplot"):
        plt.figure(figsize=(10,6))
        sns.barplot(data=df,x=categorical_column,y=numerical_column,palette='coolwarm', hue=categorical_column)
        plt.xlabel(categorical_column)
        plt.ylabel(numerical_column)
        plt.xticks(rotation=45,ha='right')
        plt.title(f"{categorical_column} vs {numerical_column} Analysis")
        plt.tight_layout()
        plt.show()

def plot_bivariate_category_vs_category(df,col,col2,plot_type):
    '''for col in categorical_columns:
        for col2 in reversed(categorical_columns):'''
    if plot_type in("scatter_plot","scatterplot"):
        plt.figure(figsize=(10,6))
        sns.jointplot(data=df, x=col, y=col2)
        plt.xlabel(col)
        plt.ylabel(col2)
        plt.xticks(rotation=45,ha='right')
        plt.tight_layout()
        plt.show()

    if plot_type in("stacked_bar_plot","stacked bar plot","stackedbar_plot","stacked_bar plot"):
        plt.figure(figsize=(10,6))
        so.Plot(df,x=col, y=col2).add(so.Bar(), so.Agg())
        plt.xlabel(col)
        plt.ylabel(col2)
        plt.xticks(rotation=45,ha='right')
        plt.tight_layout()
        plt.show()

    if plot_type in("grouped_bar_plot","grouped bar plot","groupedbar_plot","grouped_bar plot"):
        plt.figure(figsize=(10,6))
        so.Plot(df, x=col, y=col2).add(so.Bar(),so.Agg(),so.Dodge())
        plt.xlabel(col)
        plt.ylabel(col2)
        plt.xticks(rotation=45,ha='right')
        plt.tight_layout()
        plt.show()

    if plot_type in("heat_map","heatmap"):
        plt.figure(figsize=(10,6))
        sns.heatmap(df,col,col2)
        plt.xlabel(col)
        plt.ylabel(col2)
        plt.tight_layout()
        plt.show()

def plot_bivariate_numeric_vs_numeric(df,col,col2,plot):
    #scatter-plot,kex-plot,line-plot
    if plot in("scatter_plot","scatterplot"):
        plt.Figure(figsize=(10,6))
        sns.scatterplot(df,x=col,y=col2)
        plt.show()
    if plot in("line_plot","lineplot"):
        plt.Figure(figsize=(10,6))
        sns.lineplot(df,x=col,y=col2)
        plt.xlabel(col)
        plt.ylabel(col2)
        plt.show()

def plot_bivariate_time_numeric(df,time_column,numerical_column,plot):
    if plot in("line_plot","lineplot","line"):
        plt.Figure(figsize=(30,10))
        sns.lineplot(df,x=time_column,y=numerical_column)
        plt.xlabel(time_column)
        plt.ylabel(numerical_column)
        plt.title(f"{time_column} vs {numerical_column} Analsis")
        plt.xticks(rotation=45,ha='right')
        plt.show()

def plot_bivariate_categorical_vs_time(df,col,col2,plot):
    if plot in("line_plot","lineplot","line"):
            plt.Figure(figsize=(30,10))
            sns.lineplot(df,x=col,y=col2)
            plt.xlabel(col)
            plt.ylabel(col2)
            plt.title(f"{col} vs {col2} Analsis")
            plt.xticks(rotation=45,ha='right')
            plt.show()

def plot_multivariate(df,c1,c2,c3,plot_type):
    if plot_type in ("scatter_plot_with_hue","scatter_plot","scatter plot"):
        sns.scatterplot(df,x=df[c1],y=df[c2],hue=df[c3],legend=True)


# ══════════════════════════════════════════════
#  LLM Dashboard Config
# ══════════════════════════════════════════════

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

def take_llm_decision(columns):
    prompt_body = (
        f'''
Act as a Senior Lead Data Analyst with 10+ years of experience in enterprise-scale business intelligence. I am providing you with a raw schema: {columns}. 
Your objective is to design a high-impact diagnostic dashboard. Do not simply list all possible plots; curate the most strategically significant visualizations that drive decision-making.
LOGIC RULES:
1. Identify Data Types: Categorize columns into Numeric, Categorical, or Time-series.
2. Mapping:
   - Univariate Numeric: Focus on distributions (Histogram/Boxplot).
   - Univariate Categorical: Focus on frequency/proportions (Bar/Pie).
   - Univariate Time: Focus on event frequency over time (Line/Area).
   - Bivariate Numeric vs. Numeric: Focus on correlation (Scatter).
   - Bivariate Categorical vs. Numeric: Focus on group comparison (Box/Violin/Bar).
   - Bivariate Categorical vs. Categorical: Focus on relationship density (Heatmap/Stacked Bar).
   - Bivariate Numeric vs. Time: Focus on trends over time (Line).
   - Multivariate: Use exactly three columns (X, Y, and a Hue).
OUTPUT SPECIFICATIONS:
- Return ONLY a Python tuple of dictionaries.
- Schema: ["visualization": "Title", "col vs col": "c1,c2", "plot_type": "type ex: line_plot or scatter_plot or ", "function": "name"] here i used square brackets instead of curly just for me , but you have to return in given schema in Python tuple of dictionaries.
- Use a single comma (no spaces) as the delimiter in "col vs col".
- 'function' MUST be one of: plot_univariate_numeric, plot_univariate_categorical, plot_univariate_time, plot_bivariate_numeric_vs_numeric, plot_bivariate_categorical_vs_numerical, plot_bivariate_categorical_vs_categorical, plot_bivariate_numeric_vs_time, or plot_multivariate.
- Exclude unique ID columns from numeric axes; use them only for counts.'''
    )
    result = call_llm(prompt_body)
    return result


def process_dashboard_config(result):
    """
    Refined parser for Senior-Level Data Analyst output.
    Handles Tuple-strings, List-strings, and JSON objects safely.
    """
    data = []

    if isinstance(result, str):
        # 1. Clean the string (LLMs often wrap code in triple backticks)
        cleaned_result = result.strip()
        if cleaned_result.startswith("```"):
            cleaned_result = re.sub(r'^```[a-zA-Z]*\n|```$', '', cleaned_result, flags=re.MULTILINE).strip()

        try:
            # 2. Try parsing as a Python Literal (Tuple/List)
            # This is safer than eval() and matches your 'tuple' requirement
            data = ast.literal_eval(cleaned_result)
        except (ValueError, SyntaxError):
            try:
                # 3. Fallback: Try parsing as JSON
                data = json.loads(cleaned_result)
            except json.JSONDecodeError:
                # 4. Emergency Regex: If LLM gives a messy text format, 
                # extract anything that looks like a dictionary
                dict_strings = re.findall(r'\{.*?\}', cleaned_result, re.DOTALL)
                data = [ast.literal_eval(d) for d in dict_strings]
    else:
        # If result is already a Python object (tuple/list)
        data = result

    # Standardize output to a list for iteration
    return list(data) if isinstance(data, (tuple, list)) else [data]


# ══════════════════════════════════════════════
#  EDA Pipeline Orchestrator
# ══════════════════════════════════════════════

def run_statistical_eda(df):
    """
    Run the full statistical EDA pipeline on any DataFrame.
    Returns a dict of all insights.
    """
    logger.info(f"Running EDA on dataset: {df.shape}")

    numerical_columns, categorical_columns, time_column = classify_columns(df)

    insights = {
        "shape": {"rows": df.shape[0], "columns": df.shape[1]},
        "column_types": {col: str(dtype) for col, dtype in column_types_raw(df).items()},
        "missing_values": missing_value_summary(df),
        "column_classification": {
            "numerical": numerical_columns,
            "categorical": categorical_columns,
            "time": time_column,
        },
    }

    # ── Univariate ──
    if numerical_columns:
        try:
            insights["univariate_numerical_summary"] = univariate_numerical_summary(df, numerical_columns).to_dict()
        except Exception as e:
            logger.error(f"univariate_numerical_summary error: {e}")

        try:
            insights["univariate_numeric"] = univariate_numeric(df, numerical_columns)
        except Exception as e:
            logger.error(f"univariate_numeric error: {e}")

    if categorical_columns:
        try:
            insights["univariate_categorical"] = univariate_category(df, categorical_columns)
        except Exception as e:
            logger.error(f"univariate_category error: {e}")

    if time_column:
        try:
            insights["univariate_time"] = univariate_time(df, time_column)
        except Exception as e:
            logger.error(f"univariate_time error: {e}")

    # ── Bivariate ──
    if numerical_columns and categorical_columns:
        try:
            insights["bivariate_numeric_vs_category"] = bivariate_numeric_vs_category(df, numerical_columns, categorical_columns)
        except Exception as e:
            logger.error(f"bivariate_numeric_vs_category error: {e}")

    if len(numerical_columns) >= 2:
        try:
            insights["bivariate_numeric_vs_numeric"] = bivariate_numeric_vs_numeric(df, numerical_columns)
        except Exception as e:
            logger.error(f"bivariate_numeric_vs_numeric error: {e}")

    if len(categorical_columns) >= 2:
        try:
            insights["bivariate_category_vs_category"] = bivariate_category_vs_category(df, categorical_columns)
        except Exception as e:
            logger.error(f"bivariate_category_vs_category error: {e}")

    if time_column and numerical_columns:
        for tc in time_column:
            try:
                insights[f"bivariate_time_vs_numeric_{tc}"] = bivariate_time_vs_numeric(df, tc, numerical_columns)
            except Exception as e:
                logger.error(f"bivariate_time_vs_numeric error for {tc}: {e}")

    # ── Multivariate ──
    if len(numerical_columns) >= 2:
        try:
            insights["multivariate_numeric"] = multivariate_numeric(df)
        except Exception as e:
            logger.error(f"multivariate_numeric error: {e}")

    logger.info(f"EDA complete — {len(insights)} insight groups generated")
    return insights


def run_dashboard_generation(df):
    """
    Ask the LLM to pick the best visualizations, then generate them.
    """
    logger.info("Generating LLM-powered dashboard config...")
    result = take_llm_decision(df.columns.tolist())
    if result is None:
        logger.error("LLM dashboard generation failed")
        return []

    dashboard_data = process_dashboard_config(result)
    logger.info(f"Dashboard config: {len(dashboard_data)} visualizations")

    for viz in dashboard_data:
        try:
            func = viz.get("function", "")
            cols = viz.get("col vs col", "")

            if func == "plot_univariate_numeric":
                plot_univariate_numeric(df, cols)
            elif func == "plot_univariate_categorical":
                univariate_category(df, [cols])
            elif func == "plot_univariate_time":
                plot_univariate_time(df, cols)
            elif func == "plot_bivariate_categorical_vs_numerical":
                c1, c2 = cols.split(",")
                plot_bivariate_category_vs_numeric(df, c1, c2, viz.get("plot_type", "boxplot"))
            elif func == "plot_bivariate_numeric_vs_numeric":
                c1, c2 = cols.split(",")
                plot_bivariate_numeric_vs_numeric(df, c1, c2, viz.get("plot_type", "scatter_plot"))
            elif func in ("plot_bivariate_time_vs_numeric", "plot_bivariate_numeric_vs_time"):
                c1, c2 = cols.split(",")
                plot_bivariate_time_numeric(df, c1, c2, viz.get("plot_type", "line_plot"))
            elif func in ("plot_bivariate_time_vs_categorical", "plot_bivariate_categorical_vs_time"):
                c1, c2 = cols.split(",")
                plot_bivariate_categorical_vs_time(df, c1, c2, viz.get("plot_type", "line_plot"))
            elif func == "plot_multivariate":
                parts = cols.split(",")
                if len(parts) >= 3:
                    plot_multivariate(df, parts[0], parts[1], parts[2], viz.get("plot_type", "scatter_plot"))
        except Exception as e:
            logger.error(f"Dashboard viz error for {viz}: {e}")

    return dashboard_data


# ══════════════════════════════════════════════
#  EDAEngine Class
# ══════════════════════════════════════════════

class EDAEngine:
    """
    Automated EDA engine for any structured dataset.

    Usage:
        engine = EDAEngine(df)
        insights = engine.run_full_eda()
    """

    def __init__(self, df: pd.DataFrame):
        logger.info(f"Initializing EDAEngine | shape={df.shape}")
        self.df = df
        self.numerical_columns, self.categorical_columns, self.time_columns = classify_columns(df)
        self.insights: Dict[str, Any] = {}

    def run_full_eda(self) -> Dict[str, Any]:
        """Run all statistical analysis and return insights dict."""
        self.insights = run_statistical_eda(self.df)
        return self.insights

    def run_dashboard(self) -> list:
        """Ask LLM to pick best visualizations and generate them."""
        return run_dashboard_generation(self.df)

    def get_insights(self) -> Dict[str, Any]:
        """Return cached insights (run run_full_eda first)."""
        if not self.insights:
            self.run_full_eda()
        return self.insights


# ─────────────────────────────────────────────
# CLI Entry Point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python eda_engine.py <csv_file>")
        sys.exit(1)

    csv_path = sys.argv[1]
    df = pd.read_csv(csv_path)

    engine = EDAEngine(df)
    insights = engine.run_full_eda()

    print(json.dumps(insights, indent=2, default=str))