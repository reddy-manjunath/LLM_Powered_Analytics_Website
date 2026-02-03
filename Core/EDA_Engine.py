# Generated from: EDA_Engine.ipynb
# Converted at: 2026-02-03T15:58:21.938Z
# Next step (optional): refactor into modules & generate tests with RunCell
# Quick start: pip install runcell

# # EDA Engine


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gaussian_kde
import seaborn.objects as so

df = pd.read_csv("cleaned_sample.csv",parse_dates=["order_date"])

df=df.drop(columns=["Unnamed: 0"])

df.dtypes

df["order_date"].dtype

numerical_columns=[]
categorical_columns=[]
time_column=[]
for col in df.columns:
    if pd.api.types.is_numeric_dtype(df[col]):
        numerical_columns.append(col)
    elif (df[col]).dtype=='M8[us]':
        time_column.append(col)
    else:
        categorical_columns.append(col)

print(numerical_columns,categorical_columns,time_column)
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
    shape={}
    if i==0:
        return "rows:"+df.shape[0]
    elif i==1:
        return "columns:"+df.shape[1]
    else:
        shape["Rows"]=df.shape[0]
        shape["columns"]=df.shape[1]
        return shape

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


print(numerical_columns)

print(univariate_numerical_summary(df,numerical_columns))

def univariate_numeric(df,numerical_columns):
    list1={}
    if len(df) < 1:
        return {
            "distribution_type": "insufficient_data",
            "description": "Too few data points to determine distribution reliably."
        }
    for col in numerical_columns:
        skewness=df[col].skew()
        kurt=df[col].kurt()
        q1,q3 = df[col].quantile([0.25, 0.75])
        iqr = q3-q1
        value_range = df[col].max()-df[col].min()
        concentrated = iqr < 0.25 * value_range

        kde = gaussian_kde(df[col])
        xs = np.linspace(df[col].min(), df[col].max(), 200)
        density = kde(xs)
        peaks = ((density[1:-1] > density[:-2]) & (density[1:-1] > density[2:])).sum()

        if abs(skewness) < 0.5:
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
            

        list1[col]=skew_desc,tail_desc,modality,
    return list1


print(univariate_numeric(df,numerical_columns))

def univariate_time(df,time_column):
    list1={}
    for col in time_column:
        count=df[col].value_counts()
        dates=count.index
        value_count=count.values
        list1[col]=dates,value_count
    return list1

print(univariate_time(df,time_column))

def univariate_category(df,categorical_columns):
    list1={}
    for col in categorical_columns:
        count=df[col].value_counts()
        dates=count.index
        value_count=count.values
        dominant=df[col].value_counts().idxmax()
        list1[col]=dates,value_count,dominant
    return list1
    


print(univariate_category(df,categorical_columns))

def bivariate_numeric_vs_category(df,numerical_columns,categorical_columns):
    results={}
    for cat in categorical_columns:
        for num in numerical_columns:
            summary = df.groupby(cat)[num].agg(['sum', 'mean', 'median', 'std', 'count'])
            results[f"{num}_vs_{cat}"] = summary
    return results

print(bivariate_numeric_vs_category(df,numerical_columns,categorical_columns))

def bivariate_numeric_vs_numeric(df,numerical_columns):
    list_corr={}
    list_cov={}
    for num in numerical_columns:
        for num2 in numerical_columns:
            if num!=num2:
                value_corr=df[num].corr(df[num2])
                value_cov=df[num].cov(df[num2])
                list_corr[f"{num}_vs_{num2}"]=value_corr
                list_cov[f"{num}_vs_{num2}"]=value_cov
    return list_corr,list_cov

print(bivariate_numeric_vs_numeric(df,numerical_columns))

def bivariate_category_vs_category(df,categorical_columns):
    list1={}
    for col in categorical_columns:
        list2=[pd.crosstab(df[col], df[col])]
        list1[f"{col}_vs_{col}"]=list2
    return list1

print(bivariate_category_vs_category(df,categorical_columns))

def bivariate_time_vs_numeric(df,time_column,numerical_columns):
    list1={}
    for time in time_column:
        for num in numerical_columns:
            summary=df.groupby(time)[num].agg('sum')
            list1[f"{time}_vs_{num}"]=summary
    return list1


print(bivariate_time_vs_numeric(df,time_column,numerical_columns))

def multivariate(df):
    list1=[df.corr(numeric_only=True)]
    return list1

print(multivariate(df))

# ## Plots for user


def plot_univariate_numeric(df,numerical_columns):
    for col in numerical_columns:
        plt.figure()
        sns.histplot(df[col])
        plt.title(col)
        plt.show()

plot_univariate_numeric(df,numerical_columns)

def plot_univariate_category(df,categorical_columns):
    sns.set_theme(style="darkgrid")
    for col in categorical_columns:
        plt.figure(figsize=(10, 6))
        sns.countplot(data=df, x=col, palette='coolwarm', hue=col, legend=False)
        plt.title(f'Distribution of {col}', fontsize=15)
        plt.xlabel(col, fontsize=12)
        plt.ylabel('Count', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()

plot_univariate_category(df,categorical_columns)

def univariate_time(df,time_column):
    for col in time_column:
        plt.Figure(figsize=(10,6))
        sns.countplot(data=df,x=col,palette='coolwarm', hue=col)
        plt.title(col)
        plt.xlabel(col)
        plt.xticks(rotation=45,ha='right')
        plt.tight_layout()
        plt.show()

univariate_time(df,time_column)

def plot_bivariate_numeric_vs_category(df,numerical_columns,categorical_columns):
    for cat in categorical_columns:
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
            plt.show()

plot_bivariate_numeric_vs_category(df,numerical_columns,categorical_columns)

def plot_bivariate_category_vs_category(df,categorical_columns):
    for col in categorical_columns:
        for col2 in reversed(categorical_columns):
            plt.figure(figsize=(10,6))
            sns.jointplot(data=df, x=col, y=col2)
            plt.xlabel(col)
            plt.ylabel(col2)
            plt.xticks(rotation=45,ha='right')
            plt.tight_layout()
            plt.show()

            plt.figure(figsize=(10,6))
            so.Plot(df,x=col, y=col2).add(so.Bar(), so.Agg())
            plt.xlabel(col)
            plt.ylabel(col2)
            plt.xticks(rotation=45,ha='right')
            plt.tight_layout()
            plt.show()

            plt.figure(figsize=(10,6))
            so.Plot(df, x=col, y=col2).add(so.Bar(),so.Agg(),so.Dodge())
            plt.xlabel(col)
            plt.ylabel(col2)
            plt.xticks(rotation=45,ha='right')
            plt.tight_layout()
            plt.show()
            
    for col in categorical_columns:
        sns.heatmap(pd.crosstab(df[col], df[col]),annot=True)

plot_bivariate_category_vs_category(df,categorical_columns)

def plot_bivariate_numeric_vs_numeric(df,numerical_columns):
    #scatter-plot,kex-plot,line-plot
    for col in numerical_columns:
        for col2 in reversed(numerical_columns):
            if col!=col2:
                plt.Figure(figsize=(10,6))
                sns.scatterplot(df,x=col,y=col2)
                plt.xlabel(col)
                plt.ylabel(col2)
                plt.show()

                plt.Figure(figsize=(10,6))
                sns.lineplot(df,x=col,y=col2)
                plt.xlabel(col)
                plt.ylabel(col2)
                plt.show()

plot_bivariate_numeric_vs_numeric(df,numerical_columns)

def plot_bivariate_time_numeric(df,time_column,numerical_columns):
    for col in time_column:
        for col2 in numerical_columns:
            plt.Figure(figsize=(30,10))
            sns.lineplot(df,x=col,y=col2)
            plt.xlabel(col)
            plt.ylabel(col2)
            plt.xticks(rotation=45,ha='right')
            plt.show()

plot_bivariate_time_numeric(df,time_column,numerical_columns)

def plot_multivariate(df):
    plt.Figure(figsize=(10,6))
    sns.heatmap(df.corr(numeric_only=True),annot=True)
    plt.show()

plot_multivariate(df)