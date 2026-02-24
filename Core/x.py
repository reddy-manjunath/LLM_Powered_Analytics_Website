import json
import os

notebook_path = "c:/Users/Manju/OneDrive/Desktop/Analytics_Website/Core/EDA_Engine.ipynb"

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

new_code = [
    "def save_eda_summary(df, numerical_columns, categorical_columns, time_column=None, output_file=\"eda_summary.txt\"):\n",
    "    with open(output_file, \"w\") as f:\n",
    "        f.write(\"--- Dataset Shape ---\\n\")\n",
    "        try:\n",
    "            f.write(f\"Rows: {dataset_shape(df, 0)}\\n\")\n",
    "            f.write(f\"Columns: {dataset_shape(df, 1)}\\n\")\n",
    "        except Exception as e:\n",
    "            f.write(f\"Error: {e}\\n\")\n",
    "        f.write(\"\\n--- Column Types ---\\n\")\n",
    "        try:\n",
    "            types = column_types_raw(df)\n",
    "            for col, dtype in types.items():\n",
    "                f.write(f\"{col}: {dtype}\\n\")\n",
    "        except Exception as e:\n",
    "            f.write(f\"Error: {e}\\n\")\n",
    "        f.write(\"\\n--- Missing Values ---\\n\")\n",
    "        try:\n",
    "            missing = missing_value_summary(df)\n",
    "            for col, val in missing.items():\n",
    "                f.write(f\"{col}: {val}%\\n\")\n",
    "        except Exception as e:\n",
    "            f.write(f\"Error: {e}\\n\")\n",
    "        f.write(\"\\n--- Univariate Numeric ---\\n\")\n",
    "        try:\n",
    "            uni_num = univariate_numeric(df, numerical_columns)\n",
    "            for line in uni_num:\n",
    "                f.write(f\"{line}\\n\")\n",
    "        except Exception as e:\n",
    "            f.write(f\"Error: {e}\\n\")\n",
    "        f.write(\"\\n--- Univariate Categorical ---\\n\")\n",
    "        try:\n",
    "            uni_cat = univariate_category(df, categorical_columns)\n",
    "            for line in uni_cat:\n",
    "                f.write(f\"{line}\\n\")\n",
    "        except Exception as e:\n",
    "            f.write(f\"Error: {e}\\n\")\n",
    "        f.write(\"\\n--- Bivariate Numeric vs Category ---\\n\")\n",
    "        try:\n",
    "            bi_num_cat = bivariate_numeric_vs_category(df, numerical_columns, categorical_columns)\n",
    "            for line in bi_num_cat:\n",
    "                f.write(f\"{line}\\n\")\n",
    "        except Exception as e:\n",
    "            f.write(f\"Error: {e}\\n\")\n",
    "        f.write(\"\\n--- Bivariate Numeric vs Numeric ---\\n\")\n",
    "        try:\n",
    "            bi_num_num = bivariate_numeric_vs_numeric(df, numerical_columns)\n",
    "            for line in bi_num_num:\n",
    "                f.write(f\"{line}\\n\")\n",
    "        except Exception as e:\n",
    "            f.write(f\"Error: {e}\\n\")\n",
    "        f.write(\"\\n--- Bivariate Category vs Category ---\\n\")\n",
    "        try:\n",
    "            bi_cat_cat = bivariate_category_vs_category(df, categorical_columns)\n",
    "            for line in bi_cat_cat:\n",
    "                f.write(f\"{line}\\n\")\n",
    "        except Exception as e:\n",
    "            f.write(f\"Error: {e}\\n\")\n",
    "        if time_column:\n",
    "            f.write(\"\\n--- Bivariate Time vs Numeric ---\\n\")\n",
    "            try:\n",
    "                bi_time_num = bivariate_time_vs_numeric(df, time_column, numerical_columns)\n",
    "                for line in bi_time_num:\n",
    "                    f.write(f\"{line}\\n\")\n",
    "            except Exception as e:\n",
    "                f.write(f\"Error: {e}\\n\")\n",
    "        f.write(\"\\n--- Multivariate Numeric ---\\n\")\n",
    "        try:\n",
    "            multi_num = multivariate_numeric(df)\n",
    "            for line in multi_num:\n",
    "                f.write(f\"{line}\\n\")\n",
    "        except Exception as e:\n",
    "            f.write(f\"Error: {e}\\n\")\n",
    "\n",
    "try:\n",
    "    t_col = time_column if 'time_column' in globals() else None\n",
    "    if not t_col and 'order_date' in df.columns:\n",
    "        t_col = 'order_date'\n",
    "    save_eda_summary(df, numerical_columns, categorical_columns, t_col)\n",
    "    print(\"EDA summary saved to eda_summary.txt\")\n",
    "except Exception as e:\n",
    "    print(f\"Error generating summary: {e}\")"
]

new_cell = {
   "cell_type": "code",
   "execution_count": None,
   "id": "generate_summary_1",
   "metadata": {},
   "outputs": [],
   "source": new_code
}

nb['cells'].append(new_cell)

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Notebook modified successfully.")
