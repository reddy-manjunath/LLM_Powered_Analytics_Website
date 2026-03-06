"""
ai_engine.py — AI-Powered Data Analytics Engine
================================================
Agent-based reasoning layer using LangGraph + local Ollama LLM.
Analyzes any structured dataset via natural language questions.

Usage:
    from ai_engine import AIEngine

    engine = AIEngine(df)
    result = engine.ask("What is the average income?")
    print(result)
"""

import logging
import json
import os
import re
import uuid
from typing import Optional, Dict, Any, List
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server use
import matplotlib.pyplot as plt
import seaborn as sns

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END
from typing_extensions import TypedDict


# ─────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("ai_engine")


# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────

CHARTS_DIR = Path("charts")
CHARTS_DIR.mkdir(exist_ok=True)

MAX_SAMPLE_ROWS = 5          # Rows shown to LLM as context
MAX_CONTEXT_COLUMNS = 60     # Truncate column list if dataset is very wide


# ══════════════════════════════════════════════
#  Agent State
# ══════════════════════════════════════════════

class AgentState(TypedDict):
    question: str
    dataset_info: str
    reasoning: str
    selected_tool: str
    tool_input: dict
    tool_output: str
    answer: str
    chart_path: Optional[str]
    error: Optional[str]


# ══════════════════════════════════════════════
#  Tool 1 — Dataset Metadata
# ══════════════════════════════════════════════

class DatasetMetadataTool:
    """Provide information about dataset structure, dtypes, missing values, and statistics."""

    def __init__(self, df: pd.DataFrame):
        self.df = df

    def run(self, **kwargs) -> str:
        """Return structured metadata about every column."""
        info: Dict[str, Any] = {
            "shape": {"rows": len(self.df), "columns": len(self.df.columns)},
            "columns": {},
            "memory_usage_mb": round(
                self.df.memory_usage(deep=True).sum() / (1024 * 1024), 2
            ),
        }

        for col in self.df.columns:
            col_info: Dict[str, Any] = {
                "dtype": str(self.df[col].dtype),
                "missing_values": int(self.df[col].isna().sum()),
                "missing_pct": round(self.df[col].isna().mean() * 100, 1),
                "unique_values": int(self.df[col].nunique()),
            }

            if pd.api.types.is_numeric_dtype(self.df[col]):
                desc = self.df[col].describe()
                col_info["statistics"] = {
                    k: round(float(desc[k]), 2) for k in desc.index if k != "count"
                }
            elif self.df[col].dtype == "object":
                col_info["top_values"] = (
                    self.df[col].value_counts().head(5).to_dict()
                )

            info["columns"][col] = col_info

        return json.dumps(info, indent=2, default=str)


# ══════════════════════════════════════════════
#  Tool 2 — Dynamic Data Analysis
# ══════════════════════════════════════════════

class DataAnalysisTool:
    """Perform dynamic calculations: groupby, filter, ranking, percentiles, etc."""

    SUPPORTED_OPS = (
        "describe", "value_counts", "groupby", "filter",
        "correlation", "percentile", "ranking", "average", "count",
    )

    def __init__(self, df: pd.DataFrame):
        self.df = df

    # ── helpers ──

    def validate_columns(self, cols: List[str]) -> Optional[str]:
        invalid = [c for c in cols if c not in self.df.columns]
        if invalid:
            return json.dumps({
                "error": f"Columns not found: {invalid}. Available: {list(self.df.columns)}"
            })
        return None

    # ── main entry ──

    def run(
        self,
        operation: str,
        column: str = None,
        columns: List[str] = None,
        group_by: str = None,
        agg_func: str = "mean",
        filter_condition: str = None,
        top_n: int = 10,
        **kwargs,
    ) -> str:
        """Execute an analytical operation on the dataframe."""
        try:
            df = self.df

            # Column validation
            check_cols = [c for c in [column, group_by] if c]
            if columns:
                check_cols.extend(columns)
            if check_cols:
                err = self.validate_columns(check_cols)
                if err:
                    return err

            # ── describe ──
            if operation == "describe":
                target = df[column] if column else df
                return target.describe(include="all").to_json(default_handler=str)

            # ── value_counts ──
            elif operation == "value_counts":
                if not column:
                    return json.dumps({"error": "'column' is required for value_counts"})
                return df[column].value_counts().head(top_n).to_json()

            # ── groupby ──
            elif operation == "groupby":
                if not group_by:
                    return json.dumps({"error": "'group_by' is required"})
                if not column:
                    nums = df.select_dtypes(include=[np.number]).columns.tolist()
                    if not nums:
                        return json.dumps({"error": "No numeric columns for aggregation"})
                    column = nums[0]
                agg_map = {
                    "mean": "mean", "sum": "sum", "count": "count",
                    "min": "min", "max": "max", "median": "median",
                }
                func = agg_map.get(agg_func, "mean")
                result = df.groupby(group_by)[column].agg(func)
                result = result.sort_values(ascending=False).head(top_n)
                return result.to_json(default_handler=str)

            # ── filter ──
            elif operation == "filter":
                if not filter_condition:
                    return json.dumps({"error": "'filter_condition' required (pandas query syntax)"})
                # Basic sanitisation – block dangerous tokens
                BLOCKED = ["import", "exec", "eval", "__", "os.", "sys.", "open("]
                if any(b in filter_condition.lower() for b in BLOCKED):
                    return json.dumps({"error": "Unsafe filter blocked"})
                filtered = df.query(filter_condition)
                return json.dumps({
                    "matching_rows": len(filtered),
                    "total_rows": len(df),
                    "percentage": round(len(filtered) / len(df) * 100, 2),
                    "sample": filtered.head(5).to_dict(orient="records"),
                }, default=str)

            # ── correlation ──
            elif operation == "correlation":
                numeric = df.select_dtypes(include=[np.number])
                if column:
                    corr = numeric.corrwith(df[column]).sort_values(ascending=False)
                    return corr.to_json()
                return numeric.corr().to_json()

            # ── percentile ──
            elif operation == "percentile":
                if not column:
                    return json.dumps({"error": "'column' required for percentile"})
                pcts = [0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
                return df[column].quantile(pcts).to_json()

            # ── ranking ──
            elif operation == "ranking":
                if not column:
                    return json.dumps({"error": "'column' required for ranking"})
                if group_by:
                    result = (
                        df.groupby(group_by)[column]
                        .sum()
                        .sort_values(ascending=False)
                        .head(top_n)
                    )
                else:
                    show_cols = columns if columns else [column]
                    result = df.nlargest(top_n, column)[show_cols]
                return result.to_json(default_handler=str)

            # ── average / mean ──
            elif operation in ("average", "mean"):
                if column:
                    val = df[column].mean()
                    return json.dumps({"column": column, "mean": round(float(val), 2)})
                return df.select_dtypes(include=[np.number]).mean().to_json()

            # ── count ──
            elif operation == "count":
                if filter_condition:
                    BLOCKED = ["import", "exec", "eval", "__", "os.", "sys.", "open("]
                    if any(b in filter_condition.lower() for b in BLOCKED):
                        return json.dumps({"error": "Unsafe filter blocked"})
                    count = len(df.query(filter_condition))
                    return json.dumps({"count": count, "condition": filter_condition})
                elif column:
                    return json.dumps({
                        "column": column,
                        "non_null_count": int(df[column].count()),
                        "unique": int(df[column].nunique()),
                    })
                return json.dumps({"total_rows": len(df)})

            else:
                return json.dumps({
                    "error": f"Unknown operation: {operation}",
                    "supported": list(self.SUPPORTED_OPS),
                })

        except Exception as e:
            logger.error(f"DataAnalysisTool error: {e}")
            return json.dumps({"error": str(e)})


# ══════════════════════════════════════════════
#  Tool 3 — Visualization
# ══════════════════════════════════════════════

class VisualizationTool:
    """Generate charts: histogram, bar, scatter, box, line, heatmap, pie."""

    SUPPORTED_CHARTS = (
        "histogram", "bar", "scatter", "box", "line", "heatmap", "pie",
    )

    def __init__(self, df: pd.DataFrame, charts_dir: Path = CHARTS_DIR):
        self.df = df
        self.charts_dir = charts_dir
        self.charts_dir.mkdir(exist_ok=True)

    def save(self, fig: plt.Figure, chart_type: str, label: str) -> str:
        """Save figure and return JSON result."""
        uid = str(uuid.uuid4())[:8]
        filename = f"{chart_type}_{label}_{uid}.png"
        filepath = self.charts_dir / filename
        fig.savefig(filepath, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Chart saved: {filepath}")
        return json.dumps({
            "chart_type": chart_type,
            "file_path": str(filepath),
            "message": f"Chart generated: {filename}",
        })

    def run(
        self,
        chart_type: str,
        column: str = None,
        columns: List[str] = None,
        group_by: str = None,
        title: str = None,
        **kwargs,
    ) -> str:
        """Generate and save a chart, return its file path."""
        try:
            # Validate columns exist
            for c in [column, group_by]:
                if c and c not in self.df.columns:
                    return json.dumps({"error": f"Column '{c}' not found in dataset"})
            if columns:
                for c in columns:
                    if c not in self.df.columns:
                        return json.dumps({"error": f"Column '{c}' not found in dataset"})

            fig, ax = plt.subplots(figsize=(10, 6))

            # ── histogram ──
            if chart_type == "histogram":
                if not column:
                    return json.dumps({"error": "'column' required for histogram"})
                self.df[column].dropna().hist(ax=ax, bins=30, edgecolor="black", alpha=0.7, color="steelblue")
                ax.set_xlabel(column)
                ax.set_ylabel("Frequency")
                ax.set_title(title or f"Distribution of {column}")

            # ── bar ──
            elif chart_type == "bar":
                if not column:
                    return json.dumps({"error": "'column' required for bar chart"})
                if group_by:
                    data = self.df.groupby(group_by)[column].mean().sort_values(ascending=False).head(15)
                    data.plot(kind="bar", ax=ax, color="steelblue", edgecolor="black")
                    ax.set_ylabel(f"Mean {column}")
                else:
                    data = self.df[column].value_counts().head(15)
                    data.plot(kind="bar", ax=ax, color="steelblue", edgecolor="black")
                    ax.set_ylabel("Count")
                ax.set_title(title or f"Bar Chart: {column}")
                plt.xticks(rotation=45, ha="right")

            # ── scatter ──
            elif chart_type == "scatter":
                if not columns or len(columns) < 2:
                    return json.dumps({"error": "Two 'columns' required for scatter plot"})
                self.df.plot.scatter(x=columns[0], y=columns[1], ax=ax, alpha=0.5, color="steelblue")
                ax.set_title(title or f"{columns[0]} vs {columns[1]}")

            # ── box ──
            elif chart_type == "box":
                if not column:
                    return json.dumps({"error": "'column' required for box plot"})
                if group_by and group_by in self.df.columns:
                    self.df.boxplot(column=column, by=group_by, ax=ax)
                    plt.suptitle("")
                else:
                    self.df[[column]].boxplot(ax=ax)
                ax.set_title(title or f"Box Plot: {column}")

            # ── line ──
            elif chart_type == "line":
                if not column:
                    return json.dumps({"error": "'column' required for line chart"})
                self.df[column].plot(ax=ax, color="steelblue")
                ax.set_title(title or f"Line Chart: {column}")
                ax.set_ylabel(column)

            # ── heatmap ──
            elif chart_type == "heatmap":
                numeric = self.df.select_dtypes(include=[np.number])
                if numeric.empty:
                    plt.close(fig)
                    return json.dumps({"error": "No numeric columns for heatmap"})
                corr = numeric.corr()
                sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax, center=0, square=True)
                ax.set_title(title or "Correlation Heatmap")

            # ── pie ──
            elif chart_type == "pie":
                if not column:
                    return json.dumps({"error": "'column' required for pie chart"})
                data = self.df[column].value_counts().head(10)
                data.plot(kind="pie", ax=ax, autopct="%1.1f%%")
                ax.set_ylabel("")
                ax.set_title(title or f"Distribution: {column}")

            else:
                plt.close(fig)
                return json.dumps({
                    "error": f"Unknown chart type: {chart_type}",
                    "supported": list(self.SUPPORTED_CHARTS),
                })

            plt.tight_layout()
            return self.save(fig, chart_type, column or "all")

        except Exception as e:
            plt.close("all")
            logger.error(f"VisualizationTool error: {e}")
            return json.dumps({"error": str(e)})


# ══════════════════════════════════════════════
#  LLM Dataset Context Builder
# ══════════════════════════════════════════════

def build_dataset_context(df: pd.DataFrame) -> str:
    """
    Build a compact text summary of the dataset for the LLM.
    Never includes the full data — only metadata, stats, and a few sample rows.
    """
    lines = [
        f"Dataset: {len(df)} rows × {len(df.columns)} columns",
        "",
        "Columns:",
    ]

    for col in list(df.columns)[:MAX_CONTEXT_COLUMNS]:
        dtype = str(df[col].dtype)
        nulls = df[col].isna().sum()
        uniques = df[col].nunique()
        line = f"  - {col} ({dtype}) | {nulls} nulls | {uniques} unique"

        try:
            if pd.api.types.is_numeric_dtype(df[col]):
                line += f" | range: [{df[col].min():.2f}, {df[col].max():.2f}] | mean: {df[col].mean():.2f}"
            elif df[col].dtype == "object":
                top = df[col].value_counts().head(3).index.tolist()
                line += f" | top values: {top}"
        except Exception:
            pass

        lines.append(line)

    if len(df.columns) > MAX_CONTEXT_COLUMNS:
        lines.append(f"  ... and {len(df.columns) - MAX_CONTEXT_COLUMNS} more columns")

    lines.append("")
    lines.append(f"Sample rows:\n{df.head(MAX_SAMPLE_ROWS).to_string()}")

    return "\n".join(lines)


# ══════════════════════════════════════════════
#  LangGraph Agent Builder
# ══════════════════════════════════════════════

TOOL_DESCRIPTIONS = """
Available tools:

1. dataset_metadata — Get column info, dtypes, statistics, missing values
   params: {}

2. data_analysis — Perform calculations directly on the data
   params: {
     "operation": "describe | value_counts | groupby | filter | correlation | percentile | ranking | average | count",
     "column": "column_name",
     "columns": ["col1", "col2"],
     "group_by": "column_name",
     "agg_func": "mean | sum | count | min | max | median",
     "filter_condition": "pandas query expression, e.g. Age > 40",
     "top_n": 10
   }

3. visualization — Generate a chart
   params: {
     "chart_type": "histogram | bar | scatter | box | line | heatmap | pie",
     "column": "column_name",
     "columns": ["col1", "col2"],
     "group_by": "column_name",
     "title": "optional title"
   }
""".strip()


def create_agent(
    llm: ChatOllama,
    tools_map: Dict[str, Any],
    dataset_context: str,
):
    """Build and compile the LangGraph reasoning agent."""

    system_prompt = f"""You are a data analytics AI agent. You answer questions about datasets by selecting the right analytical tool.

DATASET CONTEXT:
{dataset_context}

{TOOL_DESCRIPTIONS}

RULES:
1. Respond with ONLY a valid JSON object — no explanation outside the JSON.
2. Format: {{"tool": "tool_name", "params": {{...}}, "reasoning": "one-line explanation"}}
3. Use EXACT column names from the dataset context above.
4. For distribution / "show distribution" questions → use visualization with histogram.
5. For "which X has highest Y" → use data_analysis with groupby.
6. For "how many" / threshold questions → use data_analysis with filter or count.
7. For correlation questions → use data_analysis with correlation.
8. For "compare X across Y" → use data_analysis with groupby.
9. Only include params that are needed for the chosen tool."""

    # ── Graph Nodes ──

    def reasoning_node(state: AgentState) -> AgentState:
        """LLM reasons about the question and picks a tool + params."""
        logger.info(f"[reasoning] question: {state['question']}")
        try:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=f"User question: {state['question']}"),
            ]
            response = llm.invoke(messages)
            raw = response.content.strip()
            logger.info(f"[reasoning] raw LLM output: {raw[:300]}")

            # Extract JSON from response (skip any markdown fences)
            json_match = re.search(r"\{.*\}", raw, re.DOTALL)
            if not json_match:
                state["error"] = "LLM did not return valid JSON"
                state["selected_tool"] = "none"
                state["tool_input"] = {}
                state["reasoning"] = raw
                return state

            parsed = json.loads(json_match.group())
            state["selected_tool"] = parsed.get("tool", "data_analysis")
            state["tool_input"] = parsed.get("params", {})
            state["reasoning"] = parsed.get("reasoning", "")

        except json.JSONDecodeError as e:
            logger.error(f"[reasoning] JSON parse error: {e}")
            state["error"] = f"Failed to parse LLM response"
            state["selected_tool"] = "none"
            state["tool_input"] = {}
        except Exception as e:
            logger.error(f"[reasoning] error: {e}")
            state["error"] = str(e)
            state["selected_tool"] = "none"
            state["tool_input"] = {}

        return state

    def tool_execution_node(state: AgentState) -> AgentState:
        """Run the selected analytical tool."""
        tool_name = state.get("selected_tool", "none")
        tool_input = state.get("tool_input", {})
        logger.info(f"[tool_execution] tool={tool_name} params={tool_input}")

        if tool_name not in tools_map:
            state["tool_output"] = json.dumps({"error": f"Unknown tool: {tool_name}"})
            return state

        try:
            result = tools_map[tool_name].run(**tool_input)
            state["tool_output"] = result

            # Capture chart path if it's a visualization
            if tool_name == "visualization":
                try:
                    state["chart_path"] = json.loads(result).get("file_path")
                except Exception:
                    pass
        except Exception as e:
            logger.error(f"[tool_execution] error: {e}")
            state["tool_output"] = json.dumps({"error": str(e)})

        return state

    def response_generation_node(state: AgentState) -> AgentState:
        """Turn the tool output into a clear natural-language answer."""
        if state.get("error") and not state.get("tool_output"):
            state["answer"] = f"I encountered an error: {state['error']}"
            return state

        try:
            prompt = (
                "You are a concise data analyst. Based on the analysis results below, "
                "provide a clear, direct natural language answer to the user's question.\n\n"
                f"Question: {state['question']}\n"
                f"Tool used: {state.get('selected_tool')}\n"
                f"Analysis result:\n{state.get('tool_output', 'No results')}\n\n"
                "Rules:\n"
                "- Be concise and direct\n"
                "- Include key numbers\n"
                "- If a chart was generated, tell the user\n"
                "- Do NOT output raw JSON\n"
            )
            messages = [
                SystemMessage(content="You summarise data analysis results in plain English. Be concise."),
                HumanMessage(content=prompt),
            ]
            response = llm.invoke(messages)
            state["answer"] = response.content.strip()

        except Exception as e:
            logger.error(f"[response_generation] error: {e}")
            # Fallback: surface raw tool output
            try:
                data = json.loads(state.get("tool_output", "{}"))
                if "error" in data:
                    state["answer"] = f"Analysis error: {data['error']}"
                else:
                    state["answer"] = f"Analysis complete. Result: {json.dumps(data, indent=2)}"
            except Exception:
                state["answer"] = state.get("tool_output", "Unable to generate answer.")

        return state

    # ── Build Graph ──

    graph = StateGraph(AgentState)

    graph.add_node("reasoning", reasoning_node)
    graph.add_node("tool_execution", tool_execution_node)
    graph.add_node("response_generation", response_generation_node)

    graph.set_entry_point("reasoning")
    graph.add_edge("reasoning", "tool_execution")
    graph.add_edge("tool_execution", "response_generation")
    graph.add_edge("response_generation", END)

    return graph.compile()


# ══════════════════════════════════════════════
#  AIEngine Class
# ══════════════════════════════════════════════

class AIEngine:
    """
    AI-powered data analytics engine.

    Uses a LangGraph agent with a local Ollama LLM to answer
    natural language questions about any structured dataset.

    Args:
        df:              Cleaned pandas DataFrame (from data_cleaner.py)
        eda_insights:    Optional dict of pre-computed EDA insights (from eda_engine.py)
        model:           Ollama model name (default: qwen3:8b)
        ollama_base_url: Ollama server URL
        charts_dir:      Directory to save generated charts

    Usage:
        engine = AIEngine(df)
        result = engine.ask("What is the average age?")
    """

    def __init__(
        self,
        df: pd.DataFrame,
        eda_insights: Optional[Dict] = None,
        model: str = "qwen3:8b",
        ollama_base_url: str = "http://localhost:11434",
        charts_dir: str = "charts",
    ):
        logger.info(f"Initializing AIEngine | model={model} | shape={df.shape}")

        self.df = df
        self.model = model
        self.charts_dir = Path(charts_dir)
        self.charts_dir.mkdir(exist_ok=True)

        # ── LLM ──
        self.llm = ChatOllama(
            model=model,
            base_url=ollama_base_url,
            temperature=0.1,      # Low temp for analytical accuracy
        )

        # ── Tools ──
        self.tools: Dict[str, Any] = {
            "dataset_metadata": DatasetMetadataTool(df),
            "data_analysis": DataAnalysisTool(df),
            "visualization": VisualizationTool(df, self.charts_dir),
        }

        # ── Dataset context (compact, never full data) ──
        self.dataset_context = build_dataset_context(df)

        # ── LangGraph agent ──
        self.agent = create_agent(self.llm, self.tools, self.dataset_context)

        logger.info("AIEngine ready")

    # ── Public API ──

    def ask(self, question: str) -> Dict[str, Any]:
        """
        Ask a natural language question about the dataset.

        Returns:
            {
                "answer":    str   — Natural language response,
                "chart":     str   — Path to chart image (or None),
                "data":      dict  — Raw analysis data (or None),
                "tool_used": str   — Which tool was selected,
            }
        """
        logger.info(f"ask() → {question}")

        if not question or not question.strip():
            return {
                "answer": "Please provide a question about the dataset.",
                "chart": None,
                "data": None,
                "tool_used": None,
            }

        initial_state: AgentState = {
            "question": question.strip(),
            "dataset_info": self.dataset_context,
            "reasoning": "",
            "selected_tool": "",
            "tool_input": {},
            "tool_output": "",
            "answer": "",
            "chart_path": None,
            "error": None,
        }

        try:
            final = self.agent.invoke(initial_state)

            # Try to parse structured data from tool output
            data = None
            try:
                data = json.loads(final.get("tool_output", "{}"))
            except (json.JSONDecodeError, TypeError):
                pass

            result = {
                "answer": final.get("answer", "Unable to process question."),
                "chart": final.get("chart_path"),
                "data": data,
                "tool_used": final.get("selected_tool"),
            }
            logger.info(f"ask() done | tool={result['tool_used']} | chart={result['chart']}")
            return result

        except Exception as e:
            logger.error(f"Agent pipeline error: {e}")
            return {
                "answer": f"Error processing question: {e}",
                "chart": None,
                "data": None,
                "tool_used": None,
            }

    # ── Utility Methods ──

    def get_metadata(self) -> Dict[str, Any]:
        """Return dataset metadata without using the LLM."""
        return json.loads(self.tools["dataset_metadata"].run())

    def get_available_columns(self) -> List[str]:
        """Return list of column names in the dataset."""
        return self.df.columns.tolist()


# ─────────────────────────────────────────────
# CLI Entry Point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python ai_engine.py <csv_file> [question]")
        print('Example: python ai_engine.py data.csv "What is the average age?"')
        sys.exit(1)

    csv_path = sys.argv[1]
    question = sys.argv[2] if len(sys.argv) > 2 else "Describe this dataset"

    df = pd.read_csv(csv_path)
    engine = AIEngine(df)
    result = engine.ask(question)

    print("\n" + "=" * 60)
    print(f"Question: {question}")
    print(f"Answer:   {result['answer']}")
    if result["chart"]:
        print(f"Chart:    {result['chart']}")
    print("=" * 60)
