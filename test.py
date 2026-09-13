from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.tools import tool
from langchain.agents import create_agent

from tools import (
    load_dataset,
    dataset_info,
    statistical_summary,
    correlation_analysis,
    create_histogram
)


# --------------------------------------------------
# LOAD ENVIRONMENT VARIABLES
# --------------------------------------------------

load_dotenv()


# --------------------------------------------------
# AI MODEL
# --------------------------------------------------

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0
)


# --------------------------------------------------
# AGENT TOOLS
# --------------------------------------------------

@tool
def inspect_dataset(file_path: str):
    """Load a CSV dataset and return its structure, columns, data types and missing values."""

    df = load_dataset(file_path)

    return dataset_info(df)


@tool
def analyze_statistics(file_path: str):
    """Calculate descriptive statistics for numerical columns."""

    df = load_dataset(file_path)

    return statistical_summary(df)


@tool
def analyze_correlations(file_path: str):
    """Find the strongest correlations between numerical columns."""

    df = load_dataset(file_path)

    return correlation_analysis(df)


@tool
def plot_distribution(file_path: str, column: str):
    """Create a histogram showing the distribution of a numerical column."""

    df = load_dataset(file_path)

    return create_histogram(df, column)


# --------------------------------------------------
# TOOL LIST
# --------------------------------------------------

tools = [
    inspect_dataset,
    analyze_statistics,
    analyze_correlations,
    plot_distribution
]


# --------------------------------------------------
# CREATE AI AGENT
# --------------------------------------------------

agent = create_agent(
    model=llm,
    tools=tools,

    system_prompt="""
You are an autonomous data science analysis agent.

Your job is to answer questions about CSV datasets using
the available Python data-analysis tools.

Available tools:

1. inspect_dataset
   Understand dataset structure, columns, data types
   and missing values.

2. analyze_statistics
   Calculate descriptive statistics for numerical columns.

3. analyze_correlations
   Find the strongest relationships between numerical variables.

4. plot_distribution
   Create a histogram for a numerical column.

Rules:

- Use tools whenever actual information from the dataset is required.
- Start with inspect_dataset when working with an unfamiliar dataset.
- Use only the tools necessary to answer the user's question.
- Do not repeatedly call the same tool.
- Trust the values returned by the tools.
- Never invent numerical values.
- Do not guess dataset information.
- After obtaining sufficient information, stop using tools.
- Provide a clear and concise final answer.
"""
)