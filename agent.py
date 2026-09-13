from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import StructuredTool

from tools import (
    load_dataset,
    dataset_info,
    statistical_summary,
    correlation_analysis,
    create_histogram
)

load_dotenv()


# LLM
llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0
)


# TOOL FUNCTIONS
def inspect_dataset(file_path: str):
    """Inspect the CSV structure, columns, data types and missing values."""
    df = load_dataset(file_path)
    return dataset_info(df)


def analyze_statistics(file_path: str):
    """Calculate descriptive statistics for numerical columns."""
    df = load_dataset(file_path)
    return statistical_summary(df)


def analyze_correlations(file_path: str):
    """Find the strongest correlations between numerical columns."""
    df = load_dataset(file_path)
    return correlation_analysis(df)


def plot_distribution(file_path: str, column: str):
    """Create a histogram for a numerical column."""
    df = load_dataset(file_path)
    return create_histogram(df, column)

def data_quality_report(file_path: str):
    """Generate a concise data-quality report covering missing values, numerical statistics, and strong correlations."""
    df = load_dataset(file_path)

    inspection = dataset_info(df)
    statistics = statistical_summary(df)
    correlations = correlation_analysis(df)

    return f"""
DATA QUALITY REPORT

{inspection}

{statistics}

{correlations}
"""


# LANGCHAIN TOOLS
inspect_tool = StructuredTool.from_function(
    inspect_dataset,
    name="inspect_dataset",
    description=(
        "Inspect a CSV dataset's structure, columns, "
        "data types and missing values."
    )
)

statistics_tool = StructuredTool.from_function(
    analyze_statistics,
    name="analyze_statistics",
    description=(
        "Calculate descriptive statistics such as "
        "mean, standard deviation, minimum and maximum."
    )
)

correlation_tool = StructuredTool.from_function(
    analyze_correlations,
    name="analyze_correlations",
    description=(
        "Find the strongest correlations between "
        "numerical variables."
    )
)

plot_tool = StructuredTool.from_function(
    plot_distribution,
    name="plot_distribution",
    description=(
        "Create a histogram showing the distribution "
        "of a numerical column."
    )
)


tools = [
    inspect_tool,
    statistics_tool,
    correlation_tool,
    plot_tool
]


tool_map = {
    tool.name: tool
    for tool in tools
}


# BIND TOOLS
model_with_tools = llm.bind_tools(tools)


# CONTROLLED AGENT
def run_agent(file_path, question):

    messages = [
        HumanMessage(
            content=f"""
You are DataWise, an autonomous data science agent.

The CSV dataset is located at:

{file_path}

User question:

{question}

Choose the most appropriate analysis tool.

IMPORTANT:
- Use the minimum number of tools necessary.
- Normally use only one tool.
- Call a tool at most once.
- After receiving the tool result, provide the final answer.
- Never invent numerical values.
- Trust the results returned by the Python tools.
"""
        )
    ]

    # Allow a maximum of TWO rounds
    for _ in range(2):

        response = model_with_tools.invoke(messages)

        messages.append(response)

        # MODEL HAS PROVIDED FINAL ANSWER

        if not response.tool_calls:
            return response.content

        # EXECUTE TOOL CALLS

        for tool_call in response.tool_calls:

            tool_name = tool_call["name"]
            tool_args = tool_call["args"]

            if tool_name not in tool_map:

                return (
                    f"The agent selected an unavailable tool: "
                    f"{tool_name}"
                )

            tool_result = tool_map[
                tool_name
            ].invoke(tool_args)

            messages.append(
                ToolMessage(
                    content=str(tool_result),
                    tool_call_id=tool_call["id"]
                )
            )

        # ASK MODEL FOR FINAL ANSWER

        final_response = model_with_tools.invoke(
            messages
        )

        messages.append(final_response)

        if not final_response.tool_calls:

            return final_response.content


    # SAFETY FALLBACK

    return (
        "The analysis required more steps than "
        "allowed. Please ask a more specific question."
    )