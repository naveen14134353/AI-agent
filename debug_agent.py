from agent import agent

file_path = "datasets/OwidData.csv"

result = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": f"""
For this dataset:

{file_path}

Find the strongest correlations between numerical variables.

Use analyze_correlations exactly once.
After receiving its result, give the final answer.
"""
            }
        ]
    },
    config={"recursion_limit": 4}
)

print("\n\n===== ALL MESSAGES =====\n")

for i, message in enumerate(result["messages"]):

    print(f"\n--- MESSAGE {i} ---")
    print("TYPE:", type(message).__name__)
    print("CONTENT:", message.content)

    if hasattr(message, "tool_calls"):
        print("TOOL CALLS:", message.tool_calls)