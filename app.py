import streamlit as st
import tempfile
import os

from agent import run_agent


# PAGE CONFIGURATION
st.set_page_config(
    page_title="DataWise AI",
    page_icon="🤖",
    layout="wide"
)


# HEADER
st.title("🤖 DataWise AI")
st.subheader("Autonomous Data Science Agent")

st.write(
    "Upload a CSV dataset and ask the AI agent to analyze it."
)

st.divider()


# SIDEBAR
with st.sidebar:

    st.header("⚙️ How DataWise Works")

    st.write("""
    **1. Upload Dataset**

    Upload a CSV file.

    **2. Ask a Question**

    Ask the AI a question about your data.

    **3. Agent Decision**

    The AI autonomously selects the
    appropriate analysis tool.

    **4. Python Analysis**

    Pandas and NumPy perform the
    actual calculations.

    **5. AI Insights**

    The AI interprets the results
    and provides a clear answer.
    """)

    st.divider()

    st.caption(
        "Powered by LangChain + Groq + Python"
    )


# CSV UPLOAD
st.subheader("📁 Dataset")

uploaded_file = st.file_uploader(
    "Upload your CSV dataset",
    type=["csv"]
)


# QUESTION
st.subheader("💬 Ask DataWise")

question = st.text_area(
    "What would you like to know about your dataset?",
    placeholder=(
        "Example: Find the strongest correlations "
        "between the numerical variables in this dataset."
    ),
    height=120
)


# ANALYZE BUTTON
if st.button(
    "🔍 Analyze Dataset",
    type="primary",
    use_container_width=True
):

    # Check dataset
    if uploaded_file is None:

        st.warning(
            "⚠️ Please upload a CSV dataset first."
        )

    # Check question
    elif not question.strip():

        st.warning(
            "⚠️ Please enter a question."
        )

    else:

        # SAVE UPLOADED FILE TEMPORARILY

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".csv"
        ) as temp_file:

            temp_file.write(
                uploaded_file.getvalue()
            )

            file_path = temp_file.name


        # Convert Windows backslashes to forward slashes
        # to avoid JSON/tool-calling issues
        file_path = file_path.replace("\\", "/")


        # RUN AI AGENT

        try:

            with st.spinner(
                "🤖 DataWise is analyzing your dataset..."
            ):

                final_answer = run_agent(
                    file_path,
                    question
                )


            # DISPLAY RESULT

            st.success(
                "✅ Analysis completed!"
            )

            st.divider()

            st.subheader(
                "📊 AI Agent Analysis"
            )

            st.write(
                final_answer
            )


        except Exception as e:

            st.error(
                f"❌ Something went wrong:\n\n{e}"
            )


        finally:

            # Delete temporary file
            if os.path.exists(file_path):

                os.remove(file_path)