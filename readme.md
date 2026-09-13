# 🤖 DataWise AI

### Autonomous Data Science Agent

DataWise AI is an autonomous data science agent that allows users to upload a CSV dataset and ask questions about their data using natural language.

Instead of requiring the user to manually select an analysis method, the AI agent understands the question and selects the most appropriate data-analysis tool. Python then performs the actual computation, and the AI interprets the results and provides a clear response.

---

## 🚀 Features

- 📁 Upload CSV datasets
- 💬 Ask questions using natural language
- 🤖 AI-powered tool selection
- 🔍 Dataset structure and data-quality inspection
- 📊 Descriptive statistical analysis
- 🔗 Correlation analysis
- 📈 Histogram generation
- 🧠 AI-generated interpretation of analysis results
- 🌐 Streamlit-based web interface

---

## 🧠 How It Works

The system follows an agent-based workflow:

```text
User
  ↓
Streamlit Web Interface
  ↓
LangChain
  ↓
Groq LLM
  ↓
Tool Selection
  ↓
Python Analysis Tool
  ↓
Pandas / NumPy / Matplotlib
  ↓
Analysis Result
  ↓
LLM Interpretation
  ↓
Final Answer
```
