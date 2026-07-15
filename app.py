import os
import certifi
import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain import hub
from langchain.agents import create_react_agent, AgentExecutor

# Load environment variables
os.environ["SSL_CERT_FILE"] = certifi.where()
load_dotenv()

# Streamlit page config
st.set_page_config(page_title="LangChain AI Agent", page_icon="🤖")
st.title("🤖 LangChain AI Agent (Groq + Tavily)")
st.write("Ask a question and the agent can use web search when needed.")

# Search tool
search_tool = TavilySearchResults(max_results=2)

# LLM
llm = ChatGroq(
    model_name="llama-3.1-8b-instant",
    temperature=0
)

# Prompt
prompt = hub.pull("hwchase17/react")

# Tools
tools = [search_tool]

# Create agent
agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=prompt
)

# Executor
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True
)

# User input
query = st.text_input(
    "Enter your question:",
    placeholder="Find the capital of India and its current weather"
)

if st.button("Ask Agent"):
    if query.strip():
        with st.spinner("Thinking..."):
            try:
                response = agent_executor.invoke({"input": query})
                st.subheader("Answer")
                st.write(response["output"])
            except Exception as e:
                st.error(f"Error: {str(e)}")
    else:
        st.warning("Please enter a question.")