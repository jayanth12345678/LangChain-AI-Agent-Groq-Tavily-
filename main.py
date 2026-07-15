import os
import certifi
import requests
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain import hub


from langchain.agents import create_react_agent, AgentExecutor

#create_react_agent = reasioning and action agent

#LOAD ENV VARIABLES

os.environ["SSL_CERT_FILE"] = certifi.where()
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

#INETRNET SEARCH RESULTS
search_tool = TavilySearchResults(max_results=2)

result = search_tool.invoke("give me the latest news on AI")
result

#LLM

from langchain_groq import ChatGroq

llm = ChatGroq(

model_name="llama-3.1-8b-instant",

temperature=0

)


response = llm.invoke("what year is it?")
response


#PROMPT (according to react agent we have rpedefined rpompt here in langchain hub)

prompt = hub.pull("hwchase17/react")


#TOOLS


tools = [search_tool]


#CREATE AGENT

agent = create_react_agent(
    llm = llm,
    tools = tools,
    prompt = prompt
)



#WE HAVE TO RUN THE AGENT WITH THE HELP OF AGENT EXECUTOR
#EXECUTOR

agent_executor = AgentExecutor(
    agent = agent,
    tools = tools,
    verbose = True
)

#verbose = True we can see logs whatever thought, action and observation your agent is doing


#RUN

response = agent_executor.invoke({
    "input": (
        "Find the capital of india"
        "and find its current weather."
    )
})

print(response["output"])


