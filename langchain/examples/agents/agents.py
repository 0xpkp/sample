from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.tools import Tool
from langchain_ollama import ChatOllama

import os

def get_current_time(*args, **kwargs):
    """returns current time"""
    import time
    return 'current time = ' + time.ctime(time.time())

tools = [
    Tool(
        name='time',
        func=get_current_time,
        description='useful when you want to know the current time'   #description should be as clear and detailed as possible
    )
]

# pulling a prompt template from langchain hub
# ReAct - reason and action
# https://smith.langchain.com/hub/hwchase17/react
prompt = hub.pull('hwchase17/react')

llm = ChatOllama(model='qwen2.5:0.5b', temperature=0)

# defining a agent
agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=prompt,
    stop_sequence=True
)

agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent, 
    tools=tools,
    verbose=True
)

query = 'where am i?'
response = agent_executor.invoke({'input':query})

print(f'response : {response}')