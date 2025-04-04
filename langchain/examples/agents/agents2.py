from langchain_ollama import ChatOllama
from langchain_core.tools import Tool
from langchain.agents import create_react_agent, AgentExecutor, create_structured_chat_agent
from langchain import hub
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

import os


def current_time(*args, **kwargs):
    """returns current time"""
    import time
    return 'current time is = ' + time.ctime(time.time())

def wikipedia_summary(query):
    """returns summary of the related topic from wikipedia if available """
    import wikipedia
    
    try:
        return f'wikipedia summary = {wikipedia.summary(query, sentences=2)}'
    except:
        return f'wikipedia summary is not available for the given query'

tools = [
    Tool(
        name='time',
        func=current_time,
        description='function to get current time'
    ),
    Tool(
        name='wikipedia',
        func=wikipedia_summary,
        description='function to get summary from wikipedia on the given topic'
    )
]

prompt = hub.pull('hwchase17/structured-chat-agent')

llm = ChatOllama(model='qwen2.5:0.5b', temperature=0)


# saves the conversation/chat history to the memory so that the agent can always access chat history across interactions. we use this instead of creating a chat_history list and manually saving the messages and then sending the message to the model everytime
memory = ConversationBufferMemory(
    memory_key='chat_history',
    return_messages=True
)

# creates a agent that is interactive(it uses a llm, tools and prompt template)
agent = create_structured_chat_agent(
    llm=llm, 
    tools=tools, 
    prompt=prompt
)

agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent,
    tools=tools,
    verbose=True,
    memory=memory,
    handle_parsing_errors=True
)

# saving initial message to the conversation memory. this initial memory provides context to the LLM about what to do
initial_message = 'you are a helpful AI assistant. you job is to provide answers to the given query using the tools that you have.\n if you are unable to provide correct answer, just say "I DONT KNOW THE ANSWER". give concise answer'
memory.chat_memory.add_message(SystemMessage(content = initial_message))


print('you can now start chatting with the AI. type "exit" to exit')
while True:
    query = input('User : ')
    if query.lower() == 'exit':
        break
    
    # adding the user's query to the chat memory
    memory.chat_memory.add_message(HumanMessage(content=query))
    
    response=agent_executor.invoke({'input':query})
    print(f'AI : {response["output"]}')
    
    # adding the AI response to the chat memory
    memory.chat_memory.add_message(AIMessage(content=response['output']))