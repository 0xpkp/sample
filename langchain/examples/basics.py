"""  
# langchain
used for building ai applications, rag systems, ai agents and other automation tasks using AI

reference : 
- https://www.youtube.com/watch?v=yF9kGESAi3M
- https://github.com/bhancockio/langchain-crash-course/tree/main

setup : 
- pip install langchain-chroma
- pip install chromadb
- pip install langchain 
- pip install langchain-community
- pip install langchainhub
- pip install langchain-anthropic
- pip install langchain-google-genai
- pip install langchain-google-firestore
- pip install langchain-openai
- pip install langchain-ollama
- pip install langchain-deepseek
- pip install langhcain-groq
- pip install firestore

<!-- firecrawl for crawling websites -->
<!-- tavily to convert google search results into LLM frendly format -->
"""

from langchain_deepseek import ChatDeepSeek
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain.schema.output_parser import StrOutputParser  #to parse the output string
from langchain.schema.runnable import RunnableLambda, RunnableSequence, RunnableParallel, RunnableBranch

from langchain.text_splitter import CharacterTextSplitter, TextSplitter, RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader  #to load text files
from langchain_community.document_loaders import PyPDFLoader, PyMuPDFLoader  #to load pdf

# pip install chromadb. and if you are getting import error for 'onnxruntime' module, install onnxruntime v1.15.1
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings  #to load the embedding model

from langchain.prompts import ChatPromptTemplate

from google.auth import compute_engine
from google.cloud import firestore
from langchain_google_firestore import FirestoreChatMessageHistory
# install langchain-0.1.6 langchain-community-0.0.19 langchain-core-0.1.23 langsmith-0.0.87  to mitigate        'pwd ModuleNotFoundError' 

from dotenv import load_dotenv
import os



# loading the API keys
load_dotenv()
deepseek_api=os.getenv('DEEPSEEK_API_KEY')

# using qwen2.5:0.5b model locally using ollama
model=ChatOllama(model='qwen2.5:0.5b',
                 temperature=0.2)
response=model.invoke('think and answer  what is the square root of 2398')
print(f'full response : \t{response}')
print(f'\nresponse.content : \t{response.content}')

print('-----------------adding system message-----------------')
# adding system message 
messages=[
    SystemMessage(content='you are AI assistant created by praveen'),  #system message
    HumanMessage(content='hello i am praveen. who created you')  #our query
]
response=model.invoke(messages)
messages.append(AIMessage(content=response.content))  #adding the AI's response to the message history
messages.append(HumanMessage('what is my name'))  #adding the next human query
response=model.invoke(messages)     #sending the whole chat history to the AI assistant
messages.append(AIMessage(response.content))    #adding the AI response to the message history
print(messages)
print('-------------------------------------------------------')


# chat loop
print('------------------------chat loop------------------')
chat_history=[
    SystemMessage(content='you are AI assistant created asd inc, which is a international AI research company. a team of Ai researched led by Praveen created you')
]

while(1):
    query=input('query : ')
    if query=='exit':
        break
    chat_history.append(HumanMessage(query))
    response=model.invoke(chat_history)
    print(f'query : {query}')
    print(f'AI assistant : {response.content}')
    chat_history.append(AIMessage(response.content))
    
print('----------------------------------------------------')
print(f'chat history : {messages}')
print('----------------------------------------------------')


# saving chat history to firebase
"""   
1) create a project in firebase.google.com
        create a firestore database
2) download gcloud CLI(reference : https://cloud.google.com/sdk/docs/install)
        authenticate google CLI with your account
            gcloud auth application-default login   --this command will login to save the application default 
                                                    credentials(ADC) in local system. a 'application_default_credentials.json' file will be created locally
        set default project with the project where you have created the firebase database
"""

PROJECT_ID = "temp-langchain"  #firebase project id
SESSION_ID = 'new_session_123'  #new session id
COLLECTION_NAME = 'chat_history'   #collection name for storing chat_history

# initializing firestore client
client = firestore.Client(project=PROJECT_ID)
# NOTE : langchain has options to save the chat message history in different formats like saving locally, in a file system, etc and we are saving it in firestore database
chat_history = FirestoreChatMessageHistory(
    session_id=SESSION_ID,
    collection=COLLECTION_NAME,
    client=client
)
# chat history has been created
print(f'current chat history : {chat_history.messages}')

# model
model=ChatOllama(
    model='qwen2.5:0.5b',
    temperature=0
)

# adding system message
system_message = SystemMessage(content = 'you are a AI assistnt created by smart AI team led by Dr.Praveen who is a renowed researcher in AI field')
# adding system message to the chat history
chat_history.add_message(system_message)

print("start chatting with the model. enter 'exit' to quit")
while True:
    query=input('query : ')
    if query.lower()=='exit':
        break
    chat_history.add_user_message(query) #adding user query to the chat history
    response=model.invoke(chat_history.messages)
    # adding ai response to the chat history
    chat_history.add_ai_message(response.content)
    print(f'query : {query}')
    print(f'response : {response.content}')

print('------------------FINAL CHAT HISTORY-------------------')
print(chat_history.messages)
# NOTE : you can also view 'chat_history' in firestore dashboard
print('--------------------------------------------------------')
print('\n\n')

# basic prompt template
print('------------------basic prompt template--------------')
template = 'create 3 jokes about {topic}'
prompt_template = ChatPromptTemplate.from_template(template)

prompt = prompt_template.invoke({'topic':'cats'})
print(prompt)
print('\n')


# multiple place holders
print('-----------------------multiple place holders-------------------')
template = """you are a helpful ai assistant.
Human : write a {adjective} story about {animal}
Assistant : 
"""
prompt_template = ChatPromptTemplate.from_template(template)

prompt = prompt_template.invoke({'adjective' : 'funny', 'animal' : 'panda'})
print(prompt)
print()


# prompt template for different types of messages
print('----------------------------------------------------------------')
messages=[
    ("system", "you are a comedian who tells jokes about {topic}. you are created by {author}"),
    ('human", "tell me {count} jokes'),
    ("assistant", "here you do")
]  #use human/user for HumanMessage(), system for SystemMessage() and assistant/ai for AIMessage()
# NOTE : if you use HumanMessage(), etc then make sure that there is no interpolation in the content of that message
prompt_template = ChatPromptTemplate.from_messages(messages)

prompt = prompt_template.invoke({'topic':'animals', 'author':'praveen', 'count' : '5'})
print(prompt)
print()


# using prompt template with a model
print('----------------------using prompt template with a model---------------')
messages = [
    ('system', 'you are a comedian who write jokes about {topic}. you are created by {author}'),
    ('human', 'write {count} jokes')
]

prompt_template = ChatPromptTemplate.from_messages(messages)
prompt = prompt_template.invoke({'topic' : 'humans', 'author' : 'praveen', 'count' : '5'})

model = ChatOllama(model='qwen2.5:0.5b', temperature=1.6)

response = model.invoke(prompt)

print(f'prompt : {prompt}')
print('---------------------------------------------------------')
print(f'model\'s reponse : {response.content}')
print('\n\n')



# chains
print('------------------chaining langchain functions--------------')
# chaining langchain functions together

model = ChatOllama(model='qwen2.5:0.5b')

messages = [
    ('system', 'you are a comedian. you are going to write jokes on topic {topic}'),
    ('human', 'write {count} jokes')
]

prompt_template = ChatPromptTemplate.from_messages(messages)

chain = prompt_template | model

result = chain.invoke({'topic':'panda', 'count':'5'})
print(result)
print()

print('------------------------------')
chain = prompt_template | model | StrOutputParser()  #to parse the model's output
result = chain.invoke({'topic' : 'panda', 'count':'3'})
print(result)
print('\n\n')


# runnable sequences and runnable lambda
print('----------------runnable lambda--------------------')
# runnable lambda in like the lambda functions in python
model = ChatOllama(model='qwen2.5:0.5b')
messages = [
    ('system', 'you are a comedian. you are going to write jokes on topic {topic}'),
    ('human', 'write {count} jokes')
]
prompt_template = ChatPromptTemplate.from_messages(messages)

format_prompt = RunnableLambda(lambda x : prompt_template.format_prompt(**x))
invoke_model = RunnableLambda(lambda x : model.invoke(x.to_messages()))
parse_output = RunnableLambda(lambda x : x.content)

sequence = format_prompt | invoke_model | parse_output
result = sequence.invoke({'topic':'pandas', 'count':'3'})
print(result)
print('------------------------------------------------')



""" runnable sequence has 3 attributes : first, middle, last each of them accepts RunnableLmabda(). if we only have 2 RunnableLambda(), then 'first' and 'last' parameter will each receive 1 RunnableLmanda(). if we ahve have 3 or more RunnableLmanda(), then the first RunnableLambda() will be set to 'first' parameter and last RunnableLambda() will be set to 'last' parameter and rest of the runnable parameter will be passed as a list to 'middle' parameter
"""
print('---------------------runnable sequence and runnable lambda-----------------')
format_prompt = RunnableLambda(lambda x : prompt_template.format_prompt(**x))
invoke_model = RunnableLambda(lambda x : model.invoke(x.to_messages()))
parse_output = RunnableLambda(lambda x : x.content)

sequence = RunnableSequence(first=format_prompt, middle=[invoke_model], last=parse_output)
result=sequence.invoke({'topic':'pandas', 'count':'3'})
print(result)
print('-----------------------------------------------------')




#chains extended
print('------------------------extended chains-----------------------')
model = ChatOllama(model='qwen2.5:0.5b')
messages = [
    ('system', 'you are a comedian. you are going to write jokes on topic {topic}'),
    ('human', 'write {count} jokes')
]
prompt_template = ChatPromptTemplate.from_messages(messages)

uppercase = RunnableLambda(lambda x : x.upper())
wordcount = RunnableLambda(lambda x : f"wordcount = {len(x.split())}\n{x}")

chain = prompt_template | model | StrOutputParser() | uppercase | wordcount

result = chain.invoke({'topic':'pandas', 'count':'3'})
print(result)
print('-----------------------------------------------------')



# parallel chains
print('--------------------parallel chains----------------------------')
model = ChatOllama(model='qwen2.5:0.5b')

messages = [
    ('system', 'you are a expert product reviewer.'),
    ('human', 'review the following product : {product_name}')
]
prompt_template = ChatPromptTemplate.from_messages(messages)

def analyze_pros(features):
    """function for preparing prompt for pros_chain branch"""
    messages = [
        ('system', 'you are an expert product reviewer'),
        ('human', 'given these features : {features}. list the pros of these features')
    ]
    prompt_template = ChatPromptTemplate.from_messages(messages)
    return prompt_template.format_prompt(features=features)

def analyze_cons(features):
    """function for preparing prompt for cons_chain branch"""
    messages = [
        ('system', 'you are an expert product reviewer'),
        ('human', 'given these features : {features}. list the cons of these features')
    ]
    prompt_template = ChatPromptTemplate.from_messages(messages)
    return prompt_template.format_prompt(features=features)

# pros branch
pros_chain = RunnableLambda(lambda x : analyze_pros(x)) | model | StrOutputParser()

# cons branch
cons_chain = RunnableLambda(lambda x : analyze_cons(x)) | model | StrOutputParser()

def combine_pros_and_cons(pros, cons):
    return f'\n--------------------------------------\npros : {pros}\n\ncons : {cons}'

chain = (
    prompt_template
    | model 
    | StrOutputParser()
    | RunnableParallel(branches = {'pros':pros_chain, 'cons':cons_chain})
    | RunnableLambda(lambda x : print(f'full output : {x}') or combine_pros_and_cons(x['branches']['pros'], x['branches']['cons']))
)

result = chain.invoke({'product_name':'macbook pro'})
print(result)
print('-----------------------------------------------------')



# branching chains
print('-------------------------branching chains----------------------')
model = ChatOllama(model='qwen2.5:0.5b')
messages = [
    ('system', 'you are a helpful assistant. you task is to classify the reivew as "positive", "negative", "neutral" or "esclate". always return the answer in 1 word'),
    ('human', 'review : {review}')
]
prompt_template = ChatPromptTemplate.from_messages(messages)

positive_response_template = ChatPromptTemplate.from_messages(
    [
        ('human', 'generate right responses for the following positive review : {review}')
    ]
)

negative_response_template = ChatPromptTemplate.from_messages(
    [
        ('human', 'generate right responses for the following negative review : {review}')
    ]
)

neutral_response_template = ChatPromptTemplate.from_messages(
    [
        ('human', 'generate right responses for the following neutral review : {review}')
    ]
)

esclate_response_template = ChatPromptTemplate.from_messages(
    [
        ('human', 'generate right responses to esclate the following review to a human assistant : {review}')
    ]
)

branches = RunnableBranch(   #any one branch will be executed at a time. we can also execute multiple branches
    (
        lambda x: '[positive]' in x,
        positive_response_template | model | StrOutputParser()  #'positive' branch
    ),
    (
        lambda x: '[negative]' in x, 
        negative_response_template | model | StrOutputParser()  # 'negative' branch
    ),
    (
        lambda x: '[neutral]' in x,
        neutral_response_template | model | StrOutputParser()  # 'neutral' branch
    ),
    esclate_response_template | model | StrOutputParser()  # 'esclate' branch. here no condition because if other branches are false this branch will be automatically executed
)

classification_chain = prompt_template | model | StrOutputParser()
chain = classification_chain | branches

sample_review = 'this is the worst product i have ever used'
result = chain.invoke({'review':sample_review})
print(result)
print('-----------------------------------------------------')



print('---------------classification chain output--------------------')
print(classification_chain.invoke({'review':'this is a very bad product. never buy this product. this is literally the worst product i have ever seen'}))