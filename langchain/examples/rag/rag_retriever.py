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
import asyncio


# retrieveing from the vector store
print('\n\n------retrieveing from the vector store')
# retrieveing relevant chunks from the vector store

# using the same model that use used to embed the text chunks
embedding_model=OllamaEmbeddings(model='bge-m3:567m')

# initializing database
db = Chroma(
    persist_directory=r'D:\python\ai\deep learning\ai_frameworks\langchain\examples\rag\db\chroma_db',
    embedding_function=embedding_model
)

# retriever
retriever = db.as_retriever(
    search_type='similarity_score_threshold',
    search_kwargs={'k':3, 'score_threshold':0.3}   #top 'k' results where each result has minimum threshold of 0.3
)

# query
query = 'what is computer vision and yolo'

results = retriever.invoke(query)
print(f'query = {query}')
print('relevant text chunks : ')
for i in results:
    print(f'chunk : \t{i.page_content}')
    print(f'source : \t{i.metadata["source"]}')
    print()
print('-----------------------------------------------')
print(results)