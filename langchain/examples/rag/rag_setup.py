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



# loading the files
files = [
    r'D:\python\ai\deep learning\ai_frameworks\langchain\sample_file.txt',
    r"D:\books\ai books\deep learning books\Data-Scientist-Books-main\Deep Learning for Computer Vision - Image Classification, Object Detection and Face Recognition in Python by Jason Brownlee (z-lib.org).pdf"
]

async def load_all_content(files):
    all_content = []
    for file in files:
        if '.pdf' in file:
            loader = PyPDFLoader(file)
            async for page in loader.alazy_load():  # loading each page and saving it as a string
                all_content.append(page)
        elif '.txt' in file:
            loader = TextLoader(file)
            document = loader.load()
            all_content.append(document[0])
    return all_content
print('loading files...')
all_content = asyncio.run(load_all_content(files))

# splitting the text into chunks of 5000 characters
# splitting the text using character text splitter. 
# NOTE : when you use text loaders, the loader will return the content in document format. we can either keep it as it is or else extract only the text and then split. both are ok but i prefer the 1st method(no extra steps needed)

##################  langchain also have functions to split the text based on words or by based on number of tokens(both are better than character level split because at the end we will tokenize and pass it to a language model)
text_splitter = CharacterTextSplitter(
    separator='',  #we will split based on number of characters
    chunk_size=1500,  #each chunk will have 5000 characters,
    chunk_overlap=0,
    length_function=len
)
print('splitting texts...')
chunks = text_splitter.split_documents(all_content)
"""  
NOTE : you can also add custom metadata to your chunks. like you can add source name and chunk number/page number so that we can show the user the source more precisely
"""
print('----------------------------------------------')
print(f'number of chunks = {len(chunks)}')
print(f'\nsample chunk = {chunks[0]}')
print('----------------------------------------------\n\n')


# saving the chunks and the embeddings to Chroma vector store
# create a 'db' directory
if os.path.exists(r'D:\python\ai\deep learning\ai_frameworks\langchain\examples\rag\db\chroma_db'):
    print('embedding vector store already exists')
else:
    print('creating new vectorstore...')
    os.makedirs(r'D:\python\ai\deep learning\ai_frameworks\langchain\examples\rag\db', exist_ok=True)

    # embedding model
    embedding_model = OllamaEmbeddings(model='bge-m3:567m')

    db = Chroma.from_documents(
        chunks, embedding_model, persist_directory=r'D:\python\ai\deep learning\ai_frameworks\langchain\examples\rag\db\chroma_db'
    )
    print('new vectorstore created!')
    
    
    
print('#########################################################################')
print('###########RAG SETUP COMPLETE############################################')
print('#########################################################################')
