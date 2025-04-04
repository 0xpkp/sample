from langchain.text_splitter import (
    TextSplitter,
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
    SentenceTransformersTokenTextSplitter,
    TokenTextSplitter
)
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_ollama import OllamaEmbeddings

import os
import asyncio
import time

"""
# loading sample data
files = [
    r'D:\python\ai\deep learning\ai_frameworks\langchain\examples\rag\data.txt'
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



# DEEPDIVE INTO TEXT SPLITTING IN RAG
print('---------------------deepdive into text splitting in langchain-------------------------')

# character text splitter
print('character based text splitter')
print(f'sample original text = {all_content[0].page_content[:200]}')
splitter = CharacterTextSplitter(separator='', chunk_size=50, chunk_overlap=10)
chunks = splitter.split_documents(all_content)  #you can also use split_text() when you pass string but here we pass Documents()
asd='\n\t- '.join([i.page_content for i in chunks[:3]])
print(f'sample chunks = \n\t- {asd}\n\n')

# sentence based text splitter - 
print('sentence based text splitter')
print(f'sample original text = {all_content[0].page_content[:200]}')
splitter = SentenceTransformersTokenTextSplitter(chunk_overlap=2, tokens_per_chunk=50)
chunks = splitter.split_documents(all_content)
asd='\n\t- '.join([i.page_content for i in chunks[:3]])
print(f'sample chunks = \n\t- {asd}\n\n')

# token text splitter - splits the texts based on number of tokens 
print('token text splitter')
print(f'sample original text = {all_content[0].page_content[:200]}')
splitter = TokenTextSplitter(chunk_size=50, chunk_overlap=10)
chunks = splitter.split_documents(all_content)
asd='\n\t- '.join([i.page_content for i in chunks[:3]])
print(f'sample chunks = \n\t- {asd}\n\n')

# recursive text splitter - recursively split the texts until the chunks are small enough. the default list is ['\n\n', '\n', ' ', ''] which means first splits by '\n\n', then by '\n' and so on. this has the effect of trying to keep all paragraphs, sentences and words together as long as possible. (most used splitter)
# https://python.langchain.com/v0.1/docs/modules/data_connection/document_transformers/recursive_text_splitter/

print('recursive text splitter')
print(f'sample original text = {all_content[0].page_content[:200]}')
splitter = RecursiveCharacterTextSplitter(chunk_size=50, chunk_overlap=10)
chunks = splitter.split_documents(all_content)
asd='\n\t- '.join([i.page_content for i in chunks[:3]])
print(f'sample chunks = \n\t- {asd}\n\n')

"""


# DEEPDIVE INTO INFORMATION RETRIEVAL IN RAG
print('\n\n\n\n---------------------deepdive into information retrieval in langchain-------------------------')

# embedding model(use the same embedding model that is used when embedding the chunks)
embedding_model = OllamaEmbeddings(model='bge-m3:567m')

# database
db = Chroma(
    persist_directory='./db/chroma_db',
    embedding_function=embedding_model
)

def retriever_function(database:Chroma, querys:list, search_type, search_kwargs):
    """function to retrieve relevant chunks from the database"""
    print(f'\n\n----------------------{search_type.upper()} SEARCH--------------------------------')
    print(f'search type = {search_type}\t search kwargs = {search_kwargs}')
    print('-----------------------------------------------------------------------------------')
    
    
    retriever = database.as_retriever(
        search_type=search_type,
        search_kwargs=search_kwargs
    )
    
    for query in querys:
        results = retriever.invoke(query)
        
        print(f'\nquery : {query}')
        asd = "\n\t-> ".join([i.page_content for i in results])
        print(f'relevant chunks : \n\t-> {asd}')
        
    print('\n\n')

query1 = 'what is computer vision'
query2 = 'who is praveen'
query3 = 'what are different approaches for computer vision using deep learning'

querys = [query1, query2, query3]



# similarity search 
"""  
uses cosine similarity to get the most relevant chunk
use this when we want top 'k' results but dont want any threshold
"""

# similarity search with k=3
retriever_function(database=db, querys=querys, search_type='similarity', search_kwargs={'k':3})


# MMR(max marginal relevance)
"""  
searches the most relevant chunk while also optimizing for diversity.

'fetch_k' - specifies number of chunks to initially fetch based on cosine similarity
'lambda_mult' - controls the diversity of the results. 1 for minimum diversity and 0 for maximum diversity
"""
retriever_function(database=db, querys=querys, search_type='mmr', search_kwargs={'k':3, 'fetch_k':20, 'lambda_mult':0.7})


# similarity score threshold 
"""
similarity score with minimum threshold where only those chunks are returned where the cosine similarity score is above minimum threshold  
"""
retriever_function(database=db, querys=querys, search_type='similarity_score_threshold', search_kwargs={'k':3, 'score_threshold':0.4}) 