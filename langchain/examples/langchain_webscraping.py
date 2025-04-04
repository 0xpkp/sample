# scraping web data using firecrawl and langchain web loader

from langchain_community.document_loaders import WebBaseLoader   #simple webpage loader
from langchain_community.document_loaders import FireCrawlLoader   #much better crawler. pip install firecrawl-py

from dotenv import load_dotenv
import os


# urls to scrape
urls = [
    'https://0xpkp.github.io/portfolio/',
    'https://www.apple.com/'
]

# web base loader
loader = WebBaseLoader(urls)
documents = loader.load()

print('------------documents collected using web base loader-----------------')
print(f'\n\n number of documents = {len(documents)}')
print(f'\n documents overview : ')
for i in documents:
    print(f'\n\tsource = {i.metadata["source"]}')
    print(f'\n\tcontent = {i.page_content}')

print(f'full data : {documents}')


# using firecrawl loader

# loading the api key
load_dotenv()
FIRECRAWL_API = os.getenv('FIRECRAWL')
documents=[]
for url in urls:
    loader = FireCrawlLoader(api_key=FIRECRAWL_API, url=url, mode='scrape')  # 3 modes : 1)scrape - to scrape the content of the url, 2)crawl - crawls throught the whole url, 3)map - maps the entire url
    docs = loader.load()
    documents.append(docs[0])
    
print('------------documents collected using firecrawl loader-----------------')
print(f'\n\n number of documents = {len(documents)}')
print(f'\n documents overview : ')
for i in documents:
    print(f'\n\tsource = {i.metadata["url"]}')
    print(f'\n\tcontent = {i.page_content}')

print(f'full data : {documents}')

# ########## you can see that firecrawl's response is way better than web loader

"""  

********************************************************************************************************
********************************************************************************************************
NOTE : 
 after loading the web pages, follow the same process(split the documents into chunks, embed them, store into the vectorstore database and then you can use it for retrieval purposes.)
********************************************************************************************************
********************************************************************************************************

"""