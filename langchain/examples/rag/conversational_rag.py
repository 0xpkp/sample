###conversational-rag


from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma

from langchain.chains import create_retrieval_chain   #to easily create a conversation pipeline. this function will automalically retrieve the documents and pass it to a LLM to generate response

from langchain.chains import create_history_aware_retriever    #a retriever that can retrieve relevant chunks based not only on the current query but on the context of the whole chat

from langchain.chains.combine_documents import create_stuff_documents_chain    # this is a chain that merges the retrieved documents with the user's prompt which is then sent to LLM to generate responses

from langchain_core.prompts import MessagesPlaceholder    #place holder to insert messages in the prompt template
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain.prompts import ChatPromptTemplate



# LLM 
llm = ChatOllama(model='qwen2.5:0.5b', temperature=0)

# embedding model
embedding_model = OllamaEmbeddings(model='bge-m3:567m')

# database
db = Chroma(persist_directory='./db/chroma_db', embedding_function=embedding_model)

# retriever
retriever = db.as_retriever(search_type='similarity', search_kwargs={'k':1})



# history aware retriever
contextualize_chat_history_prompt = [
    ('system', 'you are an AI assistant and your only job is to contextualize/summarize the given chat history. summarize the chat history in such a way that no crucial information is lost.'),
    MessagesPlaceholder('chat_history'),
    ('human', '{input}')
]
history_prompt_template = ChatPromptTemplate.from_messages(contextualize_chat_history_prompt)

history_aware_retriever = create_history_aware_retriever(llm, retriever, history_prompt_template)



qa_prompt = [
    ('system', "you are an AI assistant. your sole purpose is to generate responses to the user's query based on the given context.\n\n {context}"),
    MessagesPlaceholder('chat_history'),
    ('human', '{input}')
]
qa_prompt_template = ChatPromptTemplate.from_messages(qa_prompt)

question_answer_chain = create_stuff_documents_chain(llm, qa_prompt_template)



rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)



def chat_loop():
    """function to start chat loop"""
    print('you can start chatting now. type \'exit\' to stop chatting')
    
    chat_history = []   #to save chat history
    
    while True:
        query = input('USER : ')
        if query.lower() == 'exit':
            break
        result = rag_chain.invoke({'input':query, 'chat_history':chat_history})
        print(f"AI : {result['answer']}")
        
        # update chat history
        chat_history.append(HumanMessage(content=query))
        chat_history.append(AIMessage(content=result['answer']))

if __name__ == '__main__':
    chat_loop()