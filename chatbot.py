import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA

def create_chatbot(pdf_text):
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_text(pdf_text)

    embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
    vectorstore = FAISS.from_texts(chunks, embedding=embeddings)

    llm = ChatOpenAI(temperature=0.2, model_name="gpt-3.5-turbo", openai_api_key=os.getenv("OPENAI_API_KEY"))
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever())

    return qa_chain
