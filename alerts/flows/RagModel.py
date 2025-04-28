#! /Users/jerald/opt/miniconda3/envs/llm/bin/python

from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough

import pickle 

documents = pickle.load(open('bin/alerts/flows/flow.pkl', 'rb'))

# Step 1: Convert text documents to LangChain Document objects
docs = [Document(page_content=text) for text in documents]

# Step 2: Set up embeddings and vector store
embeddings = OllamaEmbeddings(model="nomic-embed-text:latest")  # Use Llama3 for embeddings
vector_store = Chroma.from_documents(documents=docs, embedding=embeddings, collection_name="stock_data")

# Step 3: Set up the retriever
retriever = vector_store.as_retriever(search_kwargs={"k": 2})  # Retrieve top 2 relevant documents

# Step 4: Initialize the LLM
llm = Ollama(model="qwen2.5:14b-instruct-8k")

# Step 5: Define the prompt template for RAG
prompt_template = """
You are a financial analyst specializing in stock options data. Use the following retrieved data to generate a concise insight (1-2 sentences) for the given ticker, connecting the data to broader market trends or investor sentiment. If relevant, compare to similar stocks.

Retrieved Data:
{context}

Ticker: {ticker}
Open Interest: {oi:,}
Call OI: {call_oi:,}
Put OI: {put_oi:,}

Generate an insight for {ticker}:
"""
prompt = PromptTemplate(input_variables=["context", "ticker", "oi", "call_oi", "put_oi"], template=prompt_template)

# Step 6: Build the RAG chain
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "ticker": lambda x: x["ticker"], "oi": lambda x: x["oi"], "call_oi": lambda x: x["call_oi"], "put_oi": lambda x: x["put_oi"]}
    | prompt
    | llm
    | RunnablePassthrough()
)

# Step 7: Test the RAG model with a new ticker
new_data = {
    "ticker": "SMCI",
    "oi": 2592614,
    "call_oi": 1448053,
    "put_oi": 1144561
}

import pandas as pd 



response = rag_chain.invoke(new_data)

print("Generated Insight for SMCI:")
print(response)