# 1. Import OS, Document Loader, Text Splitter, Bedrock Embeddings, Vector DB, VectorStoreIndex, Bedrock-LLM
import os
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import BedrockEmbeddings
from langchain.vectorstores import FAISS
from langchain.indexes import VectorstoreIndexCreator
from langchain.llms.bedrock import Bedrock


# 5c. Wrap within a function
def hr_index():
    # 2. Define the data source and load data with PDFLoader(https://www.upl-ltd.com/images/people/downloads/Leave-Policy-India.pdf)
    data_load = PyPDFLoader('https://www.upl-ltd.com/images/people/downloads/Leave-Policy-India.pdf')
    # data_test = data_load.load_and_split()
    # print(len(data_test)) # number of page
    # print(data_test[0]) # contents of page number 2

    # 3. Split the Text based on Character, Tokens etc. - Recursively split by character - ["\n\n", "\n", " ", ""]
    data_split = RecursiveCharacterTextSplitter(separators=["\n\n", "\n", " ", ""], chunk_size=100, chunk_overlap=10)
    # data_sample = "Welcome to the most comprehensive guide on Amazon Bedrock and Generative AI on AWS from a practising AWS Solution Architect and best-selling Udemy Instructor."
    # data_split_test = data_split.split_text(data_sample)
    # print(data_split_test)

    # 4. Create Embeddings -- Client connection
    data_embeddings = BedrockEmbeddings(
        # credential_profile_name="default",
        model_id="amazon.titan-embed-text-v1"
    )

    # 5a Create Vector DB, Store Embeddings and Index for Search - VectorstoreIndexCreator
    data_index = VectorstoreIndexCreator(
        text_splitter=data_split,
        embedding=data_embeddings,
        vectorstore_cls=FAISS)

    # 5b Create index for HR Report
    db_index = data_index.from_loaders([data_load])
    return db_index


# 6a. Write a function to connect to Bedrock Foundation Model
def hr_llm():
    llm = Bedrock(
        # credential_profile_name="default",
        model_id="anthropic.claude-v2",
        model_kwargs={
            "max_tokens_to_sample": 3000,
            "temperature": 0.1,
            "top_p": 0.9}
    )
    return llm


# 6b. Write a function which searches the user prompt, searches the best match from Vector DB and sends both to LLM.
def hr_rag_response(index, question):
    rag_llm = hr_llm()
    hr_rag_query = index.query(question=question, llm=rag_llm)
    return hr_rag_query
# Index creation --> https://api.python.langchain.com/en/latest/indexes/langchain.indexes.vectorstore.VectorstoreIndexCreator.ht
