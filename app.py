from langchain_community.vectorstores.qdrant import Qdrant
from qdrant_client import QdrantClient
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
import openai

import os
from dotenv import load_dotenv

# from upload import url, qdrant, embeddings

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

model_name = "BAAI/bge-large-en"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': False}
embeddings = HuggingFaceBgeEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

url = "http://localhost:6333"

client = QdrantClient(
    url=url, prefer_grpc=False
)

print(client)
print("##############")

db = Qdrant(client=client, embeddings=embeddings, collection_name="vector_db")

print(db)
print("##############")

query = "What is laryngeal cancer?"

def rag(db,query):
    context_main = db.similarity_search(query=query)
    context = context_main[0].page_content

    prompt=f"""
    Follow exactly those 3 steps:
    1. Read the context below and aggregrate this data
    Context : {context}
    2. Answer the question using only this context and do not try to answer on your own
    User Question: {query} """

    response = openai.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": f"""You are an helpful assistant who is does exactly what it
                      is asked to do."""},
                    {"role": "user", "content": prompt}
                ]
                )
    
    return {response.choices[0].message.content}


print(query)
print(rag(db,query))