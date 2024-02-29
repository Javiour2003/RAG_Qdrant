from langchain_community.vectorstores.qdrant import Qdrant
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("cancer_test.pdf")
documents = loader.load_and_split()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

chunks = text_splitter.split_documents(documents)

model_name = "BAAI/bge-large-en"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': False}
embeddings = HuggingFaceBgeEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)


print("Creating embeddings....")

url = "http://localhost:6333"

qdrant = Qdrant.from_documents(
    chunks,
    embeddings,
    url=url,
    prefer_grpc=False,
    collection_name="vector_db"
)

print("Vector DB successfully created")