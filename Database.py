from langchain_community.document_loaders import PyPDFLoader
from langchain_mistralai import MistralAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
load_dotenv()


data = PyPDFLoader("DocumentLoader/deeplearning.pdf")
docs = data.load()
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=20
)
chunks = splitter.split_documents(docs)

embeddings = MistralAIEmbeddings(model = "mistral-embed")

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory= "ChromaDB"
)
