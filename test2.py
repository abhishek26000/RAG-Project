from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from dotenv import load_dotenv
load_dotenv()
data = PyPDFLoader("DocumentLoader/ab.pdf")
docs = data.load()
print(docs[1])