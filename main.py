from dotenv import load_dotenv
load_dotenv()
from langchain_mistralai import ChatMistralAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate

data = PyPDFLoader("DocumentLoader/ab.pdf")
docs = data.load()
template = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant that answers questions based on this context: "),
    ("human", "{data}")
])

model = ChatMistralAI(model = "mistral-medium-2508")

prompt =  template.format_messages(data = docs[0].page_content)
result = model.invoke(prompt)
print(result.content)
                      