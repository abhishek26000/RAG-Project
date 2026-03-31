from dotenv import load_dotenv
load_dotenv()
from langchain_mistralai import ChatMistralAI

model = ChatMistralAI(model = "mistral-medium-2508")

result = model.invoke("what is rag")
print(result.content)
                      