from dotenv import load_dotenv
load_dotenv()
from langchain_mistralai import ChatMistralAI
from langchain_mistralai import MistralAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate

embeddings = MistralAIEmbeddings(model = "mistral-embed")

vectorstore = Chroma(
    persist_directory= "ChromaDB",
    embedding_function=embeddings
)


retriever = vectorstore.as_retriever(
    search_type = "mmr",
    search_kwargs = {"k": 4, "fetch_k": 10},
    lambda_mult = 0.4
)

llm = ChatMistralAI(model = "mistral-medium-2508")

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a helpful AI assistant.

Use ONLY the provided context to answer the question.

If the answer is not present in the context,
say: "I could not find the answer in the document."
"""
        ),
        (
            "human",
            """Context:
{context}

Question:
{question}
"""
        )
    ]
)


print("Welcome to the MistralAI PDF Question Answering System!")
print("Press 0 to exit:")

while True:
    query = input("You:")
    if query == "0":
        print("Exiting the system. Goodbye!")
        break
    
    docs = retriever.invoke(query)
    
    
    context = "\n\n".join(
        [doc.page_content for doc in docs]
    )
    
    final_prompt = prompt.invoke({
        "context" :context,
        "question": query
    })
    
    response = llm.invoke(final_prompt)

    print(f"\n AI: {response.content}")