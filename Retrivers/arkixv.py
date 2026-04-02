from langchain_community.retrievers import ArxivRetriever

retriever =  ArxivRetriever(
     load_max_docs=2,      # number of papers to retrieve
     load_all_available_meta=True
)

docs = retriever.invoke("LLMs and their applications in various domains")
for i, doc in enumerate(docs):
    print(f"\nResult {i+1}")
    print("Title:", doc.metadata.get("Title"))
    print("Authors:", doc.metadata.get("Authors"))
    print("Summary:", doc.page_content[:100])  # print first 100 characters 