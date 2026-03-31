from langchain_community.document_loaders import TextLoader

data = TextLoader("Documentloader/notes.txt")
data = data.load()
print(data[0])

