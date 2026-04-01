from langchain_community.document_loaders import WebBaseLoader
from dotenv import load_dotenv
load_dotenv()

data = WebBaseLoader("https://www.samsung.com/in/multistore/in_corporate/prelogin/?cid=in_pd_search_bing_bing-EPP_sustain_b2b-q1-2026_text_na_1ur-464708l&msclkid=cad1d08931cf178a6145b94425f77330")

data = data.load()
print(data[0].page_content)
