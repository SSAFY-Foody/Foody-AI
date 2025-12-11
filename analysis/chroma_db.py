# prepare_diabetes_vectorstore.py
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings  # 혹은 HuggingFaceEmbeddings 등
from langchain_community.embeddings import HuggingFaceEmbeddings

# 벡터로 저장하기 위한 코드

#임베딩 모델
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


PDF_PATH = "./ref/당뇨병 진료지침.pdf"
DB_DIR = "./chroma_diabetes_guideline"

def build_vectorstore():
    loader = PyPDFLoader(PDF_PATH)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    )
    split_docs = splitter.split_documents(docs)

    vectordb = Chroma.from_documents(
        split_docs,
        embedding=embeddings,
        persist_directory=DB_DIR,
        collection_name="diabetes_guideline",
    )
    vectordb.persist()
    print("✅ 당뇨병 지침 벡터스토어 생성 완료")

if __name__ == "__main__":
    build_vectorstore()
