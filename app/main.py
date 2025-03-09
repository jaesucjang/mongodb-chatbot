from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from app.models import Document, Query
import os
from dotenv import load_dotenv
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores.mongodb_atlas import MongoDBAtlasVectorSearch
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from pymongo import MongoClient

load_dotenv()

app = FastAPI()

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 설정
MONGODB_URI = os.getenv("MONGODB_URI")
DB_NAME = "chatbot_db"
COLLECTION_NAME = "documents"

# MongoDB 클라이언트 설정
client = MongoClient(MONGODB_URI)
collection = client[DB_NAME][COLLECTION_NAME]

# LangChain 컴포넌트 설정
embeddings = OpenAIEmbeddings()
vectorstore = MongoDBAtlasVectorSearch(
    collection=collection,
    embedding=embeddings,
    index_name="vector_index",
)

@app.post("/documents")
async def add_document(document: Document):
    try:
        # 문서를 청킹하고 임베딩하여 MongoDB에 저장
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = text_splitter.split_text(document.text)
        
        # LangChain을 통해 벡터스토어에 문서 추가
        vectorstore.add_texts(chunks)
        
        return {"message": "Document added successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat")
async def chat(query: Query):
    try:
        if query.use_rag:
            # RAG 사용 (문서 검색 + LLM)
            qa = RetrievalQA.from_chain_type(
                llm=ChatOpenAI(model_name="gpt-3.5-turbo"),
                chain_type="stuff",
                retriever=vectorstore.as_retriever()
            )
            response = qa.run(query.text)
        else:
            # 단순 LLM 응답
            llm = ChatOpenAI(model_name="gpt-3.5-turbo")
            response = llm.predict(query.text)
            
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)