import os
import shutil
import warnings
from typing import List
from fastapi import FastAPI, UploadFile, Form, Request, File
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from dotenv import load_dotenv
from PyPDF2 import PdfReader

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

warnings.filterwarnings('ignore')
load_dotenv()

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

UPLOAD_FOLDER = "./pdf_docs"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
gemini_embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001", google_api_key=GOOGLE_API_KEY
)
model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", convert_system_message_to_human=True)

def load_pdfs(folder):
    texts = []
    for file in os.listdir(folder):
        if file.endswith(".pdf"):
            reader = PdfReader(os.path.join(folder, file))
            full_text = ""
            for page in reader.pages:
                if page_text := page.extract_text():
                    full_text += page_text
            texts.append(full_text)
    return texts

def create_documents_from_texts(texts):
    return [Document(page_content=text, metadata={"source": f"doc_{i}"}) for i, text in enumerate(texts)]

def build_rag_chain(folder_path):
    texts = load_pdfs(folder_path)
    docs = create_documents_from_texts(texts)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    vectorstore = Chroma.from_documents(splits, embedding=gemini_embeddings)
    retriever = vectorstore.as_retriever()

    system_prompt = (
        "You are an assistant for question answering tasks. "
        "Use the following pieces of retrieved context to answer the question. "
        "If you don't know the answer, say that you don't know. "
        "Use three sentences maximum and keep the answer concise.\n\n{context}"
    )

    chat_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}")
    ])

    qa_chain = create_stuff_documents_chain(model, chat_prompt)
    rag_chain = create_retrieval_chain(retriever, qa_chain)
    return rag_chain

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("form.html", {"request": request})

@app.post("/process", response_class=HTMLResponse)
async def process(
    request: Request,
    files: List[UploadFile] = File(...),
    query: str = Form(...)
):
    # Clear existing PDFs
    for f in os.listdir(UPLOAD_FOLDER):
        if f.endswith(".pdf"):
            os.remove(os.path.join(UPLOAD_FOLDER, f))

    # Save uploaded PDFs
    for file in files:
        file_location = os.path.join(UPLOAD_FOLDER, file.filename)
        with open(file_location, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

    # Build RAG and get result
    rag_chain = build_rag_chain(UPLOAD_FOLDER)
    result = rag_chain.invoke({"input": query})["answer"]

    return templates.TemplateResponse("form.html", {
        "request": request,
        "result": result
    })
