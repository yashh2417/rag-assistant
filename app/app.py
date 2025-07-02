from fastapi import FastAPI, Request, UploadFile, File
import os
import hashlib
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from google.ai.generativelanguage_v1beta.types import Tool as GenAITool
from langchain.chains import ConversationalRetrievalChain
from langchain.chains.conversation.memory import ConversationBufferMemory
from fastapi import Form
from pydantic import BaseModel
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from fastapi.responses import JSONResponse
from fastapi.responses import RedirectResponse
from pathlib import Path

## loading env variables
from dotenv import load_dotenv

load_dotenv()

## initializing fastapi
app = FastAPI()

## setting templates and static file
app.mount("/static", StaticFiles(directory="app/app/static"), name="static")
templates = Jinja2Templates(directory="app/app/templates")

## setting GOOGLE API KEY in the environment variable
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

## setting LANGSMITH API KEY in environment variable for tracking
os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
os.environ["LANGSMITH_TRACING"] = "true"

## setting PINECONE API KEY nad PINECONE ENVIRONMENT in the environment
os.environ["PINECONE_API_KEY"] = os.getenv("PINECONE_API_KEY")
os.environ["PINECONE_API_ENV"] = "us-east-1"

## initialising llm
from langchain_google_genai import ChatGoogleGenerativeAI
model = ChatGoogleGenerativeAI(model = "gemini-2.0-flash", convert_system_message_to_human=True)

## initializing embeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
gemini_embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001"
)

## initializing vectorstore (pinecone)
vectorstore = PineconeVectorStore(index_name="smart-assistant", embedding=gemini_embeddings)

## json files for hash and file name storing
import json

UPLOAD_TRACK_FILE = Path("uploaded_files.json")
HASH_TRACK_FILE = Path("uploaded_hashes.json")

## Load uploaded filenames
if UPLOAD_TRACK_FILE.exists():
    with open(UPLOAD_TRACK_FILE, "r") as f:
        uploaded_files = json.load(f)
else:
    uploaded_files = []

## Load uploaded file hashes
if HASH_TRACK_FILE.exists():
    with open(HASH_TRACK_FILE, "r") as f:
        stored_hashes = json.load(f)
else:
    stored_hashes = []

## Utility for hashing
def compute_file_hash(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()

## Load previously uploaded files (or start with empty list)
if UPLOAD_TRACK_FILE.exists():
    with open(UPLOAD_TRACK_FILE, "r") as f:
        uploaded_files = json.load(f)
else:
    uploaded_files = []

## function for loading and chunking the data
def data_loader_and_chunking(file_path, ext):

    if ext == "pdf":
        loader = PyMuPDFLoader(file_path)
        data = loader.load()

    elif ext == "csv":
        loader = CSVLoader(file_path=file_path)
        data = loader.load()

    else:
        data = []
    
    ## initializing RecursiveCharacterTextSplitter for chunking the data
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=100)
    text_chunks = text_splitter.split_documents(data)

    return text_chunks

## routes
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "uploaded_files": uploaded_files})

@app.post("/upload", response_class=HTMLResponse)
async def upload_document(request: Request, file: UploadFile = File(...)):
    contents = await file.read()
    ext = file.filename.split(".")[-1].lower()

    file_hash = compute_file_hash(contents)

    ## Skip if duplicate content already embedded
    if file_hash in stored_hashes:
        return RedirectResponse(url="/?msg=duplicate", status_code=303)

    ## Save the file temporarily
    upload_dir = Path("temp_uploads")
    upload_dir.mkdir(exist_ok=True)
    file_path = upload_dir / file.filename

    with open(file_path, "wb") as buffer:
        buffer.write(contents)

    try:
        text_chunks = data_loader_and_chunking(file_path, ext)
    finally:
        file_path.unlink()

    ## Embed and store metadata
    PineconeVectorStore.from_documents(
        text_chunks,
        index_name="smart-assistant",
        embedding=gemini_embeddings
    )

    ## Update JSON tracking files
    uploaded_files.append(file.filename)
    stored_hashes.append(file_hash)

    with open(UPLOAD_TRACK_FILE, "w") as f:
        json.dump(uploaded_files, f)

    with open(HASH_TRACK_FILE, "w") as f:
        json.dump(stored_hashes, f)

    return RedirectResponse(url="/", status_code=303)

## Chat endpoint
class ChatRequest(BaseModel):
    session_id: str
    message: str

## session id for chating
sessions = {}

@app.post("/chat")
async def chat(req: ChatRequest):
    session_id = req.session_id
    message = req.message

    if session_id not in sessions:
        ## memory for buffer memory
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        chain = ConversationalRetrievalChain.from_llm(
            llm=model,
            retriever=vectorstore.as_retriever(),
            memory=memory,
        )
        sessions[session_id] = chain

    chain = sessions[session_id]
    response = chain.run(message)

    return JSONResponse({"response": response})

