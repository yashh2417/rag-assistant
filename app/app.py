from fastapi import FastAPI, Request, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from pydantic import BaseModel
from pathlib import Path
from dotenv import load_dotenv
import os
import json
import hashlib

# LangChain / LLM imports
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.chains import ConversationalRetrievalChain
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain_community.document_loaders import PyMuPDFLoader, CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from motor.motor_asyncio import AsyncIOMotorClient
from google.api_core import exceptions

# Load environment variables
load_dotenv()

# Initialize FastAPI
app = FastAPI()

# Define base directory
BASE_DIR = Path(__file__).resolve().parent
 
# Mount static and templates
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
templates = Jinja2Templates(directory=BASE_DIR / "templates")

# temp dir
temp_dir = BASE_DIR / "temp_uploads"
temp_dir.mkdir(exist_ok=True)

# Load envs
MONGODB_URI = os.getenv("MONGODB_URI")
client = AsyncIOMotorClient(MONGODB_URI)
db_name = os.getenv("MONGODB_DB")
db = client[db_name]
uploads_collection = db["uploads"]

# Set environment variables for APIs
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "RAG Assistant"
os.environ["PINECONE_API_KEY"] = os.getenv("PINECONE_API_KEY")
os.environ["PINECONE_API_ENV"] = os.getenv("PINECONE_API_ENV", "us-east-1")

# Initialize LLM and embeddings
model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", convert_system_message_to_human=True)
gemini_embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# Initialize vector store
vectorstore = PineconeVectorStore(index_name="smart-assistant", embedding=gemini_embeddings)

def compute_file_hash(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()

async def is_duplicate(file_hash: str) -> bool:
    return await uploads_collection.find_one({"file_hash": file_hash}) is not None

async def record_upload(filename: str, file_hash: str):
    await uploads_collection.insert_one({"filename": filename, "file_hash": file_hash})

# Utility function to compute SHA256 hash of file
def compute_file_hash(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()

# Load and chunk the file content
def data_loader_and_chunking(file_path: str, ext: str) -> list:
    if ext == "pdf":
        loader = PyMuPDFLoader(file_path)
    elif ext == "csv":
        loader = CSVLoader(file_path=file_path)
    else:
        return []

    data = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=100)
    return splitter.split_documents(data)

# Routes
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    uploads = await uploads_collection.find().to_list(length=100)
    filenames = [u["filename"] for u in uploads]
    return templates.TemplateResponse("index.html", {"request": request, "uploaded_files": filenames})

@app.post("/upload", response_class=HTMLResponse)
async def upload_document(file: UploadFile = File(...)):
    contents = await file.read()
    ext = file.filename.split(".")[-1].lower()
    file_hash = compute_file_hash(contents)

    if await is_duplicate(file_hash):
        return RedirectResponse(url="/", status_code=303)

    file_path = temp_dir / file.filename
    file_path.write_bytes(contents)

    with open(file_path, "wb") as f:
        f.write(contents)

    try:
        chunks = data_loader_and_chunking(file_path, ext)
        PineconeVectorStore.from_documents(chunks, index_name="smart-assistant", embedding=gemini_embeddings)
    finally:
        file_path.unlink()

    await record_upload(file.filename, file_hash)
    return RedirectResponse(url="/", status_code=303)

# Chat request schema
class ChatRequest(BaseModel):
    session_id: str
    message: str

# In-memory session tracking
sessions = {}

@app.post("/chat")
async def chat(req: ChatRequest):
    if req.session_id not in sessions:
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        chain = ConversationalRetrievalChain.from_llm(
            llm=model,
            retriever=vectorstore.as_retriever(),
            memory=memory,
        )
        sessions[req.session_id] = chain

    chain = sessions[req.session_id]
    try:
        # result = chain.run(req.message)
        result = chain.invoke({"question": req.message})
        return JSONResponse({"response": result.get("answer", "Sorry, I could not process your request.")})
        # return JSONResponse({"response": result})
    except exceptions.ResourceExhausted as e:
        error_message = "I'm sorry, but I've exceeded my current usage limit for the AI model. Please check the service plan and billing details."
        return JSONResponse({"response": error_message}, status_code=429)

    except Exception as e:
        return JSONResponse({"response": f"Error: {str(e)}"}, status_code=500)
