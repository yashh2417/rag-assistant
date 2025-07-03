---

# 🔍 Gemini RAG Assistant

A production-ready RAG (Retrieval-Augmented Generation) web application that leverages Google Gemini Pro for question answering using user-uploaded PDFs or CSVs. Built with **FastAPI**, **LangChain**, **Pinecone**, and **LangSmith** for robust LLMOps, and deployed using **Docker**, **GitHub Actions**, and **Render**.

---

## 🚀 Features

- 🔮 Uses **Gemini 2.0 Flash** for conversational LLM responses
- 📚 Upload **PDF** or **CSV** files to dynamically extend knowledge
- 🧠 Context storage using **Pinecone Vector Database**
- 🧩 Built with **LangChain** for easy chaining, Memory and RAG architecture
- 🧪 Tracks sessions, queries, cost, latency via **LangSmith**
- 🛠️ Built with **FastAPI** + **Jinja2 templates** for the web UI
- 🔐 Prevents duplicate uploads using **SHA256 content hashing**
- 📦 Automatically tested and deployed via **GitHub Actions**
- 🐳 Packaged in a **Docker** container and deployed to **Render**

---

## 🖼️ UI Screenshots

- **Home Page**

![Home Page](https://github.com/yashh2417/faltu/blob/main/home-rag.png?raw=true)

---
````
## 🧱 Project Structure

.
├── app/
│   ├── main.py                  # FastAPI backend logic
│   ├── templates/               # Jinja2 HTML templates
│   ├── static/                  # Static CSS/JS/images
│   ├── uploaded_files.json      # Tracks uploaded filenames
│   ├── uploaded_hashes.json     # Tracks file hashes to prevent duplicates
│   └── temp_uploads/            # Temporary storage during processing
├── requirements.txt
├── .env                         # API keys and secrets (not committed)
├── Dockerfile
├── .github/
│   └── workflows/
│       └── deploy.yaml          # GitHub Actions CI/CD pipeline
└── README.md

````

---

## 🧠 LLM Stack Overview

| Component          | Tech Used                                    |
| ------------------ | -------------------------------------------- |
| Language Model     | Gemini 2.0 Flash (`ChatGoogleGenerativeAI`)  |
| Embeddings         | `GoogleGenerativeAIEmbeddings`               |
| Vector Store       | Pinecone                                     |
| Chunking           | LangChain's `RecursiveCharacterTextSplitter` |
| RAG Chain          | `ConversationalRetrievalChain`               |
| Memory             | `ConversationBufferMemory`                   |
| Tracing/Monitoring | LangSmith                                    |

---

## 📊 LangSmith Integration

LangSmith is used to trace and monitor:

* ✅ Query and response pairs
* 🕒 Latency per interaction
* 💰 Token usage and cost
* 🧵 Session chat history

Make sure your `.env` file contains:

```env
LANGSMITH_API_KEY=your_langsmith_key
LANGSMITH_TRACING=true
```

---

## 🔧 Requirements

Install all dependencies via:

```bash
pip install -r requirements.txt
```

---

## 📄 .env Configuration

Make sure you create a `.env` file in the root directory with:

```env
GOOGLE_API_KEY=your_google_api_key
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_API_ENV=us-east-1
LANGSMITH_API_KEY=your_langsmith_api_key
LANGSMITH_TRACING=true
```

---

## 🔬 Testing & CI/CD

This project uses GitHub Actions to:

1. Run 3 test cases for app health check and upload/chat routes
2. On successful tests, build a Docker image
3. Push the image to DockerHub automatically

Example workflow file: `.github/workflows/deploy.yaml`

---

## 🐳 Docker Build & Deployment

### Build Locally

```bash
docker build -t gemini-rag-assistant .
docker run -p 8000:8000 gemini-rag-assistant
```

### Automatic Deployment

After CI tests pass:

* Docker image is built using `requirements.txt`
* Image is pushed to DockerHub
* Deploy on **Render.com** with correct API keys injected via Render Dashboard

---

## 🌐 Live Deployment

> direct link : https://rag-assistant-diwc.onrender.com/

---

## 🧪 Example Chat Session

- **Live Conversation**

![Live convo](https://github.com/yashh2417/faltu/blob/main/convo-rag.png?raw=true)

---

## 🛡️ File Deduplication

Files are hashed using SHA256 to avoid re-uploading and re-indexing the same file with different names.

---

## 🔂 LangSmith Tracing

- **Language logs**

![Language logs](https://github.com/yashh2417/faltu/blob/main/langsmith-rag.png?raw=true)

---

## 📦 Future Enhancements

* ✅ Add authentication
* 🔍 Support other formats like DOCX
* 📈 Usage analytics dashboard
* 💬 WebSocket support for live chat streaming

---

## 📹 Reference Video

- **Youtube Video**

[Rag-Assistant-Video](https://youtu.be/UgsadziomOc?si=-V-qnAwtc1EIrzhx)

---

## 🧑‍💻 Maintainer

* **Name:** Yash
* **GitHub:** [yashh2417](https://github.com/yashh2417/)
* **LinkedIn:** [yashh2417](https://www.linkedin.com/in/yashh2417?lipi=urn%3Ali%3Apage%3Ad_flagship3_profile_view_base_contact_details%3BKhtZ14%2FjRHm5TYOH0ezmVQ%3D%3D)
* **Email:** [yashh2417@gmail.com](mailto:yashh2417@gmail.com)

---

## 📝 License

MIT License – feel free to use and modify.

---

