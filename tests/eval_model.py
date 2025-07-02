import json
import os
from langchain.chains import ConversationalRetrievalChain
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_pinecone import PineconeVectorStore

from dotenv import load_dotenv

load_dotenv()

## setting GOOGLE API KEY in the environment variable
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

## setting LANGSMITH API KEY in environment variable for tracking
os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
os.environ["LANGSMITH_TRACING"] = "true"

## setting PINECONE API KEY nad PINECONE ENVIRONMENT in the environment
os.environ["PINECONE_API_KEY"] = os.getenv("PINECONE_API_KEY")
os.environ["PINECONE_API_ENV"] = "us-east-1"

model = ChatGoogleGenerativeAI(model = "gemini-2.0-flash", convert_system_message_to_human=True)

gemini_embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001"
)

## initializing vectorstore (pinecone)
vectorstore = PineconeVectorStore(index_name="smart-assistant", embedding=gemini_embeddings)

EVAL_FILE = "eval_queries.json"

def load_tests():
    path = os.path.join(os.path.dirname(__file__), EVAL_FILE)
    with open(path, "r") as f:
        return json.load(f)

def run_tests():
    tests = load_tests()
    passed = 0

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    chain = ConversationalRetrievalChain.from_llm(
        llm=model,
        retriever=vectorstore.as_retriever(),
        memory=memory,
    )

    for test in tests:
        query = test["query"]
        expected_keywords = test["expected_keywords"]

        result = chain.run(query)
        print(f"\n🟡 Query: {query}")
        print(f"📨 Response: {result}")

        if all(word.lower() in result.lower() for word in expected_keywords):
            print("✅ PASSED")
            passed += 1
        else:
            print(f"❌ FAILED. Missing keywords: {[w for w in expected_keywords if w.lower() not in result.lower()]}")

    print(f"\n🏁 {passed}/{len(tests)} tests passed.")

if __name__ == "__main__":
    run_tests()
