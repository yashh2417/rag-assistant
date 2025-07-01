import json
import sys
import os

# Add the parent directory (rag-assistant/) to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Now import from app
from app.app import model, vectorstore

from app import model, vectorstore  # reuse your loaded model and vectorstore
from langchain.chains import ConversationalRetrievalChain
from langchain.chains.conversation.memory import ConversationBufferMemory

EVAL_FILE = "tests/eval_queries.json"

def load_tests(path=EVAL_FILE):
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
