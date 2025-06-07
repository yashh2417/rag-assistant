from setuptools import setup, find_packages

setup(
    name='mcqgenerator',
    version='0.0.1',
    description='generate mcq on the basis of context given.',
    author='Yash',
    author_email='yashh2417@gmail.com',
    packages=find_packages(),
    install_requires=[
        'streamlit',
        'google-generativeai',
        'python-dotenv',
        'langchain',
        'PyPDF2',
        'chromadb',
        'faiss-cpu',
        'langchain_google_genai',
        'fastapi',
        'pydantic'
    ],
    python_requires='>=3.8',
)

-e .
