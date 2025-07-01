from setuptools import setup, find_packages

setup(
    name='smart-assistant',
    version='0.0.1',
    description='provide answers based on given context documents.',
    author='Yash',
    author_email='yashh2417@gmail.com',
    packages=find_packages(),
    install_requires=[
        'streamlit',
        'google-generativeai',
        'python-dotenv',
        'langchain',
        'PyPDF2',
        'langchain_google_genai',
        'fastapi',
        'pydantic'
    ],
    python_requires='>=3.8',
)

