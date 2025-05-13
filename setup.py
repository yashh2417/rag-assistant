from setuptools import setup, find_packages
  
setup( 
    name='mcqgenerator', 
    version='0.0.1', 
    author='Yash', 
    author_email='yashh2417@gmail.com', 
    install_requires=[ 
        "openai",
        "langchain",
        "streamlit",
        "python-dotenv",
        "PyPDF2",
        "langchain-deepseek"
    ], 
    packages=find_packages() 
) 