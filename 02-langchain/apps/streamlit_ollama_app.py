import os 
from dotenv import load_dotenv
from langchain_community.llms import Ollama
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
load_dotenv()

os.environ['LANGSMITH_API_KEY'] = os.getenv("LANGSMITH_API_KEY")
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_PROJECT"] = os.getenv("LANGSMITH_PROJECT")

## Prompt Template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant . Please respond to the question asked"),
    ("user", "{question}")
])

## streamlit framework 
st.title("Langchain with gemma2 ")
input_text = st.text_input("Enter your question here")

### Ollama Gemma2 LLM
llm = Ollama(model="gemma:2b")
output_parser = StrOutputParser()
chain = prompt | llm | output_parser

if input_text:
    response = chain.invoke({"question": input_text})
    st.write("Response: ", response)