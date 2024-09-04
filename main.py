import os

import streamlit as st
from groq import Groq

from dotenv import load_dotenv


from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, ServiceContext  #Vector store index is for indexing the vector
from llama_index.llms.huggingface import HuggingFaceLLM

from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from llama_index.core import Settings
from langchain.embeddings.huggingface import HuggingFaceEmbeddings


documents = SimpleDirectoryReader('./src/pdfs').load_data()

system_prompt="""
You are a Q&A assistant. Your goal is to answer questions as
accurately as possible based on the instructions and context provided.
"""

import torch

llm = HuggingFaceLLM(
    context_window=4096,
    max_new_tokens=256,
    generate_kwargs={"temperature": 0.0, "do_sample": False},
    system_prompt=system_prompt,
    tokenizer_name="meta-llama/Meta-Llama-3.1-8B",
    model_name="meta-llama/Meta-Llama-3.1-8B",
    device_map="auto",
    # loading model in 8bit for reducing memoru
    model_kwargs={"torch_dtype": torch.float16 , "load_in_8bit":True}
) 

embed_model= HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

# settings=Settings(
#     chunk_size=1024,
#     llm=llm,
#     embed_model=embed_model
# )
Settings.llm = llm
Settings.embed_model = embed_model
Settings.chunk_size = 1024
settings=Settings

index=VectorStoreIndex.from_documents(documents,settings=settings)
query_engine=index.as_query_engine()


# streamlit page configuration
st.set_page_config(
    page_title="LLAMA 3.1. Chat",
    page_icon="🦙",
    layout="centered"
)

working_dir = os.path.dirname(os.path.abspath(__file__))

load_dotenv()

GROQ_API_KEY=os.getenv('GROQ_API_KEY')


# save the api key to environment variable
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

client = Groq()

# initialize the chat history as streamlit session state of not present already
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# streamlit page title
st.title("🦙 LLAMA 3.1. ChatBot")

# display chat history
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# input field for user's message:
user_prompt = st.chat_input("Ask LLAMA...")

if user_prompt:

    st.chat_message("user").markdown(user_prompt)
    st.session_state.chat_history.append({"role": "user", "content": user_prompt})

    # sens user's message to the LLM and get a response
    messages = [
        {"role": "system", "content": "You are a helpful assistant"},
        *st.session_state.chat_history
    ]

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=messages
    )
    
    response = query_engine.query(user_prompt)

    st.session_state.chat_history.append({"role": "assistant", "content": response})

    # display the LLM's response
    with st.chat_message("assistant"):
        st.markdown(response) 