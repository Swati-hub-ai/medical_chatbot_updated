import os
import streamlit as st
from dotenv import load_dotenv, find_dotenv

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

# ----------------------------
# 🔹 Load Environment Variables
# ----------------------------
load_dotenv(find_dotenv())
HF_TOKEN = os.getenv("HF_TOKEN")
HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"

if not HF_TOKEN:
    st.error("❌ HF_TOKEN not found in .env.")
    st.stop()

# ----------------------------
# 🔹 Streamlit Page Setup
# ----------------------------
st.set_page_config(page_title="🧠 Medical RAG Chatbot", layout="wide")
st.title("💬 Conversational Medical Chatbot")

# ✅ Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ----------------------------
# 🔹 Load LLM
# ----------------------------
def load_llm():
    return HuggingFaceEndpoint(
        repo_id=HUGGINGFACE_REPO_ID,
        task="text-generation",
        temperature=0.5,
        max_new_tokens=512,
        huggingfacehub_api_token=HF_TOKEN
    )

# ----------------------------
# 🔹 Prompt Template
# ----------------------------
CUSTOM_PROMPT = """
Use the pieces of information provided in the context to answer the user's question.
If you don't know, just say you don't know. Don't try to make up an answer.
Only use the context provided.

Context: {context}
Question: {question}

Answer:
"""

def set_prompt(template):
    return PromptTemplate(template=template, input_variables=["context", "question"])

# ----------------------------
# 🔹 Load FAISS Vector DB
# ----------------------------
DB_FAISS_PATH = "vectorstore/db_faiss"
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)

# ----------------------------
# 🔹 Memory for Chat History
# ----------------------------
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
    output_key="answer"
)

# ----------------------------
# 🔹 Create Conversational RAG Chain
# ----------------------------
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=load_llm(),
    retriever=vector_db.as_retriever(search_kwargs={"k": 3}),
    memory=memory,
    condense_question_prompt=set_prompt(CUSTOM_PROMPT),
    return_source_documents=True,
    output_key="answer"
)

# ----------------------------
# 🔹 Show Chat History in Streamlit
# ----------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ----------------------------
# 🔹 Handle User Input & Response
# ----------------------------
if prompt := st.chat_input("Ask me anything medical..."):
    # Display user prompt
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Process with RAG pipeline
    with st.spinner("💡 Thinking..."):
        response = qa_chain.invoke({"question": prompt})
        answer = response["answer"]

    # ✅ Display only the answer
    st.session_state.messages.append({"role": "assistant", "content": answer})
    with st.chat_message("assistant"):
        st.markdown(answer)
