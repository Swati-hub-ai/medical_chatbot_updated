import os
import streamlit as st

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA

DB_FAISS_PATH = "vectorstore/db_faiss"

@st.cache_resource
def get_vectorstore():
    """Loads the FAISS vector database with embeddings."""
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    return db

def set_custom_prompt():
    """Returns a custom prompt template for retrieval-based QA."""
    custom_prompt_template = """
    Use the pieces of information provided in the context to answer the user's question.
    If you don’t know the answer, just say that you don’t know. Don’t try to make up an answer.
    Don’t provide anything out of the given context.

    Context: {context}
    Question: {question}

    Start the answer directly. No small talk please.
    """
    return PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])

def load_llm(model_name="llama3:latest"):
    """Loads the locally installed Llama 3 model using Ollama."""
    return OllamaLLM(model=model_name)

def main():
    # Title and Introduction Message
    st.title("Ask Chatbot!")
    
    # Greeting message at the start of the chat
    if 'greeted' not in st.session_state:
        st.session_state.greeted = True
        st.chat_message('assistant').markdown("**Hi, I am your Medical AI Assistant!** 😄 **How can I help you today?**")

    if 'messages' not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for message in st.session_state.messages:
        st.chat_message(message['role']).markdown(message['content'])

    # User input prompt
    prompt = st.chat_input("Ask your medical question here...")

    if prompt:
        # Display user message in chat
        st.chat_message('user').markdown(prompt)
        st.session_state.messages.append({'role': 'user', 'content': prompt})

        try:
            # Get the vector store (retriever for relevant documents)
            vectorstore = get_vectorstore()
            if vectorstore is None:
                st.error("Failed to load the vector store")
                return

            # Create the QA chain with Llama model and vectorstore retriever
            qa_chain = RetrievalQA.from_chain_type(
                llm=load_llm("llama3:latest"),  # Using local Llama 3 model from Ollama
                chain_type="stuff",
                retriever=vectorstore.as_retriever(search_kwargs={'k': 3}),
                return_source_documents=True,  # Still retrieving source docs, but we will exclude them from the result
                chain_type_kwargs={'prompt': set_custom_prompt()}
            )

            # Invoke the QA chain and get the result
            response = qa_chain.invoke({'query': prompt})
            result = response["result"]
            # Do not include source documents in the final response:
            result_to_show = f"{result}"

            # Display the assistant's response
            st.chat_message('assistant').markdown(result_to_show)
            st.session_state.messages.append({'role': 'assistant', 'content': result_to_show})

        except Exception as e:
            st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
