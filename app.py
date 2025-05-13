import os
import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from groq import Groq
from dotenv import load_dotenv
import time
from datetime import datetime

# Initialize Streamlit page configuration
st.set_page_config(
    page_title="MedGPT - Medical Knowledge Assistant",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for ChatGPT-like interface
st.markdown("""
<style>
    .main {padding: 1rem; max-width: 800px; margin: 0 auto;}
    
    /* Chat container */
    .chat-container {max-width: 800px; margin: 0 auto;}
    
    /* Message styling */
    .user-message {background-color: #f7f7f8; padding: 1rem; border-radius: 0.5rem; margin: 1rem 0;}
    .assistant-message {background-color: white; padding: 1rem; border-radius: 0.5rem; margin: 1rem 0;}
    
    /* Input area */
    .stTextInput {margin-bottom: 1rem;}
    .stTextInput>div>div>input {border-radius: 0.5rem; padding: 0.75rem;}
    
    /* Send button */
    .stButton>button {border-radius: 0.5rem; padding: 0.5rem 1rem;}
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Timestamp */
    .timestamp {color: #666; font-size: 0.8rem; margin-top: 0.25rem;}
    
    /* Avatar */
    .avatar {font-size: 1.5rem; margin-right: 0.5rem;}
    .message-header {display: flex; align-items: center; margin-bottom: 0.5rem;}
</style>
""", unsafe_allow_html=True)

# Load API key from Streamlit secrets
try:
    api_key = st.secrets.api_keys.groq
except Exception:
    st.error("""🔑 API key not found! 

To run this app:

1. For local development:
   - Create `.streamlit/secrets.toml`
   - Add your API key:
     ```toml
     [api_keys]
     groq = "your-api-key-here"
     ```

2. For Streamlit Cloud:
   - Go to your app settings
   - Add your API key in the secrets section using the same format
    """)
    st.stop()

# Initialize session state for chat history
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Display header
st.markdown('<h1 style="text-align: center; margin-bottom: 2rem;">🏥 MedGPT</h1>', unsafe_allow_html=True)

# Display chat messages
for message in st.session_state.messages:
    role = message["role"]
    content = message["content"]
    timestamp = message.get("timestamp", "")
    
    if role == "user":
        st.markdown(f"""
        <div class="user-message">
            <div class="message-header">
                <span class="avatar">👤</span>
                <strong>You</strong>
            </div>
            {content}
            <div class="timestamp">{timestamp}</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="assistant-message">
            <div class="message-header">
                <span class="avatar">🏥</span>
                <strong>MedGPT</strong>
            </div>
            {content}
            <div class="timestamp">{timestamp}</div>
        </div>
        """, unsafe_allow_html=True)





try:
    # Set up the embeddings
    model_name = "BAAI/bge-large-en"
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': False}
    embeddings = HuggingFaceBgeEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )

    # Load the vector store from the local drive
    script_dir = os.path.dirname(os.path.abspath(__file__))
    persist_directory = os.path.join(script_dir, 'Embedded_Med_books')
    
    # System status in sidebar
    with st.sidebar:
        st.markdown("---")
        st.markdown("### System Status")
        status_placeholder = st.empty()
        
        system_status = {
            "Vector Store": "✅ Connected" if os.path.exists(persist_directory) else "❌ Not Found",
            "API Key": "✅ Loaded" if api_key else "❌ Missing",
            "Model": "LLama3-70B"
        }
        
        for k, v in system_status.items():
            st.markdown(f"**{k}:** {v}")

    # Check vector store directory
    if not os.path.exists(persist_directory):
        st.error(f"Vector store directory not found at: {persist_directory}")
        if st.button("Create Directory"):
            os.makedirs(persist_directory)
            st.success("Directory created!")
    
    # Initialize vector store with error handling
    try:
        vector_store = Chroma(
            persist_directory=persist_directory,
            embedding_function=embeddings
        )
        retriever = vector_store.as_retriever(search_kwargs={'k': 1})
    except Exception as e:
        st.error(f"Error initializing vector store: {str(e)}")
        st.info("Creating a new vector store...")
        try:
            # Create directory if it doesn't exist
            os.makedirs(persist_directory, exist_ok=True)
            # Initialize empty vector store
            vector_store = Chroma(
                persist_directory=persist_directory,
                embedding_function=embeddings
            )
            retriever = vector_store.as_retriever(search_kwargs={'k': 1})
            st.success("Vector store initialized successfully!")
        except Exception as e:
            st.error(f"Failed to create vector store: {str(e)}")
            st.stop()
    
    # Initialize Groq client
    client = Groq(api_key=api_key)

    # Streamlit input
    # Query input with example
    st.markdown("### Ask Your Medical Question")
    query = st.text_input(
        "",
        placeholder="e.g., What are the symptoms of type 2 diabetes?",
        help="Enter any medical question you'd like to know about"
    )

    def query_with_groq(query, retriever):
        try:
            # Retrieve relevant documents with error handling
            try:
                docs = retriever.get_relevant_documents(query)
                if not docs:
                    return "I apologize, but I couldn't find any relevant medical information in my knowledge base to answer your question accurately. Please try rephrasing your question or ask something else."
                context = "\n".join([doc.page_content for doc in docs])
            except Exception as e:
                st.error(f"Error retrieving documents: {str(e)}")
                return "I encountered an error while searching the medical knowledge base. Please try again or rephrase your question."

            # Call the Groq API with the query and context
            completion = client.chat.completions.create(
                model="llama3-70b-8192",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a knowledgeable medical assistant. For any medical term or disease, include comprehensive information covering: "
                            "definitions, types, historical background, major theories, known causes, and contributing risk factors. "
                            "Explain the genesis or theories on its origin, if applicable. Use a structured, thorough approach and keep language accessible. "
                            "provide symptoms, diagnosis, and treatment and post operative care , address all with indepth explanation , with specific details and step-by-step processes where relevant. "
                            "If the context does not adequately cover the user's question, respond with: 'I cannot provide an answer based on the available medical dataset.'"
                        )
                    },
                    {
                        "role": "system",
                        "content": (
                            "If the user asks for a medical explanation, ensure accuracy, don't include layman's terms if complex terms are used, "
                            "and organize responses in a structured way."
                        )
                    },
                    {
                        "role": "system",
                        "content": (
                            "When comparing two terms or conditions, provide a clear, concise, and structured comparison. Highlight key differences in their "
                            "definitions, symptoms, causes, diagnoses, and treatments with indepth explanation of each. If relevant, include any overlapping characteristics."
                        )
                    },
                    {
                        "role": "user",
                        "content": f"{context}\n\nQ: {query}\nA:"
                    }
                ],
                temperature=0.7,
                max_tokens=3000,
                stream=True
            )

            # Create a placeholder for streaming response
            response_container = st.empty()
            response = ""
            
            # Stream the response
            for chunk in completion:
                if chunk.choices[0].delta.content:
                    response += chunk.choices[0].delta.content
                    response_container.markdown(response)
            
            return response
            
        except Exception as e:
            st.error(f"Error during query processing: {str(e)}")
            return None

except Exception as e:
    st.error(f"Initialization error: {str(e)}")