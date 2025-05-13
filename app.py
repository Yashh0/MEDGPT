import os
import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from groq import Groq
from dotenv import load_dotenv
import time

# Initialize Streamlit page configuration
st.set_page_config(
    page_title="MedGPT - Medical Knowledge Assistant",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main {padding: 2rem}
    .stTextInput {max-width: 800px}
    .stButton>button {background-color: #FF4B4B; color: white}
    .stButton>button:hover {background-color: #FF6B6B}
    .css-1y4p8pa {max-width: 800px}
    .st-emotion-cache-1y4p8pa {max-width: 800px}
</style>
""", unsafe_allow_html=True)

# Sidebar configuration
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/medical-doctor.png", width=100)
    st.header("🔑 API Configuration")
    
    api_key = st.text_input("Enter your Groq API Key:", type="password", help="Get your API key from Groq's website")
    if api_key:
        os.environ['GROQ_API_KEY'] = api_key
        st.success("✅ API Key set successfully!")
    else:
        # Try loading from .env file
        load_dotenv()
        api_key = os.getenv("GROQ_API_KEY")
        if api_key:
            st.success("✅ API Key loaded from .env file")
    
    st.markdown("---")
    st.markdown("### About")
    st.markdown("""
    MedGPT is your intelligent medical knowledge assistant.
    Ask any medical question and get detailed, accurate information from trusted sources.
    
    **Features:**
    - Comprehensive medical information
    - Evidence-based responses
    - Structured explanations
    """)

# Check for API key before proceeding
if not api_key:
    st.warning("Please enter your Groq API key in the sidebar to continue.")
    st.stop()

# Main app content
st.title("🏥 MedGPT - Medical Knowledge Assistant")
st.markdown("""<div style='margin-bottom: 2rem'>
Your intelligent companion for medical knowledge and information. 
Ask any medical question to get comprehensive, evidence-based answers.
</div>""", unsafe_allow_html=True)

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

    col1, col2 = st.columns([1, 6])
    with col1:
        if st.button("🔍 Search", use_container_width=True):
            if query:
                with st.spinner("🔄 Analyzing medical literature..."):
                    try:
                        answer = query_with_groq(query, retriever)
                        
                        # Display answer in a nice format
                        st.markdown("### Medical Information")
                        st.markdown("<div style='background-color: #f0f2f6; padding: 20px; border-radius: 10px;'>", unsafe_allow_html=True)
                        st.markdown(answer)
                        st.markdown("</div>", unsafe_allow_html=True)
                        
                        # Add timestamp
                        st.caption(f"Generated on {time.strftime('%Y-%m-%d %H:%M:%S')}")
                    except Exception as e:
                        st.error(f"An error occurred: {str(e)}. Please try again.")
                if answer:
                    st.success("Query processed successfully!")
        else:
            st.warning("Please enter a query.")

except Exception as e:
    st.error(f"Initialization error: {str(e)}")