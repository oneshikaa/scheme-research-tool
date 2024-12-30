import streamlit as st
import pickle
import os
from langchain.document_loaders import UnstructuredURLLoader, SeleniumURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains.question_answering import load_qa_chain
import time
import pandas as pd
from typing import List, Dict, Any
import logging
from pathlib import Path
import faiss
import requests
from bs4 import BeautifulSoup

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Define custom prompt templates
SCHEME_QA_TEMPLATE = """
You are a knowledgeable assistant specializing in Indian government schemes. Use the following context to answer questions about the scheme, providing detailed and accurate information.

Context: {context}

Question: {question}

If the information is not explicitly mentioned in the context, indicate that it's not specified in the source material. However, make sure to provide all available relevant information from the context.

Answer:"""

CONDENSE_QUESTION_TEMPLATE = """
Given the following conversation history and a new question, rephrase the new question to be a standalone question that captures all necessary context.

Chat History: {chat_history}
New Question: {question}

Standalone Question:"""

class SchemeResearchTool:
    def __init__(self):
        """Initialize the Scheme Research Tool with necessary configurations."""
        try:
            self.setup_streamlit_page()
            self.setup_data_directory()
            self.load_api_key()
            self.initialize_session_state()
            self.load_existing_index()
            self.setup_qa_templates()
        except Exception as e:
            logger.error(f"Initialization error: {str(e)}")
            st.error(f"Error during initialization: {str(e)}")

    def setup_qa_templates(self):
        """Setup QA templates for better response generation."""
        self.qa_template = PromptTemplate(
            template=SCHEME_QA_TEMPLATE,
            input_variables=["context", "question"]
        )
        self.condense_question_template = PromptTemplate(
            template=CONDENSE_QUESTION_TEMPLATE,
            input_variables=["chat_history", "question"]
        )

    def setup_streamlit_page(self):
        """Configure the Streamlit page settings."""
        try:
            st.set_page_config(
                page_title="Scheme Research Tool",
                page_icon="📚",
                layout="wide",
                initial_sidebar_state="expanded"
            )
            st.title("Government Scheme Research Assistant")
        except Exception as e:
            logger.error(f"Page setup error: {str(e)}")
            raise

    def setup_data_directory(self):
        """Create data directory if it doesn't exist."""
        try:
            self.data_dir = Path("data")
            self.data_dir.mkdir(exist_ok=True)
            self.faiss_index_path = self.data_dir / "faiss_store_openai.pkl"
            self.urls_log_path = self.data_dir / "processed_urls.txt"
            logger.info(f"Data directory setup at {self.data_dir}")
        except Exception as e:
            logger.error(f"Data directory setup error: {str(e)}")
            raise

    def load_api_key(self):
        """Load OpenAI API key from configuration."""
        try:
            self.api_key = st.secrets.get("OPENAI_API_KEY", None)
            if not self.api_key:
                self.api_key = os.getenv("OPENAI_API_KEY")
            if not self.api_key:
                self.api_key = st.sidebar.text_input(
                    "Enter your OpenAI API key:",
                    type="password",
                    help="Enter your OpenAI API key. It will not be stored permanently."
                )
                if self.api_key:
                    os.environ["OPENAI_API_KEY"] = self.api_key
                    logger.info("API key set from user input")
            
            if not self.api_key:
                st.warning("Please provide an OpenAI API key to continue.")
        except Exception as e:
            logger.error(f"API key loading error: {str(e)}")
            raise

    def initialize_session_state(self):
        """Initialize Streamlit session state variables."""
        try:
            if 'processed_urls' not in st.session_state:
                st.session_state.processed_urls = self.load_processed_urls()
            if 'qa_chain' not in st.session_state:
                st.session_state.qa_chain = None
            if 'chat_history' not in st.session_state:
                st.session_state.chat_history = []
            logger.info("Session state initialized")
        except Exception as e:
            logger.error(f"Session state initialization error: {str(e)}")
            raise

    def load_processed_urls(self) -> set:
        """Load previously processed URLs from file."""
        try:
            if self.urls_log_path.exists():
                with open(self.urls_log_path, "r") as f:
                    urls = set(line.strip() for line in f if line.strip())
                logger.info(f"Loaded {len(urls)} processed URLs")
                return urls
            return set()
        except Exception as e:
            logger.error(f"Error loading processed URLs: {str(e)}")
            return set()

    def save_processed_urls(self, urls: set):
        """Save processed URLs to file."""
        try:
            with open(self.urls_log_path, "w") as f:
                for url in urls:
                    f.write(f"{url}\n")
            logger.info(f"Saved {len(urls)} processed URLs")
        except Exception as e:
            logger.error(f"Error saving processed URLs: {str(e)}")
            st.error("Failed to save processed URLs")

    def save_faiss_index(self):
        """Save FAISS index with proper serialization."""
        try:
            if self.vector_store is None:
                logger.warning("No vector store to save")
                return

            serialized_data = {
                'index': faiss.serialize_index(self.vector_store.index),
                'docstore': self.vector_store.docstore._dict,
                'embedding_function': None  # We'll recreate this when loading
            }
            
            with open(self.faiss_index_path, "wb") as f:
                pickle.dump(serialized_data, f)
                
            logger.info("Successfully saved FAISS index")
        except Exception as e:
            logger.error(f"Error saving FAISS index: {str(e)}")
            raise

    def load_existing_index(self):
        """Load existing FAISS index if available."""
        try:
            if self.faiss_index_path.exists():
                with open(self.faiss_index_path, "rb") as f:
                    serialized_data = pickle.load(f)
                
                embeddings = OpenAIEmbeddings()
                index = faiss.deserialize_index(serialized_data['index'])
                
                self.vector_store = FAISS(
                    embedding_function=embeddings,
                    index=index,
                    docstore=serialized_data.get('docstore', {})
                )
                logger.info("Loaded existing FAISS index")
            else:
                self.vector_store = None
                logger.info("No existing FAISS index found - will create new one when processing URLs")
        except Exception as e:
            logger.error(f"Unexpected error loading FAISS index: {str(e)}")
            self.vector_store = None

    def validate_url(self, url: str) -> bool:
        """Validate URL format."""
        try:
            if not url.startswith(('http://', 'https://')):
                return False
            return True
        except Exception:
            return False

    def extract_content(self, url: str) -> str:
        """Extract content from URL using both Selenium and BeautifulSoup with improved extraction."""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Connection': 'keep-alive',
            }
            
            response = requests.get(url, headers=headers, timeout=15)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Remove unwanted elements
            for element in soup(['script', 'style', 'nav', 'footer', 'iframe', 'meta']):
                element.decompose()
            
            # Extract main content with improved selector targeting
            content_elements = []
            
            # Get main content areas
            main_content = soup.find_all(['main', 'article', 'section', 'div'], 
                                      class_=lambda x: x and any(term in str(x).lower() 
                                                               for term in ['content', 'main', 'article', 'body']))
            
            if main_content:
                for element in main_content:
                    # Extract text from paragraphs and headers
                    content_elements.extend(element.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'li']))
            else:
                # Fallback to direct element extraction
                content_elements = soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'li'])
            
            # Process and clean the extracted text
            content = ' '.join(
                element.get_text(strip=True, separator=' ')
                for element in content_elements
                if element.get_text(strip=True)
            )
            
            # Additional cleaning
            content = ' '.join(content.split())  # Remove extra whitespace
            
            if not content:
                logger.warning(f"No content extracted from {url}")
                return ""
                
            logger.info(f"Successfully extracted {len(content)} characters from {url}")
            return content
            
        except Exception as e:
            logger.error(f"Error extracting content from {url}: {str(e)}")
            return ""

    def process_urls(self, urls: List[str]) -> None:
        """Process URLs to extract and index content."""
        try:
            valid_urls = [url for url in urls if self.validate_url(url)]
            if not valid_urls:
                st.error("No valid URLs provided. Please ensure URLs are accessible.")
                return

            new_urls = [url for url in valid_urls if url not in st.session_state.processed_urls]
            if not new_urls:
                st.warning("No new URLs to process")
                return

            with st.spinner("Processing URLs..."):
                # Create text splitter outside the URL loop
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,  # Increased chunk size
                    chunk_overlap=200,  # Increased overlap
                    separators=["\n\n", "\n", ". ", " ", ""],
                    length_function=len,
                    is_separator_regex=False
                )
                
                all_texts = []
                for url in new_urls:
                    content = self.extract_content(url)
                    if content:
                        # Add URL context to the content
                        content = f"Source URL: {url}\n\n{content}"
                        chunks = text_splitter.split_text(content)
                        # Add chunks to all_texts
                        all_texts.extend(chunks)
                
                if not all_texts:
                    st.error("Could not extract content from URLs")
                    return

                embeddings = OpenAIEmbeddings()
                
                # Create metadata for each chunk of text
                metadatas = []
                for i, chunk in enumerate(all_texts):
                    # Find which URL this chunk came from based on the index
                    url_index = 0
                    current_count = 0
                    for url_idx, url in enumerate(new_urls):
                        chunks_for_this_url = len(text_splitter.split_text(self.extract_content(url)))
                        if current_count + chunks_for_this_url > i:
                            url_index = url_idx
                            break
                        current_count += chunks_for_this_url
                    metadatas.append({"source": new_urls[url_index]})

                if self.vector_store:
                    new_vectorstore = FAISS.from_texts(
                        texts=all_texts,
                        embedding=embeddings,
                        metadatas=metadatas
                    )
                    self.vector_store.merge_from(new_vectorstore)
                else:
                    self.vector_store = FAISS.from_texts(
                        texts=all_texts,
                        embedding=embeddings,
                        metadatas=metadatas
                    )
                
                self.save_faiss_index()
                
                st.session_state.processed_urls.update(new_urls)
                self.save_processed_urls(st.session_state.processed_urls)
                
                llm = ChatOpenAI(
                    temperature=0.3,
                    max_tokens=1000,
                    model_name="gpt-3.5-turbo-16k"
                )
                
                memory = ConversationBufferMemory(
                    memory_key="chat_history",
                    output_key="answer",
                    return_messages=True
                )
                
                st.session_state.qa_chain = ConversationalRetrievalChain.from_llm(
                    llm=llm,
                    retriever=self.vector_store.as_retriever(
                        search_kwargs={"k": 5}
                    ),
                    memory=memory,
                    verbose=True,
                    return_source_documents=True,
                    chain_type="stuff",
                    combine_docs_chain_kwargs={"prompt": self.qa_template},
                    rephrase_question=False,
                    return_generated_question=False
                )
                
                st.success(f"Successfully processed {len(new_urls)} new URLs")
                
        except Exception as e:
            logger.error(f"Error processing URLs: {str(e)}")
            st.error(f"Error processing URLs: {str(e)}")

    def get_scheme_summary(self, url: str) -> Dict:
        """Generate a structured summary of a scheme from its URL."""
        try:
            if not st.session_state.qa_chain:
                st.warning("Please process some URLs first")
                return None

            if url not in st.session_state.processed_urls:
                st.warning("This URL hasn't been processed yet. Please process it first.")
                return None
                
            summary_prompts = {
                "Benefits": """Please provide a comprehensive list of benefits and advantages offered under this scheme. 
                             Include both direct and indirect benefits, monetary and non-monetary advantages. 
                             Format the response as a clear, structured list.""",
                
                "Eligibility": """Who is eligible for this scheme? Please provide:
                                1. All eligibility criteria
                                2. Any age/income/category restrictions
                                3. Special provisions or exceptions
                                Format as clear bullet points.""",
                
                "Application Process": """Describe the complete application process for this scheme:
                                       1. Step-by-step application procedure
                                       2. Where and how to apply
                                       3. Important deadlines or timing requirements
                                       4. Any fees or charges involved""",
                
                "Required Documents": """List all documents required for this scheme:
                                      1. Mandatory documents
                                      2. Additional supporting documents
                                      3. Format/specifications of documents
                                      4. Any special authentication requirements"""
            }
            
            summary = {}
            
            # First, get general context about the scheme
            initial_context = st.session_state.qa_chain(
                {
                    "question": f"What is the main purpose and overview of the scheme at {url}?",
                    "chat_history": []
                }
            )
            
            if not initial_context.get("answer"):
                st.error("Could not retrieve scheme information")
                return None
            
            for aspect, prompt in summary_prompts.items():
                with st.spinner(f"Analyzing {aspect}..."):
                    # Include initial context in the prompt
                    context_prompt = f"""Given this scheme information: {initial_context['answer']}\n\n
                    {prompt}\n\n
                    If specific details are not available in the source material, clearly indicate what information is not provided."""
                    
                    try:
                        result = st.session_state.qa_chain({
                            "question": context_prompt,
                            "chat_history": []
                        })
                        
                        # Handle both possible return formats
                        answer = result.get("answer", result.get("response", "No answer available"))
                        
                        # Only try backup prompt if the first answer is inadequate
                        if any(phrase in answer.lower() for phrase in ["i don't know", "not specified", "no information", "cannot find"]):
                            backup_prompt = f"What specific information can you find about the {aspect.lower()} of this scheme? Please provide any relevant details."
                            backup_result = st.session_state.qa_chain(
                                {
                                    "question": backup_prompt,
                                    "chat_history": []
                                }
                            )
                            if backup_result["answer"] and len(backup_result["answer"]) > len(answer):
                                answer = backup_result["answer"]
                        
                        summary[aspect] = answer
                    except Exception as e:
                        logger.warning(f"Error processing {aspect}: {str(e)}")
                        summary[aspect] = f"Could not retrieve information about {aspect}"
                        continue
                
            if not any(summary.values()):
                st.error("Could not generate meaningful summary from the available content")
                return None
                
            return summary
            
        except Exception as e:
            logger.error(f"Error generating summary: {str(e)}")
            st.error(f"Failed to generate summary: {str(e)}")
            return None

    def render_sidebar(self) -> List[str]:
        """Render the sidebar UI and collect URLs."""
        urls = []
        
        with st.sidebar:
            st.header("Add Scheme URLs")
            
            # Direct URL input
            direct_url = st.text_input(
                "Enter URL:",
                help="Enter the URL of a government scheme page"
            )
            if direct_url:
                if self.validate_url(direct_url):
                    urls.append(direct_url)
                else:
                    st.error("Please enter a valid URL starting with http:// or https://")
            
            # File upload
            uploaded_file = st.file_uploader(
                "Upload file containing URLs (one per line)",
                type=['txt'],
                help="Upload a text file with one URL per line"
            )
            if uploaded_file:
                try:
                    file_urls = [
                        line.decode("utf-8").strip() 
                        for line in uploaded_file 
                        if line.decode("utf-8").strip()
                    ]
                    valid_urls = [url for url in file_urls if self.validate_url(url)]
                    urls.extend(valid_urls)
                    
                    if len(valid_urls) < len(file_urls):
                        st.warning(f"Found {len(file_urls) - len(valid_urls)} invalid URLs in file")
                except Exception as e:
                    st.error(f"Error reading file: {str(e)}")
            
            # Process button
            if urls and st.button("Process URLs", help="Click to process the entered URLs"):
                return urls
                
        return []

    def render_qa_interface(self):
        """Render the question-answering interface."""
        st.header("Ask Questions About Schemes")
        
        if not st.session_state.qa_chain:
            st.info("Please process some URLs first to enable question answering")
            return
            
        user_question = st.text_input(
            "Enter your question:",
            help="Ask any question about the processed schemes"
        )
        
        if user_question:
            try:
                with st.spinner("Searching for answer..."):
                    result = st.session_state.qa_chain(
                        {"question": user_question, "chat_history": st.session_state.chat_history}
                    )
                    
                    st.write("Answer:", result["answer"])
                    
                    # Display sources
                    st.subheader("Sources:")
                    for doc in result["source_documents"]:
                        st.write(f"- {doc.metadata.get('source', 'Unknown source')}")
                    
                    # Update chat history
                    st.session_state.chat_history.append((user_question, result["answer"]))
            except Exception as e:
                logger.error(f"Error processing question: {str(e)}")
                st.error("Failed to process your question. Please try again.")

    def main(self):
        """Main application loop."""
        try:
            if not self.api_key:
                st.warning("Please provide an OpenAI API key in the sidebar to continue.")
                return

            # Render sidebar and get URLs
            urls = self.render_sidebar()
            if urls:
                self.process_urls(urls)
            
            # Display processed URLs
            if st.session_state.processed_urls:
                st.header("Processed Schemes")
                for url in st.session_state.processed_urls:
                    with st.expander(f"Scheme: {url}"):
                        st.write(f"URL: {url}")
                        if st.button(f"Generate Summary for {url}", key=f"btn_{url}"):
                            summary = self.get_scheme_summary(url)
                            if summary:
                                for aspect, content in summary.items():
                                    st.subheader(aspect)
                                    st.write(content)
            
            # Render QA interface
            self.render_qa_interface()

        except Exception as e:
            logger.error(f"Main loop error: {str(e)}")
            st.error("An error occurred. Please refresh the page and try again.")

if __name__ == "__main__":
    try:
        app = SchemeResearchTool()
        app.main()
    except Exception as e:
        st.error(f"Application failed to start: {str(e)}")
        logger.critical(f"Application failed to start: {str(e)}")