import os
from typing import List, Optional
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import WebBaseLoader
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document
import chromadb
from chromadb.config import Settings
import bs4
import logging
import hashlib

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGChainBuilder:
    def __init__(self, persist_directory: str = "./chroma_db"):
        self.llm = None
        self.embeddings = None
        self.vectorstore = None
        self.rag_chain = None
        self.persist_directory = persist_directory
        self._init_environment()

    def _init_environment(self) -> None:
        """Initialize environment variables and API keys."""
        load_dotenv()
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")

    def _get_collection_name(self, url: str) -> str:
        """Generate a consistent collection name from URL."""
        return hashlib.md5(url.encode()).hexdigest()

    def initialize_llm(self) -> None:
        """Initialize the language model."""
        try:
            self.llm = ChatOpenAI(
                model="gpt-4o-2024-11-20",
                openai_api_key=self.api_key,
                temperature=0.1
            )
            logger.info("LLM initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {e}")
            raise

    def load_web_content(self, url: str) -> List[Document]:
        """Load and parse web content."""
        try:
            loader = WebBaseLoader(
                web_paths=(url,),
                bs_kwargs={
                    'parse_only': bs4.SoupStrainer(
                        ["p", "h1", "h2", "h3", "h4", "h5", "h6"]
                    )
                }
            )
            docs = loader.load()
            if not docs:
                raise ValueError(f"No content loaded from {url}")
            logger.info(f"Successfully loaded {len(docs)} documents from {url}")
            return docs
        except Exception as e:
            logger.error(f"Error loading web content: {e}")
            raise

    def split_documents(self, docs: List[Document]) -> List[Document]:
        """Split documents into smaller chunks."""
        try:
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=200,
                length_function=len,
                is_separator_regex=False
            )
            splits = text_splitter.split_documents(docs)
            if not splits:
                raise ValueError("Document splitting produced no chunks")
            logger.info(f"Created {len(splits)} text chunks")
            return splits
        except Exception as e:
            logger.error(f"Error splitting documents: {e}")
            raise

    def setup_vectorstore(self, splits: List[Document], url: str) -> None:
        """Set up the persistent vector store with document chunks."""
        try:
            # Initialize embeddings
            self.embeddings = OpenAIEmbeddings(openai_api_key=self.api_key)
            
            # Generate collection name from URL
            collection_name = self._get_collection_name(url)
            
            # Initialize Chroma with persistence
            self.vectorstore = Chroma.from_documents(
                documents=splits,
                embedding=self.embeddings,
                persist_directory=self.persist_directory,
                collection_name=collection_name
            )
            
            logger.info(f"Vector store created/updated successfully with collection: {collection_name}")
            
        except Exception as e:
            logger.error(f"Error setting up vector store: {e}")
            raise

    def setup_rag_chain(self) -> None:
        """Set up the RAG chain for question answering."""
        try:
            prompt = hub.pull("rlm/rag-prompt")
            retriever = self.vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 3}
            )

            def format_docs(docs):
                return "\n\n".join(doc.page_content for doc in docs)

            self.rag_chain = (
                {"context": retriever | format_docs, "question": RunnablePassthrough()}
                | prompt
                | self.llm
                | StrOutputParser()
            )
            logger.info("RAG chain setup completed")
        except Exception as e:
            logger.error(f"Error setting up RAG chain: {e}")
            raise

    def query(self, question: str) -> str:
        """Execute a query using the RAG chain."""
        try:
            if not self.rag_chain:
                raise ValueError("RAG chain not initialized")
            return self.rag_chain.invoke(question)
        except Exception as e:
            logger.error(f"Error during query execution: {e}")
            raise

def main():
    try:
        # Initialize RAG chain builder with persistent directory
        rag_builder = RAGChainBuilder(persist_directory="./chroma_db")
        rag_builder.initialize_llm()

        # Load and process content
        url = "https://www.presidency.ucsb.edu/documents/presidential-debate-atlanta-georgia"
        docs = rag_builder.load_web_content(url)
        splits = rag_builder.split_documents(docs)
        
        # Set up vector store and RAG chain
        rag_builder.setup_vectorstore(splits, url)
        rag_builder.setup_rag_chain()

        # Execute sample queries
        questions = [
            "What were the key topics discussed during the debate?",
            "What were the candidates' positions on foreign policy?",
            "How did the candidates address economic issues?"
        ]
        
        for question in questions:
            print(f"\nQuestion: {question}")
            answer = rag_builder.query(question)
            print(f"Answer: {answer}")

    except Exception as e:
        logger.error(f"Application error: {e}")
        raise

if __name__ == "__main__":
    main()