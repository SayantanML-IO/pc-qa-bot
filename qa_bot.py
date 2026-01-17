"""
Improved QA bot with robust error handling and type hints.
"""
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel, RunnableLambda
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_core.documents import Document
import logging
import os
import time
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from validators import InputValidator, ValidationError
from device_manager import get_device_kwargs
from metrics import MetricsTracker

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


class QABotError(Exception):
    """Custom exception for QA bot errors."""
    pass


def create_compression_retriever(base_compressor, base_retriever):
    """
    Create a compression retriever as a runnable lambda.
    
    Args:
        base_compressor: Compressor/reranker object
        base_retriever: Base retriever object
        
    Returns:
        RunnableLambda for retrieval and compression
    """
    def retrieve_and_compress(query: str) -> List[Any]:
        """Get documents from base retriever and compress them."""
        docs = base_retriever.invoke(query)
        compressed_docs = base_compressor.compress_documents(docs, query)
        return compressed_docs
    
    return RunnableLambda(retrieve_and_compress)


class CrossEncoderReranker:
    """Cross-encoder reranker for document compression."""
    
    def __init__(self, model: HuggingFaceCrossEncoder, top_n: int = 3):
        """
        Initialize reranker.
        
        Args:
            model: HuggingFace cross-encoder model
            top_n: Number of top documents to return
        """
        self.model = model
        self.top_n = top_n
    
    def compress_documents(
        self, 
        documents: List[Document], 
        query: str
    ) -> List[Document]:
        """
        Rerank documents using cross-encoder model.
        
        Args:
            documents: List of documents to rerank
            query: Query string
            
        Returns:
            Top N reranked documents
        """
        if not documents:
            return []
        
        try:
            # Score each document
            pairs = [[query, doc.page_content] for doc in documents]
            scores = self.model.score(pairs)
            
            # Sort by score and return top N
            doc_scores = list(zip(documents, scores))
            doc_scores.sort(key=lambda x: x[1], reverse=True)
            
            return [doc for doc, score in doc_scores[:self.top_n]]
        
        except Exception as e:
            logger.error(f"Reranking failed: {e}, returning original documents")
            return documents[:self.top_n]


class QABot:
    """Question-answering bot with RAG architecture."""
    
    def __init__(
        self, 
        vector_store, 
        config: Dict,
        metrics_tracker: Optional[MetricsTracker] = None
    ):
        """
        Initialize Q&A bot with vector store and LLM.
        
        Args:
            vector_store: VectorStore object
            config: Configuration dictionary
            metrics_tracker: Optional metrics tracker
            
        Raises:
            QABotError: If initialization fails
        """
        self.vector_store = vector_store
        self.config = config
        self.metrics = metrics_tracker
        
        try:
            self.llm = self._initialize_llm()
            self.retriever = self._initialize_retriever()
            self.qa_chain = self._create_qa_chain()
            logger.info("✓ QA Bot initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize QA bot: {e}")
            raise QABotError(f"QA bot initialization failed: {e}")
    
    def _initialize_llm(self):
        """
        Initialize the LLM based on config.
        
        Returns:
            Initialized LLM object
            
        Raises:
            QABotError: If LLM initialization fails
        """
        provider = self.config['llm']['provider']
        
        if provider == "groq":
            logger.info("Initializing Groq LLM...")
            
            # Get API key from environment variable
            api_key = os.getenv('GROQ_API_KEY')
            if not api_key:
                raise QABotError(
                    "GROQ_API_KEY not found in environment variables. "
                    "Please set it in your .env file."
                )
            
            try:
                from langchain_groq import ChatGroq
                llm = ChatGroq(
                    model=self.config['llm']['groq']['model'],
                    groq_api_key=api_key,
                    temperature=self.config['qa']['temperature']
                )
                logger.info(f"✓ Groq LLM initialized: {self.config['llm']['groq']['model']}")
                return llm
            
            except ImportError:
                raise QABotError(
                    "langchain-groq not installed. Install with: pip install langchain-groq"
                )
            except Exception as e:
                raise QABotError(f"Groq LLM initialization failed: {e}")
        
        else:
            raise QABotError(f"Unknown LLM provider: {provider}")
    
    def _initialize_retriever(self):
        """
        Initialize retriever with reranking.
        
        Returns:
            Configured retriever
            
        Raises:
            QABotError: If retriever initialization fails
        """
        try:
            # Create base retriever
            base_retriever = self.vector_store.get_retriever(
                k=self.config['qa']['retriever_k']
            )
            
            # Initialize reranker model
            logger.info("Initializing reranker model (bge-reranker-base)...")
            device_kwargs = get_device_kwargs()
            
            reranker_model = HuggingFaceCrossEncoder(
                model_name='BAAI/bge-reranker-base',
                model_kwargs=device_kwargs
            )
            
            # Create compressor with reranker
            compressor = CrossEncoderReranker(
                model=reranker_model,
                top_n=self.config['qa']['reranker_top_n']
            )
            
            # Create final retriever with reranking
            retriever = create_compression_retriever(
                base_compressor=compressor,
                base_retriever=base_retriever
            )
            
            logger.info(
                f"✓ Retriever initialized: retrieve {self.config['qa']['retriever_k']} "
                f"chunks, rerank to top {self.config['qa']['reranker_top_n']}"
            )
            
            return retriever
        
        except Exception as e:
            logger.error(f"Retriever initialization failed: {e}")
            raise QABotError(f"Failed to initialize retriever: {e}")
    
    def _create_qa_chain(self):
        """
        Create the question-answering chain.
        
        Returns:
            Configured QA chain
        """
        # Updated prompt template for PC hardware news
        prompt_template = """You are a helpful assistant that answers questions about PC hardware news, product releases, reviews, and industry updates.

Use the following pieces of context from recent hardware articles to answer the question at the end. If you don't know the answer based on the context, just say "I don't have enough information in the recent articles to answer that question."

Be specific and cite which news source or article the information comes from when possible. Include relevant details like product names, specifications, prices, and release dates when available.

Context:
{context}

Question: {question}

Answer:"""

        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)
        
        # Answer generation chain
        answer_generation_chain = (
            RunnablePassthrough.assign(
                context=(lambda x: format_docs(x["docs"]))
            )
            | prompt
            | self.llm
            | StrOutputParser()
        )

        # Source documents chain
        source_documents_chain = lambda x: x["docs"]

        # Combined parallel chain
        combined_chain = (
            RunnableParallel(
                docs=self.retriever,
                question=RunnablePassthrough()
            )
            .assign(
                answer=answer_generation_chain,
                source_docs=source_documents_chain
            )
        )
        
        logger.info("✓ Q&A chain created successfully")
        return combined_chain
    
    def ask(self, question: str) -> Dict[str, Any]:
        """
        Ask a question and get an answer.
        
        Args:
            question: User's question string
            
        Returns:
            Dict with 'answer' and 'sources'
            
        Raises:
            QABotError: If question processing fails
        """
        # Validate and sanitize question
        try:
            question = InputValidator.validate_question(question)
        except ValidationError as e:
            logger.error(f"Invalid question: {e}")
            return {
                'answer': f"Invalid question: {e}",
                'sources': []
            }
        
        logger.info(f"Processing question: {question[:100]}...")
        start_time = time.time()
        
        try:
            # Invoke the combined chain
            result = self.qa_chain.invoke(question)
            
            answer = result['answer']
            source_docs = result['source_docs']
            
            # Format sources
            formatted_sources = []
            for i, doc in enumerate(source_docs, 1):
                formatted_sources.append({
                    'chunk_id': i,
                    'content': doc.page_content[:200] + "...",
                    'source': doc.metadata.get('source', 'Unknown'),
                    'title': doc.metadata.get('title', 'N/A'),
                    'url': doc.metadata.get('url', 'N/A'),
                    'date': doc.metadata.get('date', 'N/A')
                })
            
            response_time = time.time() - start_time
            logger.info(f"✓ Answer generated in {response_time:.2f}s")
            
            # Log metrics
            if self.metrics:
                self.metrics.log_query(
                    question=question,
                    response_time=response_time,
                    chunks_retrieved=len(source_docs),
                    success=True
                )
            
            return {
                'answer': answer,
                'sources': formatted_sources,
                'response_time': response_time
            }
        
        except Exception as e:
            error_msg = f"Error generating answer: {e}"
            logger.error(error_msg)
            
            # Log failed query
            if self.metrics:
                self.metrics.log_query(
                    question=question,
                    response_time=time.time() - start_time,
                    chunks_retrieved=0,
                    success=False,
                    error_message=str(e)
                )
            
            return {
                'answer': error_msg,
                'sources': [],
                'response_time': time.time() - start_time
            }
    
    def ask_simple(self, question: str) -> str:
        """
        Simple version that just returns the answer text.
        
        Args:
            question: User's question
            
        Returns:
            Answer text
        """
        result = self.ask(question)
        return result['answer']