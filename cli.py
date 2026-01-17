"""
CLI with metrics dashboard and UX.
"""
import yaml
import logging
import os
import sys
from typing import Dict, Optional
from pathlib import Path

from vector_store import VectorStore, VectorStoreError
from qa_bot import QABot, QABotError
from validators import InputValidator, ValidationError
from metrics import MetricsTracker
from device_manager import DeviceManager
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_config(config_path: str = 'config.yaml') -> Optional[Dict]:
    """Load configuration from YAML file."""
    try:
        with open(config_path, 'r', encoding='utf-8') as file:
            config = yaml.safe_load(file)
            InputValidator.validate_config(config)
            return config
    except FileNotFoundError:
        print(f"\n ERROR: Config file not found: {config_path}")
        return None
    except ValidationError as e:
        print(f"\n ERROR: Invalid configuration: {e}")
        return None
    except Exception as e:
        print(f"\n ERROR: Failed to load config: {e}")
        return None


def validate_environment() -> bool:
    """Validate that required environment variables are set."""
    if not os.getenv('GROQ_API_KEY'):
        print("\n ERROR: GROQ_API_KEY environment variable not set!")
        print("\nPlease follow these steps:")
        print("1. Copy .env.example to .env")
        print("2. Edit .env and add your Groq API key")
        print("3. Get a free API key from: https://console.groq.com")
        return False
    return True


def display_metrics(metrics: MetricsTracker) -> None:
    """Display system metrics dashboard."""
    try:
        stats = metrics.get_recent_stats(days=7)
        failure_rate = metrics.get_failure_rate(hours=24)
        
        print("\n" + "="*70)
        print(" SYSTEM METRICS (Last 7 Days)")
        print("="*70)
        
        # Indexing stats
        indexing = stats.get('indexing', {})
        print(f"\n Indexing:")
        print(f"   Runs:            {indexing.get('runs', 0)}")
        print(f"   Articles Indexed: {indexing.get('total_indexed', 0)}")
        print(f"   Avg Duration:     {indexing.get('avg_duration', 0):.1f}s")
        
        # Query stats
        queries = stats.get('queries', {})
        print(f"\n Queries:")
        print(f"   Total:           {queries.get('total', 0)}")
        print(f"   Avg Response:    {queries.get('avg_response_time', 0):.3f}s")
        print(f"   Success Rate:    {queries.get('success_rate', 0):.1f}%")
        
        # System health
        system = stats.get('system', {})
        print(f"\n System:")
        print(f"   Vector Store:    {system.get('vector_store_size', 0):,} chunks")
        print(f"   Total Articles:  {system.get('total_articles', 0)}")
        print(f"   Device:          {system.get('device', 'unknown').upper()}")
        
        # Recent failures
        print(f"\n Recent Failures (24h):")
        print(f"   Total Attempts:  {failure_rate.get('total_attempts', 0)}")
        print(f"   Failures:        {failure_rate.get('failures', 0)}")
        print(f"   Failure Rate:    {failure_rate.get('failure_rate', 0):.1f}%")
        
        print("="*70)
    
    except Exception as e:
        logger.error(f"Failed to display metrics: {e}")
        print(f"\n  Could not load metrics: {e}")


def get_topic_summary(bot: QABot) -> str:
    """Get a summary of hardware topics available in the documents."""
    logger.info("Generating topic summary...")
    
    question = """List all the PC hardware topics covered in these articles. Group them by category like:
- GPU releases (specific models mentioned)
- CPU releases (specific models mentioned)  
- Motherboard news
- Storage news
- Other hardware

Also mention which sources these come from (like Tom's Hardware, TechPowerUp, etc.). Be specific about product names and models."""
    
    try:
        result = bot.ask(question)
        return result['answer'].strip()
    except Exception as e:
        logger.error(f"Could not generate summary: {e}")
        return "Unable to generate topic summary."


def print_answer(result: Dict) -> None:
    """Pretty print the answer and sources."""
    print("\n" + "="*70)
    print(" ANSWER:")
    print("="*70)
    print(result['answer'])
    
    if result.get('sources'):
        print("\n" + "="*70)
        print(" SOURCES:")
        print("="*70)
        for source in result['sources']:
            print(f"\n[Source {source['chunk_id']}]")
            print(f"  Title:   {source['title']}")
            print(f"  From:    {source['source']}")
            print(f"  Date:    {source['date']}")
            print(f"  URL:     {source['url']}")
            print(f"  Preview: {source['content']}")
    
    # Show response time
    if 'response_time' in result:
        print(f"\n⏱  Response time: {result['response_time']:.2f}s")
    
    print("\n" + "="*70)


def interactive_mode(bot: QABot, metrics: MetricsTracker) -> None:
    """Run interactive Q&A session."""
    print("\n" + "="*70)
    print(" PC HARDWARE Q&A BOT - INTERACTIVE MODE")
    print("="*70)
    
    # Display metrics first
    display_metrics(metrics)
    
    # Get and display topic summary
    print("\n Generating available topics from indexed articles...")
    summary = get_topic_summary(bot)
    
    print("\n" + "="*70)
    print(" AVAILABLE HARDWARE TOPICS:")
    print("="*70)
    print(summary)
    print("="*70)
    
    print("\n Ask questions about PC hardware from the topics above.")
    print("Commands:")
    print("  - Type your question")
    print("  - 'metrics' to show system metrics")
    print("  - 'quit', 'exit', or 'q' to stop")
    print()
    
    while True:
        try:
            question = input("\n Your question: ").strip()
            
            if question.lower() in ['quit', 'exit', 'q']:
                print("\n Goodbye!")
                break
            
            if question.lower() == 'metrics':
                display_metrics(metrics)
                continue
            
            if not question:
                continue
            
            # Validate question
            try:
                InputValidator.validate_question(question)
            except ValidationError as e:
                print(f"\n  Invalid question: {e}")
                continue
            
            # Get answer
            print("\n Searching knowledge base...")
            result = bot.ask(question)
            print_answer(result)
        
        except KeyboardInterrupt:
            print("\n\n Goodbye!")
            break
        except Exception as e:
            print(f"\n Error: {e}")
            logger.error(f"Error in interactive mode: {e}")


def single_question_mode(bot: QABot, question: str) -> None:
    """Answer a single question and exit."""
    print(f"\n Processing question: {question}")
    
    try:
        result = bot.ask(question)
        print_answer(result)
    except Exception as e:
        print(f"\n Error processing question: {e}")
        logger.error(f"Error in single question mode: {e}")


def show_help() -> None:
    """Display help information."""
    help_text = """
╔══════════════════════════════════════════════════════════════════════╗
║                  PC HARDWARE Q&A BOT - HELP                          ║
╚══════════════════════════════════════════════════════════════════════╝

USAGE:
  python cli.py                  # Interactive mode
  python cli.py [question]       # Single question mode
  python cli.py --help           # Show this help
  python cli.py --metrics        # Show metrics only

INTERACTIVE MODE COMMANDS:
  - Type any question about PC hardware
  - 'metrics'     Show system metrics dashboard
  - 'quit'        Exit the program
  - 'exit'        Exit the program
  - 'q'           Exit the program

EXAMPLES:
  python cli.py
  python cli.py "What are the latest RTX graphics cards?"
  python cli.py "Tell me about new AMD processors"
  python cli.py --metrics

FEATURES:
  ✓ Semantic search across hardware news articles
  ✓ Real-time metrics and performance tracking
  ✓ Hardware relevance filtering
  ✓ Multi-source article aggregation
  ✓ GPU acceleration (CUDA) with CPU fallback

For more information, check the project documentation.
"""
    print(help_text)


def main() -> None:
    """Main CLI entry point."""
    # Check for help flag
    if '--help' in sys.argv or '-h' in sys.argv:
        show_help()
        return
    
    # Validate environment
    if not validate_environment():
        return
    
    # Initialize device manager
    device_manager = DeviceManager()
    
    # Load config
    config = load_config()
    if not config:
        print("\n Failed to load configuration. Exiting.")
        return
    
    # Initialize metrics
    metrics = MetricsTracker()
    
    # Check for metrics-only flag
    if '--metrics' in sys.argv:
        display_metrics(metrics)
        return
    
    # Initialize vector store
    print("\n Loading vector store...")
    try:
        vector_store = VectorStore(
            persist_directory=config['vector_db']['persist_directory'],
            collection_name=config['vector_db']['collection_name']
        )
    except VectorStoreError as e:
        print(f"\n Failed to initialize vector store: {e}")
        return
    
    # Check if vector store has documents
    doc_count = vector_store.count_documents()
    if doc_count == 0:
        print("\n No documents found in vector store!")
        print("Please run 'python main_improved.py' first to index articles.")
        return
    
    print(f"✓ Vector store loaded: {doc_count:,} document chunks available.")
    
    # Initialize Q&A bot
    print("\n Initializing Q&A bot...")
    try:
        bot = QABot(vector_store, config, metrics_tracker=metrics)
        print("✓ Bot ready!")
    except QABotError as e:
        print(f"\n Failed to initialize bot: {e}")
        logger.error(f"Bot initialization error: {e}")
        return
    
    # Log system health
    metrics.log_system_health(
        vector_store_size=doc_count,
        total_articles=doc_count // 5,  # Rough estimate
        device=device_manager.device
    )
    
    # Check command line arguments
    # Filter out flags
    args = [arg for arg in sys.argv[1:] if not arg.startswith('--')]
    
    if args:
        # Single question mode
        question = ' '.join(args)
        single_question_mode(bot, question)
    else:
        # Interactive mode
        interactive_mode(bot, metrics)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        print(f"\n Fatal error: {e}")
        exit(1)