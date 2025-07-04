"""
RAG Agent: Personal Mental Health Assistant
A smart RAG system that routes between internal knowledge and external data sources.
"""

__version__ = "0.1.0"
__author__ = "RAG Agent Team"

# Import main classes for clean external usage
from .agent import MentalHealthRAG

# Import supporting classes if needed directly
from .knowledge_base import KnowledgeBase, TherapeuticContent
from .database import Database, QueryLog
from .utils import setup_logging, calculate_confidence, determine_query_urgency

# Public API - what users can import from rag_agent
__all__ = [
    "MentalHealthRAG",
    "KnowledgeBase", 
    "TherapeuticContent",
    "Database",
    "QueryLog",
    "setup_logging",
    "calculate_confidence",
    "determine_query_urgency"
]

# Initialize logging when package is imported
setup_logging()