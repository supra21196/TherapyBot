"""
Essential utility functions for the smart RAG Agent system.
"""

import logging
import sys
import tempfile
import os
from typing import List, Tuple
from datetime import datetime

def setup_logging():
    """Setup simple logging configuration."""
    logger = logging.getLogger("rag_agent")
    
    if logger.handlers:
        return logger
    
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(levelname)s - %(message)s"))
    logger.addHandler(handler)
    
    # Reduce noise from ML libraries
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
    
    return logger

def calculate_confidence(similarity_scores: List[float], query_length: int, source_type: str = "internal") -> float:
    """
    Calculate confidence score from similarity scores, query length, and source type.
    
    Args:
        similarity_scores: List of similarity scores from search
        query_length: Length of the user query
        source_type: Source of the response ("internal", "external_api", etc.)
    """
    if not similarity_scores:
        return 0.0
    
    confidence = max(similarity_scores)
    
    # Boost for multiple good matches
    if len([s for s in similarity_scores if s > 0.3]) > 1:
        confidence += 0.1
    
    # Boost for longer queries
    if query_length > 20:
        confidence += 0.05
    
    # Adjust confidence based on source type
    if source_type == "external_api":
        confidence = min(confidence + 0.2, 1.0)  # External APIs typically more factual
    elif source_type == "no_results":
        confidence = 0.0
    
    return max(0.0, min(1.0, confidence))

def validate_query(query: str) -> Tuple[bool, str]:
    """Validate user query for safety and completeness."""
    if not query or len(query.strip()) < 3:
        return False, "Please provide more detail about what you're experiencing."
    
    # Check for crisis indicators
    crisis_terms = ['suicide method', 'how to kill', 'ways to die', 'overdose amount']
    if any(term in query.lower() for term in crisis_terms):
        return False, (
            "I'm concerned about your safety. Please reach out immediately:\n"
            "• Crisis Lifeline: 988\n"
            "• Emergency: 911"
        )
    
    return True, ""

def determine_query_urgency(query: str) -> str:
    """
    Determine urgency level for smart routing decisions.
    
    Args:
        query: User query string
        
    Returns:
        Urgency level: "emergency", "urgent", "moderate", "low"
    """
    query_lower = query.lower()
    
    # Emergency indicators
    if any(term in query_lower for term in ['crisis', 'suicide', 'emergency', 'harm myself', 'kill myself']):
        return "emergency"
    
    # Urgent indicators  
    if any(term in query_lower for term in ['panic attack', 'can\'t breathe', 'right now', 'immediately']):
        return "urgent"
    
    # Moderate indicators
    if any(term in query_lower for term in ['help me', 'struggling', 'can\'t sleep', 'feel terrible']):
        return "moderate"
    
    return "low"