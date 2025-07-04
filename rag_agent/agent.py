"""
Main MentalHealthRAG agent class.
Orchestrates knowledge base search, external data retrieval, and response generation.
"""

import asyncio
import logging
from typing import Dict, Any, Optional, Tuple
from datetime import datetime
import aiohttp

from .knowledge_base import KnowledgeBase
from .database import Database
from .utils import calculate_confidence, validate_query

class MentalHealthRAG:
    """
    Intelligent mental health RAG assistant that routes queries between internal knowledge
    and external data sources based on query type and content.
    """
    
    def __init__(self):
        """Initialize the mental health RAG system."""
        self.logger = logging.getLogger("rag_agent")
        
        # Initialize components
        self.knowledge_base = KnowledgeBase()
        self.database = Database()
        
        # Routing configuration
        self.min_similarity = 0.2
        self.max_results = 3
        self.external_api_timeout = 10
        
        self.logger.info("Smart MentalHealthRAG system initialized")
    
    async def query(self, question: str) -> str:
        """
        Intelligently process a user query by routing to appropriate data sources.
        
        Args:
            question: User's question or concern
            
        Returns:
            Formatted response from best available source
        """
        start_time = datetime.now()
        
        try:
            # Validate the query for safety
            is_valid, error_message = validate_query(question)
            if not is_valid:
                await self._log_query(question, error_message, 0.0, 0.0, "validation_error")
                return error_message
            
            # Intelligent routing decision
            query_type, needs_external = self._analyze_query_type(question)
            
            if needs_external:
                # Try external sources first for real-time/factual queries
                external_response = await self._get_external_data(question, query_type)
                if external_response:
                    processing_time = self._get_elapsed_time(start_time)
                    await self._log_query(question, external_response, 0.8, processing_time, "external_api")
                    return external_response
                
                # Fall back to internal knowledge if external fails
                self.logger.info("External API failed, falling back to internal knowledge")
            
            # Search internal knowledge base
            search_results = await self.knowledge_base.search(question, limit=self.max_results)
            
            if not search_results:
                response = self._handle_no_results(question, query_type)
                await self._log_query(question, response, 0.0, self._get_elapsed_time(start_time), "no_results")
                return response
            
            # Generate response from internal knowledge
            response = self._generate_response(search_results, question, query_type)
            
            # Calculate confidence
            similarities = [result['similarity'] for result in search_results]
            confidence = calculate_confidence(similarities, len(question))
            
            # Log the interaction
            processing_time = self._get_elapsed_time(start_time)
            await self._log_query(question, response, confidence, processing_time, "internal_knowledge")
            
            return response
            
        except Exception as e:
            self.logger.error(f"Error processing query: {e}")
            error_response = "I'm experiencing some technical difficulties. Please try again in a moment."
            await self._log_query(question, error_response, 0.0, self._get_elapsed_time(start_time), "system_error")
            return error_response
    
    def _analyze_query_type(self, question: str) -> Tuple[str, bool]:
        """
        Analyze query to determine type and whether external data is needed.
        
        Args:
            question: User question
            
        Returns:
            Tuple of (query_type, needs_external_data)
        """
        question_lower = question.lower()
        
        # Crisis/emergency - always use internal knowledge for safety
        if any(term in question_lower for term in ['crisis', 'suicide', 'emergency', 'harm myself']):
            return "crisis", False
        
        # Current events or real-time mental health news
        if any(term in question_lower for term in ['current', 'latest', 'recent', 'news', 'today', 'this week']):
            if any(term in question_lower for term in ['research', 'study', 'therapy', 'treatment', 'mental health']):
                return "current_research", True
        
        # Factual questions about mental health conditions
        if any(term in question_lower for term in ['what is', 'define', 'statistics', 'prevalence', 'facts about']):
            if any(term in question_lower for term in ['depression', 'anxiety', 'ptsd', 'bipolar', 'adhd', 'ocd']):
                return "factual_condition", True
        
        # Medication or professional treatment info
        if any(term in question_lower for term in ['medication', 'drug', 'prescription', 'side effects', 'dosage']):
            return "medical_info", True
        
        # Local resources or services
        if any(term in question_lower for term in ['near me', 'in my area', 'local', 'therapist near', 'clinic']):
            return "local_resources", True
        
        # Personal coping and therapeutic techniques - use internal knowledge
        if any(term in question_lower for term in ['help me', 'coping', 'technique', 'strategy', 'feel better']):
            return "coping_strategy", False
        
        # Default to internal knowledge for personal support
        return "personal_support", False
    
    async def _get_external_data(self, question: str, query_type: str) -> Optional[str]:
        """
        Retrieve data from external sources based on query type.
        
        Args:
            question: User question
            query_type: Type of query requiring external data
            
        Returns:
            External data response or None if unavailable
        """
        try:
            if query_type == "current_research":
                return await self._get_mental_health_research(question)
            elif query_type == "factual_condition":
                return await self._get_condition_facts(question)
            elif query_type == "medical_info":
                return await self._get_medical_information(question)
            elif query_type == "local_resources":
                return await self._get_local_resources(question)
            
            return None
            
        except Exception as e:
            self.logger.error(f"External API error: {e}")
            return None
    
    async def _get_mental_health_research(self, question: str) -> Optional[str]:
        """Get current mental health research (placeholder for real API)."""
        # In production, this would call PubMed API, Google Scholar API, etc.
        
        # Mock response for demonstration
        if "depression" in question.lower():
            return (
                "Recent research shows that combination therapy (medication + psychotherapy) "
                "is most effective for treating depression. A 2024 study found that cognitive "
                "behavioral therapy with mindfulness components showed 70% improvement rates.\n\n"
                "However, for immediate personal support, I recommend using proven coping strategies. "
                "Would you like me to share some techniques that can help right now?"
            )
        
        return None
    
    async def _get_condition_facts(self, question: str) -> Optional[str]:
        """Get factual information about mental health conditions."""
        # Mock response - in production would call medical APIs
        
        if "anxiety" in question.lower():
            return (
                "Anxiety disorders affect about 40 million adults in the US (18.1% of the population). "
                "They're highly treatable, yet only 36.9% of those suffering receive treatment.\n\n"
                "Common symptoms include: excessive worry, restlessness, fatigue, difficulty concentrating, "
                "and physical symptoms like rapid heartbeat.\n\n"
                "If you're experiencing anxiety symptoms, I can share some immediate coping techniques. "
                "Would that be helpful?"
            )
        
        return None
    
    async def _get_medical_information(self, question: str) -> Optional[str]:
        """Get medical/medication information (with safety disclaimers)."""
        return (
            "I can't provide specific medical or medication advice. This type of information "
            "should come from a qualified healthcare provider who knows your medical history.\n\n"
            "For medication questions, please consult:\n"
            "• Your prescribing doctor\n"
            "• A pharmacist\n"
            "• Your healthcare team\n\n"
            "I can help with general coping strategies and emotional support techniques. "
            "Would you like me to share some of those instead?"
        )
    
    async def _get_local_resources(self, question: str) -> Optional[str]:
        """Get local mental health resources."""
        # In production, would use location APIs + mental health directories
        
        return (
            "I don't have access to location-specific data, but here are ways to find local resources:\n\n"
            "• Psychology Today therapist finder: psychologytoday.com\n"
            "• SAMHSA treatment locator: findtreatment.gov\n"
            "• Your insurance provider's website\n"
            "• Call 211 for local community resources\n"
            "• Ask your primary care doctor for referrals\n\n"
            "For immediate support, I can share coping strategies that work anywhere. "
            "Would that be helpful while you search for local resources?"
        )
    
    def _generate_response(self, search_results: list, question: str, query_type: str) -> str:
        """
        Generate a formatted response from internal knowledge search results.
        
        Args:
            search_results: List of matching documents from knowledge base
            question: Original user question
            query_type: Type of query for context
            
        Returns:
            Formatted response string
        """
        if not search_results:
            return self._handle_no_results(question, query_type)
        
        # Use the best matching result as the primary response
        best_result = search_results[0]
        primary_content = best_result['content']
        confidence_score = best_result['similarity']
        
        # Format the response based on confidence and query type
        if query_type == "crisis":
            # Crisis responses get special handling
            response = f"🆘 Here's immediate help:\n\n{primary_content}"
            response += "\n\nIf you're in immediate danger: Call 988 (Crisis Lifeline) or 911"
            
        elif confidence_score > 0.6:
            # High confidence - direct response
            response = f"Here's a technique that can help:\n\n{primary_content}"
            
            # Add secondary information if available and relevant
            if len(search_results) > 1 and search_results[1]['similarity'] > 0.4:
                secondary_content = search_results[1]['content']
                if not self._content_too_similar(primary_content, secondary_content):
                    response += f"\n\nAdditionally:\n{secondary_content[:150]}..."
                    
        elif confidence_score > 0.3:
            # Moderate confidence - qualified response
            response = f"I found some guidance that may help:\n\n{primary_content}"
            response += "\n\nThis may not fully address your specific situation, but it's a starting point."
            
        else:
            # Lower confidence - cautious response
            response = f"I found some related information:\n\n{primary_content}"
            response += "\n\nThis is general guidance and may not fully match your situation. "
            response += "Consider speaking with a mental health professional for personalized support."
        
        # Add source context
        if query_type != "crisis":
            response += "\n\n💙 This comes from my therapeutic knowledge base."
        
        return response
    
    def _handle_no_results(self, question: str, query_type: str) -> str:
        """Handle cases where no relevant knowledge is found."""
        if query_type == "crisis":
            return (
                "🆘 I'm concerned about your safety. Please reach out for immediate help:\n\n"
                "• Crisis Lifeline: 988\n"
                "• Emergency Services: 911\n"
                "• Crisis Text Line: Text HOME to 741741\n\n"
                "You matter, and help is available."
            )
        
        if query_type in ["current_research", "factual_condition", "medical_info"]:
            return (
                "I don't have current external data for this question. For the most up-to-date "
                "information, please consult:\n\n"
                "• A mental health professional\n"
                "• Reputable medical websites (Mayo Clinic, WebMD)\n"
                "• Your healthcare provider\n\n"
                "I can help with coping strategies and emotional support techniques. "
                "Would you like me to share some of those instead?"
            )
        
        return (
            "I don't have specific guidance for your question in my current knowledge base. "
            "Here are some general resources:\n\n"
            "• Consider speaking with a mental health professional\n"
            "• Try mindfulness: Take 5 slow, deep breaths\n"
            "• Reach out to someone you trust\n\n"
            "If you're in distress: Crisis Lifeline 988 is always available."
        )
    
    def _content_too_similar(self, content1: str, content2: str) -> bool:
        """Check if two pieces of content are too similar to include both."""
        words1 = set(content1.lower().split())
        words2 = set(content2.lower().split())
        
        if len(words1) == 0 or len(words2) == 0:
            return True
        
        overlap = len(words1.intersection(words2))
        similarity = overlap / min(len(words1), len(words2))
        
        return similarity > 0.7
    
    async def add_knowledge(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Add new therapeutic knowledge to the internal knowledge base."""
        try:
            success = await self.knowledge_base.add_document(content, metadata or {})
            if success:
                self.logger.info("Added new knowledge to internal knowledge base")
            return success
        except Exception as e:
            self.logger.error(f"Error adding knowledge: {e}")
            return False
    
    async def add_feedback(self, question: str, rating: float) -> bool:
        """Add user feedback for a previous query."""
        try:
            success = await self.database.add_feedback(question, rating)
            if success:
                self.logger.info(f"Recorded feedback: {rating}/5.0")
            return success
        except Exception as e:
            self.logger.error(f"Error recording feedback: {e}")
            return False
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get system usage statistics."""
        try:
            db_stats = await self.database.get_stats()
            kb_stats = self.knowledge_base.get_stats()
            
            return {
                'total_queries': db_stats.get('total_queries', 0),
                'avg_rating': db_stats.get('avg_rating', 0.0),
                'total_documents': kb_stats.get('total_documents', 0),
                'avg_confidence': db_stats.get('avg_confidence', 0.0),
                'external_api_calls': db_stats.get('external_calls', 0),
                'internal_kb_calls': db_stats.get('internal_calls', 0),
                'system_ready': True
            }
        except Exception as e:
            self.logger.error(f"Error getting stats: {e}")
            return {'error': str(e), 'system_ready': False}
    
    async def _log_query(self, question: str, response: str, confidence: float, 
                        processing_time: float, source_type: str):
        """Log query interaction with source information."""
        try:
            await self.database.log_query(question, response, confidence, processing_time, source_type)
        except Exception as e:
            self.logger.error(f"Error logging query: {e}")
    
    def _get_elapsed_time(self, start_time: datetime) -> float:
        """Calculate elapsed time in seconds."""
        return (datetime.now() - start_time).total_seconds()
    
    async def close(self):
        """Clean up resources."""
        try:
            await self.database.close()
            self.logger.info("Smart MentalHealthRAG system closed")
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")
    
    def __str__(self) -> str:
        """String representation of the system."""
        return f"SmartMentalHealthRAG(kb_docs={self.knowledge_base.get_document_count()})"