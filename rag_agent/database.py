"""
Database operations for the RAG Agent system.
Handles query logging, feedback storage, and usage analytics.
"""

import asyncio
import logging
import json
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import aiosqlite
import tempfile
import os

class QueryLog:
    """Data model for query interactions."""
    
    def __init__(self, query: str, response: str, confidence: float, 
                 processing_time: float, source_type: str = "internal", 
                 feedback_rating: Optional[float] = None):
        self.query = query
        self.response = response
        self.confidence = confidence
        self.processing_time = processing_time
        self.source_type = source_type
        self.feedback_rating = feedback_rating
        self.timestamp = datetime.now()
        self.id: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            'id': self.id,
            'query': self.query,
            'response': self.response,
            'confidence': self.confidence,
            'processing_time': self.processing_time,
            'source_type': self.source_type,
            'feedback_rating': self.feedback_rating,
            'timestamp': self.timestamp.isoformat()
        }

class Database:
    """
    Manages database operations for the RAG system.
    Handles query logging, feedback storage, and performance analytics using SQLite.
    """
    
    def __init__(self):
        """Initialize database manager."""
        self.logger = logging.getLogger("rag_agent")
        
        # Use temporary directory for CodeSandbox
        self.db_path = os.path.join(tempfile.gettempdir(), "mental_health_rag.db")
        self._initialized = False
        
        self.logger.info(f"Database manager initialized: {self.db_path}")
    
    async def _initialize(self):
        """Initialize database schema if not already done."""
        if self._initialized:
            return
        
        try:
            async with aiosqlite.connect(self.db_path) as db:
                # Main query logs table
                await db.execute("""
                    CREATE TABLE IF NOT EXISTS query_logs (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        query TEXT NOT NULL,
                        response TEXT NOT NULL,
                        confidence REAL NOT NULL,
                        processing_time REAL NOT NULL,
                        source_type TEXT NOT NULL DEFAULT 'internal',
                        feedback_rating REAL,
                        timestamp DATETIME NOT NULL,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Performance metrics table
                await db.execute("""
                    CREATE TABLE IF NOT EXISTS performance_metrics (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        metric_name TEXT NOT NULL,
                        metric_value REAL NOT NULL,
                        source_type TEXT,
                        timestamp DATETIME NOT NULL,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Create indexes for better performance
                await db.execute("""
                    CREATE INDEX IF NOT EXISTS idx_query_logs_timestamp 
                    ON query_logs(timestamp)
                """)
                await db.execute("""
                    CREATE INDEX IF NOT EXISTS idx_query_logs_source_type 
                    ON query_logs(source_type)
                """)
                await db.execute("""
                    CREATE INDEX IF NOT EXISTS idx_query_logs_feedback 
                    ON query_logs(feedback_rating) WHERE feedback_rating IS NOT NULL
                """)
                
                await db.commit()
                
            self._initialized = True
            self.logger.info("Database schema initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Error initializing database: {e}")
            raise
    
    async def log_query(self, query: str, response: str, confidence: float, 
                       processing_time: float, source_type: str = "internal") -> bool:
        """
        Log a query interaction to the database.
        
        Args:
            query: User query text
            response: System response
            confidence: Confidence score (0.0-1.0)
            processing_time: Time taken to process query (seconds)
            source_type: Source of response ("internal", "external_api", etc.)
            
        Returns:
            True if query was logged successfully
        """
        await self._initialize()
        
        try:
            async with aiosqlite.connect(self.db_path) as db:
                cursor = await db.execute("""
                    INSERT INTO query_logs 
                    (query, response, confidence, processing_time, source_type, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    query[:1000],  # Limit query length
                    response[:2000],  # Limit response length
                    confidence,
                    processing_time,
                    source_type,
                    datetime.now()
                ))
                
                await db.commit()
                query_id = cursor.lastrowid
                
                self.logger.debug(f"Query logged with ID: {query_id}")
                return True
                
        except Exception as e:
            self.logger.error(f"Error logging query: {e}")
            return False
    
    async def add_feedback(self, query: str, rating: float) -> bool:
        """
        Add user feedback for the most recent matching query.
        
        Args:
            query: Original query text
            rating: User rating (1.0-5.0)
            
        Returns:
            True if feedback was recorded successfully
        """
        await self._initialize()
        
        try:
            async with aiosqlite.connect(self.db_path) as db:
                # Find the most recent matching query
                cursor = await db.execute("""
                    SELECT id FROM query_logs 
                    WHERE query = ?
                    ORDER BY timestamp DESC 
                    LIMIT 1
                """, (query,))
                
                row = await cursor.fetchone()
                if not row:
                    self.logger.warning(f"No matching query found for feedback")
                    return False
                
                query_id = row[0]
                
                # Update with feedback rating
                await db.execute("""
                    UPDATE query_logs 
                    SET feedback_rating = ?
                    WHERE id = ?
                """, (rating, query_id))
                
                await db.commit()
                
                self.logger.info(f"Feedback recorded: {rating}/5.0 for query {query_id}")
                return True
                
        except Exception as e:
            self.logger.error(f"Error adding feedback: {e}")
            return False
    
    async def get_stats(self, days: int = 30) -> Dict[str, Any]:
        """
        Get comprehensive system statistics.
        
        Args:
            days: Number of days to analyze (default: 30)
            
        Returns:
            Dictionary with system statistics
        """
        await self._initialize()
        
        try:
            since_date = datetime.now() - timedelta(days=days)
            
            async with aiosqlite.connect(self.db_path) as db:
                stats = {}
                
                # Total queries
                cursor = await db.execute("""
                    SELECT COUNT(*) FROM query_logs 
                    WHERE timestamp >= ?
                """, (since_date,))
                stats['total_queries'] = (await cursor.fetchone())[0]
                
                # Average confidence
                cursor = await db.execute("""
                    SELECT AVG(confidence) FROM query_logs 
                    WHERE timestamp >= ?
                """, (since_date,))
                stats['avg_confidence'] = round((await cursor.fetchone())[0] or 0.0, 3)
                
                # Average processing time
                cursor = await db.execute("""
                    SELECT AVG(processing_time) FROM query_logs 
                    WHERE timestamp >= ?
                """, (since_date,))
                stats['avg_processing_time'] = round((await cursor.fetchone())[0] or 0.0, 3)
                
                # Source type distribution
                cursor = await db.execute("""
                    SELECT source_type, COUNT(*) FROM query_logs 
                    WHERE timestamp >= ?
                    GROUP BY source_type
                """, (since_date,))
                source_distribution = dict(await cursor.fetchall())
                stats['internal_kb_calls'] = source_distribution.get('internal_knowledge', 0)
                stats['external_api_calls'] = source_distribution.get('external_api', 0)
                stats['source_distribution'] = source_distribution
                
                # Feedback statistics
                cursor = await db.execute("""
                    SELECT AVG(feedback_rating), COUNT(*) FROM query_logs 
                    WHERE timestamp >= ? AND feedback_rating IS NOT NULL
                """, (since_date,))
                feedback_row = await cursor.fetchone()
                stats['avg_rating'] = round(feedback_row[0] or 0.0, 2)
                stats['feedback_count'] = feedback_row[1]
                
                # Feedback rate
                if stats['total_queries'] > 0:
                    stats['feedback_rate'] = round(stats['feedback_count'] / stats['total_queries'], 3)
                else:
                    stats['feedback_rate'] = 0.0
                
                # Query patterns (most common words)
                cursor = await db.execute("""
                    SELECT query FROM query_logs 
                    WHERE timestamp >= ?
                    ORDER BY timestamp DESC
                    LIMIT 100
                """, (since_date,))
                recent_queries = [row[0] for row in await cursor.fetchall()]
                stats['query_patterns'] = self._analyze_query_patterns(recent_queries)
                
                return stats
                
        except Exception as e:
            self.logger.error(f"Error getting stats: {e}")
            return {'error': str(e)}
    
    def _analyze_query_patterns(self, queries: List[str]) -> Dict[str, int]:
        """Analyze common patterns in queries."""
        patterns = {}
        
        # Mental health keywords to track
        keywords = [
            'anxiety', 'anxious', 'panic', 'stress', 'depression', 'depressed',
            'sleep', 'insomnia', 'worry', 'fear', 'sad', 'overwhelmed',
            'crisis', 'help', 'breathing', 'calm', 'technique'
        ]
        
        for query in queries:
            query_lower = query.lower()
            for keyword in keywords:
                if keyword in query_lower:
                    patterns[keyword] = patterns.get(keyword, 0) + 1
        
        # Return top 10 most common patterns
        sorted_patterns = sorted(patterns.items(), key=lambda x: x[1], reverse=True)
        return dict(sorted_patterns[:10])
    
    async def get_recent_queries(self, limit: int = 10, source_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get recent query logs.
        
        Args:
            limit: Maximum number of queries to return
            source_type: Filter by source type (optional)
            
        Returns:
            List of recent query dictionaries
        """
        await self._initialize()
        
        try:
            async with aiosqlite.connect(self.db_path) as db:
                if source_type:
                    cursor = await db.execute("""
                        SELECT * FROM query_logs 
                        WHERE source_type = ?
                        ORDER BY timestamp DESC 
                        LIMIT ?
                    """, (source_type, limit))
                else:
                    cursor = await db.execute("""
                        SELECT * FROM query_logs 
                        ORDER BY timestamp DESC 
                        LIMIT ?
                    """, (limit,))
                
                rows = await cursor.fetchall()
                columns = [description[0] for description in cursor.description]
                
                queries = []
                for row in rows:
                    query_dict = dict(zip(columns, row))
                    queries.append(query_dict)
                
                return queries
                
        except Exception as e:
            self.logger.error(f"Error getting recent queries: {e}")
            return []
    
    async def record_performance_metric(self, metric_name: str, value: float, 
                                      source_type: Optional[str] = None) -> bool:
        """
        Record a performance metric.
        
        Args:
            metric_name: Name of the metric
            value: Metric value
            source_type: Optional source type context
            
        Returns:
            True if metric was recorded successfully
        """
        await self._initialize()
        
        try:
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute("""
                    INSERT INTO performance_metrics (metric_name, metric_value, source_type, timestamp)
                    VALUES (?, ?, ?, ?)
                """, (metric_name, value, source_type, datetime.now()))
                
                await db.commit()
                return True
                
        except Exception as e:
            self.logger.error(f"Error recording performance metric: {e}")
            return False
    
    async def get_feedback_summary(self) -> Dict[str, Any]:
        """
        Get detailed feedback analysis.
        
        Returns:
            Dictionary with feedback statistics
        """
        await self._initialize()
        
        try:
            async with aiosqlite.connect(self.db_path) as db:
                # Rating distribution
                cursor = await db.execute("""
                    SELECT feedback_rating, COUNT(*) FROM query_logs 
                    WHERE feedback_rating IS NOT NULL
                    GROUP BY feedback_rating
                    ORDER BY feedback_rating
                """)
                rating_distribution = dict(await cursor.fetchall())
                
                # Source type performance
                cursor = await db.execute("""
                    SELECT source_type, AVG(feedback_rating), COUNT(*) FROM query_logs 
                    WHERE feedback_rating IS NOT NULL
                    GROUP BY source_type
                """)
                source_performance = {}
                for row in await cursor.fetchall():
                    source_type, avg_rating, count = row
                    source_performance[source_type] = {
                        'avg_rating': round(avg_rating, 2),
                        'count': count
                    }
                
                # Recent feedback trends
                cursor = await db.execute("""
                    SELECT DATE(timestamp) as date, AVG(feedback_rating) as avg_rating
                    FROM query_logs 
                    WHERE feedback_rating IS NOT NULL 
                        AND timestamp >= date('now', '-7 days')
                    GROUP BY DATE(timestamp)
                    ORDER BY date
                """)
                daily_trends = dict(await cursor.fetchall())
                
                return {
                    'rating_distribution': rating_distribution,
                    'source_performance': source_performance,
                    'daily_trends': daily_trends
                }
                
        except Exception as e:
            self.logger.error(f"Error getting feedback summary: {e}")
            return {}
    
    async def cleanup_old_logs(self, days_to_keep: int = 90) -> int:
        """
        Clean up old log entries to manage database size.
        
        Args:
            days_to_keep: Number of days of logs to retain
            
        Returns:
            Number of records deleted
        """
        await self._initialize()
        
        try:
            cutoff_date = datetime.now() - timedelta(days=days_to_keep)
            
            async with aiosqlite.connect(self.db_path) as db:
                cursor = await db.execute("""
                    DELETE FROM query_logs 
                    WHERE timestamp < ?
                """, (cutoff_date,))
                
                deleted_count = cursor.rowcount
                
                await db.execute("""
                    DELETE FROM performance_metrics 
                    WHERE timestamp < ?
                """, (cutoff_date,))
                
                await db.commit()
                
                self.logger.info(f"Cleaned up {deleted_count} old log entries")
                return deleted_count
                
        except Exception as e:
            self.logger.error(f"Error cleaning up old logs: {e}")
            return 0
    
    async def export_data(self, days: int = 30) -> Dict[str, Any]:
        """
        Export system data for analysis or backup.
        
        Args:
            days: Number of days of data to export
            
        Returns:
            Dictionary with exported data
        """
        await self._initialize()
        
        try:
            since_date = datetime.now() - timedelta(days=days)
            
            async with aiosqlite.connect(self.db_path) as db:
                # Export query logs
                cursor = await db.execute("""
                    SELECT * FROM query_logs 
                    WHERE timestamp >= ?
                    ORDER BY timestamp
                """, (since_date,))
                
                query_logs = []
                columns = [description[0] for description in cursor.description]
                for row in await cursor.fetchall():
                    query_logs.append(dict(zip(columns, row)))
                
                # Export performance metrics
                cursor = await db.execute("""
                    SELECT * FROM performance_metrics 
                    WHERE timestamp >= ?
                    ORDER BY timestamp
                """, (since_date,))
                
                performance_metrics = []
                for row in await cursor.fetchall():
                    performance_metrics.append(dict(zip(columns, row)))
                
                return {
                    'export_date': datetime.now().isoformat(),
                    'days_exported': days,
                    'query_logs': query_logs,
                    'performance_metrics': performance_metrics,
                    'total_queries': len(query_logs),
                    'total_metrics': len(performance_metrics)
                }
                
        except Exception as e:
            self.logger.error(f"Error exporting data: {e}")
            return {'error': str(e)}
    
    async def close(self):
        """Close database connections and cleanup."""
        try:
            # SQLite connections are automatically closed in async context managers
            # Just log the closure
            self.logger.info("Database connections closed")
        except Exception as e:
            self.logger.error(f"Error during database cleanup: {e}")
    
    def __str__(self) -> str:
        """String representation of the database."""
        return f"Database(path={self.db_path})"