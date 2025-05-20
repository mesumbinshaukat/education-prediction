import json
from pathlib import Path
from typing import Dict, List, Any
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/knowledge_base.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class KnowledgeLoader:
    """Utility class for loading and processing the knowledge base."""
    
    def __init__(self, knowledge_dir: str = "knowledge"):
        """Initialize the knowledge loader.
        
        Args:
            knowledge_dir (str): Directory containing knowledge base files
        """
        self.knowledge_dir = Path(knowledge_dir)
        self.knowledge_base = {}
        self.processed_chunks = []
        
    def load_knowledge_base(self) -> Dict[str, Any]:
        """Load the knowledge base from JSON files.
        
        Returns:
            Dict[str, Any]: The loaded knowledge base
        """
        try:
            # Load main knowledge file
            knowledge_file = self.knowledge_dir / "education_knowledge.json"
            if not knowledge_file.exists():
                logger.error(f"Knowledge file not found: {knowledge_file}")
                return {}
                
            with open(knowledge_file, 'r', encoding='utf-8') as f:
                self.knowledge_base = json.load(f)
                
            logger.info("Successfully loaded knowledge base")
            return self.knowledge_base
            
        except Exception as e:
            logger.error(f"Error loading knowledge base: {str(e)}")
            return {}
            
    def process_knowledge_chunks(self) -> List[Dict[str, Any]]:
        """Process the knowledge base into searchable chunks.
        
        Returns:
            List[Dict[str, Any]]: List of processed knowledge chunks
        """
        if not self.knowledge_base:
            logger.warning("Knowledge base not loaded. Loading now...")
            self.load_knowledge_base()
            
        try:
            chunks = []
            
            # Process system features
            if "system_features" in self.knowledge_base:
                for feature_type, features in self.knowledge_base["system_features"].items():
                    if feature_type == "prediction_system":
                        # Process metrics
                        for metric, details in features["metrics"].items():
                            chunk = {
                                "type": "system_feature",
                                "category": "prediction_metric",
                                "metric": metric,
                                "content": details,
                                "timestamp": datetime.now().isoformat()
                            }
                            chunks.append(chunk)
                            
                        # Process performance categories
                        for category, details in features["performance_categories"].items():
                            chunk = {
                                "type": "system_feature",
                                "category": "performance_category",
                                "name": category,
                                "content": details,
                                "timestamp": datetime.now().isoformat()
                            }
                            chunks.append(chunk)
                            
                    elif feature_type == "features":
                        for feature, details in features.items():
                            chunk = {
                                "type": "system_feature",
                                "category": "feature",
                                "name": feature,
                                "content": details,
                                "timestamp": datetime.now().isoformat()
                            }
                            chunks.append(chunk)
                            
            # Process educational concepts
            if "educational_concepts" in self.knowledge_base:
                for concept_type, concepts in self.knowledge_base["educational_concepts"].items():
                    for concept, details in concepts.items():
                        chunk = {
                            "type": "educational_concept",
                            "category": concept_type,
                            "name": concept,
                            "content": details,
                            "timestamp": datetime.now().isoformat()
                        }
                        chunks.append(chunk)
                        
            # Process system usage
            if "system_usage" in self.knowledge_base:
                for usage_type, details in self.knowledge_base["system_usage"].items():
                    chunk = {
                        "type": "system_usage",
                        "category": usage_type,
                        "content": details,
                        "timestamp": datetime.now().isoformat()
                    }
                    chunks.append(chunk)
                    
            self.processed_chunks = chunks
            logger.info(f"Successfully processed {len(chunks)} knowledge chunks")
            return chunks
            
        except Exception as e:
            logger.error(f"Error processing knowledge chunks: {str(e)}")
            return []
            
    def get_knowledge_by_type(self, knowledge_type: str) -> List[Dict[str, Any]]:
        """Get knowledge chunks of a specific type.
        
        Args:
            knowledge_type (str): Type of knowledge to retrieve
            
        Returns:
            List[Dict[str, Any]]: List of matching knowledge chunks
        """
        if not self.processed_chunks:
            logger.warning("Knowledge chunks not processed. Processing now...")
            self.process_knowledge_chunks()
            
        return [chunk for chunk in self.processed_chunks if chunk["type"] == knowledge_type]
        
    def get_knowledge_by_category(self, category: str) -> List[Dict[str, Any]]:
        """Get knowledge chunks of a specific category.
        
        Args:
            category (str): Category of knowledge to retrieve
            
        Returns:
            List[Dict[str, Any]]: List of matching knowledge chunks
        """
        if not self.processed_chunks:
            logger.warning("Knowledge chunks not processed. Processing now...")
            self.process_knowledge_chunks()
            
        return [chunk for chunk in self.processed_chunks if chunk["category"] == category]
        
    def search_knowledge(self, query: str) -> List[Dict[str, Any]]:
        """Search knowledge chunks by content.
        
        Args:
            query (str): Search query
            
        Returns:
            List[Dict[str, Any]]: List of matching knowledge chunks
        """
        if not self.processed_chunks:
            logger.warning("Knowledge chunks not processed. Processing now...")
            self.process_knowledge_chunks()
            
        query = query.lower()
        matches = []
        
        for chunk in self.processed_chunks:
            # Search in content
            content_str = str(chunk["content"]).lower()
            if query in content_str:
                matches.append(chunk)
                continue
                
            # Search in name if available
            if "name" in chunk and query in chunk["name"].lower():
                matches.append(chunk)
                continue
                
            # Search in category
            if query in chunk["category"].lower():
                matches.append(chunk)
                
        return matches 