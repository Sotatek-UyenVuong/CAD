"""
Image Search Service for CAD Document Chat Assistant
Handles image extraction, embedding, and semantic search using Qdrant + MongoDB
Supports multiple embedding providers: OpenAI, Cohere
"""

import os
import io
import base64
import hashlib
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, asdict
import json

# PDF and Image processing
from pdf2image import convert_from_path
from PIL import Image
import numpy as np

# Vector store and database
from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models
from qdrant_client.http.models import Distance, VectorParams, PointStruct
from pymongo import MongoClient

# Gemini for image understanding
import google.generativeai as genai


@dataclass
class EmbeddingConfig:
    """Configuration for embedding providers"""
    provider: str = "openai"
    model: str = "text-embedding-ada-002"
    batch_size: int = 16
    
    # Dimension mappings for different models
    DIMENSIONS: Dict[str, int] = None
    
    def __post_init__(self):
        if self.DIMENSIONS is None:
            self.DIMENSIONS = {
                # OpenAI models
                "text-embedding-3-small": 1536,
                "text-embedding-3-large": 3072,
                "text-embedding-ada-002": 1536,
                # Cohere models
                "embed-english-v3.0": 1024,
                "embed-multilingual-v3.0": 1024,
                "embed-english-light-v3.0": 384,
                "embed-multilingual-light-v3.0": 384,
            }
    
    @property
    def dimension(self) -> int:
        """Get embedding dimension for current model"""
        return self.DIMENSIONS.get(self.model, 1536)
    
    @classmethod
    def from_env(cls) -> 'EmbeddingConfig':
        """Create config from environment variables"""
        return cls(
            provider=os.getenv("EMBEDDING_PROVIDER", "openai"),
            model=os.getenv("EMBEDDING_MODEL", "text-embedding-ada-002"),
            batch_size=int(os.getenv("EMBEDDING_BATCH_SIZE", "16"))
        )


@dataclass
class Document:
    """Document model for MongoDB"""
    document_id: str
    name: str
    url: str
    total_pages: int = 0
    pdf_path: str = ""
    created_at: datetime = None
    updated_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()
        if self.updated_at is None:
            self.updated_at = datetime.utcnow()
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for MongoDB"""
        return {
            "document_id": self.document_id,
            "name": self.name,
            "url": self.url,
            "total_pages": self.total_pages,
            "pdf_path": self.pdf_path,
            "created_at": self.created_at,
            "updated_at": self.updated_at
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Document':
        """Create from dictionary"""
        return cls(
            document_id=data.get("document_id", ""),
            name=data.get("name", ""),
            url=data.get("url", ""),
            total_pages=data.get("total_pages", 0),
            pdf_path=data.get("pdf_path", ""),
            created_at=data.get("created_at"),
            updated_at=data.get("updated_at")
        )


@dataclass
class Chatbot:
    """Chatbot session model for MongoDB"""
    chatbot_id: str
    document_id: str
    chat_history: List[Dict[str, str]]  # List of {"question": "...", "answer": "..."}
    url: str
    created_at: datetime = None
    updated_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()
        if self.updated_at is None:
            self.updated_at = datetime.utcnow()
        if self.chat_history is None:
            self.chat_history = []
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for MongoDB"""
        return {
            "chatbot_id": self.chatbot_id,
            "document_id": self.document_id,
            "chat_history": self.chat_history,
            "url": self.url,
            "created_at": self.created_at,
            "updated_at": self.updated_at
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Chatbot':
        """Create from dictionary"""
        return cls(
            chatbot_id=data.get("chatbot_id", ""),
            document_id=data.get("document_id", ""),
            chat_history=data.get("chat_history", []),
            url=data.get("url", ""),
            created_at=data.get("created_at"),
            updated_at=data.get("updated_at")
        )


@dataclass
class ImageMetadata:
    """Metadata for extracted images (simplified for embedding)"""
    image_id: str
    document_id: str
    document_name: str
    page_number: int
    description: str          # Mô tả ảnh - dùng cho embedding
    image_type: str           # Loại bản vẽ - dùng cho embedding
    thumbnail_base64: str
    full_image_path: str
    created_at: datetime
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for MongoDB"""
        return {
            "image_id": self.image_id,
            "document_id": self.document_id,
            "document_name": self.document_name,
            "page_number": self.page_number,
            "description": self.description,
            "image_type": self.image_type,
            "thumbnail_base64": self.thumbnail_base64,
            "full_image_path": self.full_image_path,
            "created_at": self.created_at.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'ImageMetadata':
        """Create from dictionary"""
        created_at = data.get('created_at')
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at)
        return cls(
            image_id=data.get("image_id", ""),
            document_id=data.get("document_id", ""),
            document_name=data.get("document_name", ""),
            page_number=data.get("page_number", 0),
            description=data.get("description", ""),
            image_type=data.get("image_type", ""),
            thumbnail_base64=data.get("thumbnail_base64", ""),
            full_image_path=data.get("full_image_path", ""),
            created_at=created_at
        )


class EmbeddingProvider:
    """Base class for embedding providers"""
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        raise NotImplementedError
    
    def embed_query(self, text: str) -> List[float]:
        raise NotImplementedError


class OpenAIEmbedding(EmbeddingProvider):
    """OpenAI embedding provider"""
    
    def __init__(self, api_key: str, model: str = "text-embedding-ada-002"):
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key)
        self.model = model
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple documents"""
        response = self.client.embeddings.create(
            input=texts,
            model=self.model
        )
        return [item.embedding for item in response.data]
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query"""
        response = self.client.embeddings.create(
            input=[text],
            model=self.model
        )
        return response.data[0].embedding


class CohereEmbedding(EmbeddingProvider):
    """Cohere embedding provider"""
    
    def __init__(self, api_key: str, model: str = "embed-english-v3.0"):
        import cohere
        self.client = cohere.ClientV2(api_key=api_key)
        self.model = model
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple documents"""
        response = self.client.embed(
            texts=texts,
            model=self.model,
            input_type="search_document",
            embedding_types=["float"]
        )
        return [list(emb) for emb in response.embeddings.float_]
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query"""
        response = self.client.embed(
            texts=[text],
            model=self.model,
            input_type="search_query",
            embedding_types=["float"]
        )
        return list(response.embeddings.float_[0])


class ImageSearchService:
    """
    Service for extracting images from CAD documents,
    generating embeddings, and performing semantic search
    """
    
    COLLECTION_NAME = "cad_images"
    
    def __init__(
        self,
        gemini_api_key: str,
        embedding_config: EmbeddingConfig,
        openai_api_key: Optional[str] = None,
        cohere_api_key: Optional[str] = None,
        qdrant_url: str = "localhost",
        qdrant_port: int = 6333,
        mongodb_uri: str = "mongodb://localhost:27017",
        mongodb_db: str = "cad_assistant",
        images_dir: str = "./extracted_images"
    ):
        """
        Initialize the Image Search Service
        
        Args:
            gemini_api_key: API key for Google Gemini (image analysis)
            embedding_config: Configuration for embedding provider
            openai_api_key: API key for OpenAI (if using OpenAI embeddings)
            cohere_api_key: API key for Cohere (if using Cohere embeddings)
            qdrant_url: Qdrant server URL
            qdrant_port: Qdrant server port
            mongodb_uri: MongoDB connection URI
            mongodb_db: MongoDB database name
            images_dir: Directory to store extracted images
        """
        # Configure Gemini for image analysis
        genai.configure(api_key=gemini_api_key)
        self.vision_model = genai.GenerativeModel("models/gemini-2.5-pro")
        
        # Store embedding config
        self.embedding_config = embedding_config
        self.embedding_dim = embedding_config.dimension
        
        # Initialize embedding provider based on config
        self.embedding_provider = self._create_embedding_provider(
            embedding_config, openai_api_key, cohere_api_key
        )
        
        # Initialize Qdrant (support URL with http:// or host:port)
        qdrant_api_key = os.getenv("QDRANT_API_KEY")
        if qdrant_api_key and qdrant_api_key.lower() != "none":
            # With API key
            self.qdrant = QdrantClient(
                url=qdrant_url if qdrant_url.startswith("http") else f"http://{qdrant_url}:{qdrant_port}",
                api_key=qdrant_api_key
            )
        elif qdrant_url.startswith("http"):
            # URL format (e.g., http://192.168.200.23:6333)
            self.qdrant = QdrantClient(url=qdrant_url)
        else:
            # Host:port format
            self.qdrant = QdrantClient(host=qdrant_url, port=qdrant_port)
        self._ensure_collection_exists()
        
        # Initialize MongoDB
        self.mongo_client = MongoClient(mongodb_uri)
        self.db = self.mongo_client[mongodb_db]
        self.images_collection = self.db["images"]
        self.documents_collection = self.db["documents"]
        self.chatbots_collection = self.db["chatbots"]
        
        # Create indexes
        self._ensure_mongodb_indexes()
        
        # Images directory
        self.images_dir = Path(images_dir)
        self.images_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"✅ Embedding Provider: {embedding_config.provider}")
        print(f"✅ Embedding Model: {embedding_config.model}")
        print(f"✅ Embedding Dimension: {self.embedding_dim}")
    
    def _create_embedding_provider(
        self,
        config: EmbeddingConfig,
        openai_api_key: Optional[str],
        cohere_api_key: Optional[str]
    ) -> EmbeddingProvider:
        """Create the appropriate embedding provider"""
        
        if config.provider.lower() == "openai":
            if not openai_api_key:
                raise ValueError("OpenAI API key required for OpenAI embeddings")
            return OpenAIEmbedding(openai_api_key, config.model)
        
        elif config.provider.lower() == "cohere":
            if not cohere_api_key:
                raise ValueError("Cohere API key required for Cohere embeddings")
            return CohereEmbedding(cohere_api_key, config.model)
        
        else:
            raise ValueError(f"Unsupported embedding provider: {config.provider}")
    
    def _ensure_collection_exists(self):
        """Ensure Qdrant collection exists with correct dimensions"""
        collections = self.qdrant.get_collections().collections
        collection_names = [c.name for c in collections]
        
        if self.COLLECTION_NAME not in collection_names:
            self.qdrant.create_collection(
                collection_name=self.COLLECTION_NAME,
                vectors_config=VectorParams(
                    size=self.embedding_dim,
                    distance=Distance.COSINE
                )
            )
            print(f"✅ Created Qdrant collection: {self.COLLECTION_NAME} (dim={self.embedding_dim})")
        else:
            # Verify existing collection has correct dimensions
            collection_info = self.qdrant.get_collection(self.COLLECTION_NAME)
            existing_dim = collection_info.config.params.vectors.size
            if existing_dim != self.embedding_dim:
                print(f"⚠️ Warning: Existing collection has dim={existing_dim}, expected {self.embedding_dim}")
                print(f"   Consider deleting and recreating the collection")
    
    def _ensure_mongodb_indexes(self):
        """Create MongoDB indexes for efficient queries"""
        # Images collection indexes
        self.images_collection.create_index("image_id", unique=True)
        self.images_collection.create_index("document_id")
        self.images_collection.create_index("page_number")
        self.images_collection.create_index([("description", "text")])
        
        # Documents collection indexes
        self.documents_collection.create_index("document_id", unique=True)
        self.documents_collection.create_index("name")
        self.documents_collection.create_index("url")
        
        # Chatbots collection indexes
        self.chatbots_collection.create_index("chatbot_id", unique=True)
        self.chatbots_collection.create_index("document_id")
    
    def _generate_image_id(self, document_id: str, page_num: int, image_index: int) -> str:
        """Generate unique image ID"""
        content = f"{document_id}_{page_num}_{image_index}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def _image_to_base64(self, image: Image.Image, max_size: int = 200) -> str:
        """Convert PIL Image to base64 thumbnail"""
        # Create thumbnail
        image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        
        # Convert to base64
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode()
    
    def _save_image(self, image: Image.Image, image_id: str) -> str:
        """Save image to disk and return path"""
        image_path = self.images_dir / f"{image_id}.png"
        image.save(image_path, format="PNG")
        return str(image_path)
    
    def _analyze_pdf_with_gemini(self, pdf_path: str) -> List[Dict[str, Any]]:
        """
        Use Gemini to analyze ENTIRE PDF at once and return info for all pages.
        Much faster than analyzing page by page!
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            List of {page_number, description, drawing_type} for each page
        """
        # Upload PDF to Gemini
        uploaded_file = genai.upload_file(pdf_path)
        
        prompt = """Analyze this CAD/architectural document and provide information for EACH PAGE.

For each page, extract:
1. **page_number**: The page number (starting from 1)
2. **description**: A detailed description of what this page shows (room layout, equipment, symbols, annotations, dimensions, specifications, drawing title, etc.)
3. **drawing_type**: Classify the page into one of these categories:
   - floor_plan (mặt bằng)
   - elevation (mặt đứng)
   - section (mặt cắt)
   - detail (chi tiết)
   - electrical_plan (điện)
   - fire_safety (PCCC)
   - plumbing (cấp thoát nước)
   - hvac (điều hòa)
   - structural (kết cấu)
   - site_plan (tổng mặt bằng)
   - cover_page (trang bìa)
   - index (mục lục)
   - other

IMPORTANT: Return a JSON array with information for ALL pages in the document.

Response format:
[
    {"page_number": 1, "description": "Cover page with project title...", "drawing_type": "cover_page"},
    {"page_number": 2, "description": "First floor plan showing...", "drawing_type": "floor_plan"},
    {"page_number": 3, "description": "Electrical layout for...", "drawing_type": "electrical_plan"}
]"""

        try:
            print("  🔍 Analyzing entire PDF with Gemini (1 API call)...")
            response = self.vision_model.generate_content([uploaded_file, prompt])
            
            # Parse JSON response
            text = response.text
            # Extract JSON from response (handle markdown code blocks)
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0]
            elif "```" in text:
                text = text.split("```")[1].split("```")[0]
            
            pages_info = json.loads(text.strip())
            print(f"  ✅ Gemini analyzed {len(pages_info)} pages")
            return pages_info
            
        except Exception as e:
            print(f"⚠️ Error analyzing PDF with Gemini: {e}")
            return []
    
    async def _analyze_single_image_with_gemini(self, image: Image.Image) -> Dict[str, Any]:
        """
        Analyze a single image with Gemini (used for image search by upload)
        
        Returns:
            Dictionary with description and drawing_type
        """
        # Convert image to bytes for Gemini
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        image_bytes = buffer.getvalue()
        
        prompt = """Analyze this CAD/architectural drawing and provide:

1. **Description**: A detailed description of what this drawing shows

2. **Drawing Type**: Classify this drawing (floor_plan, elevation, section, detail, electrical_plan, fire_safety, plumbing, hvac, structural, site_plan, other)

Respond in JSON format:
{
    "description": "detailed description here...",
    "drawing_type": "floor_plan"
}"""

        try:
            response = self.vision_model.generate_content([
                prompt,
                {"mime_type": "image/png", "data": image_bytes}
            ])
            
            # Parse JSON response
            text = response.text
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0]
            elif "```" in text:
                text = text.split("```")[1].split("```")[0]
            
            return json.loads(text.strip())
            
        except Exception as e:
            print(f"⚠️ Error analyzing image: {e}")
            return {
                "description": "CAD drawing - analysis failed",
                "drawing_type": "unknown"
            }
    
    def _get_embedding(self, text: str) -> List[float]:
        """Generate embedding for document text"""
        return self.embedding_provider.embed_documents([text])[0]
    
    def _get_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts in batch"""
        batch_size = self.embedding_config.batch_size
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            embeddings = self.embedding_provider.embed_documents(batch)
            all_embeddings.extend(embeddings)
        
        return all_embeddings
    
    def _get_query_embedding(self, text: str) -> List[float]:
        """Generate embedding for search query"""
        return self.embedding_provider.embed_query(text)
    
    async def process_document(
        self,
        pdf_path: str,
        document_name: str,
        document_id: Optional[str] = None,
        dpi: int = 150
    ) -> Dict[str, Any]:
        """
        Process a PDF document: analyze entire PDF at once, then extract thumbnails
        
        OPTIMIZED: Uses single Gemini API call for entire PDF instead of per-page
        
        Args:
            pdf_path: Path to the PDF file
            document_name: Name of the document
            document_id: Optional document ID (generated if not provided)
            dpi: DPI for image extraction (for thumbnails only)
            
        Returns:
            Dictionary with processing results
        """
        if document_id is None:
            document_id = hashlib.md5(pdf_path.encode()).hexdigest()
        
        print(f"📄 Processing document: {document_name}")
        
        results = {
            "document_id": document_id,
            "document_name": document_name,
            "total_pages": 0,
            "images_processed": 0,
            "errors": []
        }
        
        # Step 1: Analyze ENTIRE PDF with Gemini (1 API call - FAST!)
        pages_analysis = self._analyze_pdf_with_gemini(pdf_path)
        
        if not pages_analysis:
            results["errors"].append("Failed to analyze PDF with Gemini")
            return results
        
        # Step 2: Convert PDF pages to images (for thumbnails)
        print("  📸 Extracting page thumbnails...")
        pages = convert_from_path(pdf_path, dpi=dpi)
        results["total_pages"] = len(pages)
        
        # Create lookup dict for analysis by page number
        analysis_by_page = {item.get("page_number", 0): item for item in pages_analysis}
        
        # Store document metadata
        self.documents_collection.update_one(
            {"document_id": document_id},
            {"$set": {
                "document_id": document_id,
                "document_name": document_name,
                "total_pages": len(pages),
                "pdf_path": pdf_path,
                "processed_at": datetime.utcnow(),
                "embedding_provider": self.embedding_config.provider,
                "embedding_model": self.embedding_config.model
            }},
            upsert=True
        )
        
        # Step 3: Process each page (create thumbnails, embeddings, store)
        print("  💾 Storing pages and generating embeddings...")
        
        # Prepare batch embedding texts
        embedding_texts = []
        page_data_list = []
        
        for page_num, page_image in enumerate(pages, start=1):
            image_id = self._generate_image_id(document_id, page_num, 0)
            
            # Skip if already processed
            existing = self.images_collection.find_one({"image_id": image_id})
            if existing:
                print(f"    ⏭️ Page {page_num} already processed, skipping")
                results["images_processed"] += 1
                continue
            
            # Get analysis for this page (fallback if missing)
            analysis = analysis_by_page.get(page_num, {
                "description": f"Page {page_num} of {document_name}",
                "drawing_type": "unknown"
            })
            
            description = analysis.get("description", "")
            image_type = analysis.get("drawing_type", "unknown")
            
            # Generate thumbnail and save full image
            thumbnail_b64 = self._image_to_base64(page_image.copy())
            full_image_path = self._save_image(page_image, image_id)
            
            # Prepare data for batch processing
            embedding_text = f"{description} [Type: {image_type}]"
            embedding_texts.append(embedding_text)
            
            page_data_list.append({
                "image_id": image_id,
                "page_num": page_num,
                "description": description,
                "image_type": image_type,
                "thumbnail_b64": thumbnail_b64,
                "full_image_path": full_image_path
            })
        
        # Step 4: Batch generate embeddings (more efficient)
        if embedding_texts:
            print(f"  🔢 Generating {len(embedding_texts)} embeddings in batch...")
            try:
                embeddings = self._get_embeddings_batch(embedding_texts)
                
                # Step 5: Store everything
                for i, page_data in enumerate(page_data_list):
                    try:
                        metadata = ImageMetadata(
                            image_id=page_data["image_id"],
                            document_id=document_id,
                            document_name=document_name,
                            page_number=page_data["page_num"],
                            description=page_data["description"],
                            image_type=page_data["image_type"],
                            thumbnail_base64=page_data["thumbnail_b64"],
                            full_image_path=page_data["full_image_path"],
                            created_at=datetime.utcnow()
                        )
                        
                        # Store in Qdrant
                        self.qdrant.upsert(
                            collection_name=self.COLLECTION_NAME,
                            points=[
                                PointStruct(
                                    id=int(hashlib.md5(page_data["image_id"].encode()).hexdigest()[:8], 16),
                                    vector=embeddings[i],
                                    payload={
                                        "image_id": page_data["image_id"],
                                        "document_id": document_id,
                                        "page_number": page_data["page_num"]
                                    }
                                )
                            ]
                        )
                        
                        # Store in MongoDB
                        self.images_collection.update_one(
                            {"image_id": page_data["image_id"]},
                            {"$set": metadata.to_dict()},
                            upsert=True
                        )
                        
                        results["images_processed"] += 1
                        
                    except Exception as e:
                        error_msg = f"Error storing page {page_data['page_num']}: {str(e)}"
                        results["errors"].append(error_msg)
                        print(f"    ❌ {error_msg}")
                        
            except Exception as e:
                error_msg = f"Error generating embeddings: {str(e)}"
                results["errors"].append(error_msg)
                print(f"  ❌ {error_msg}")
        
        print(f"✅ Document processing complete: {results['images_processed']}/{len(pages)} pages")
        return results
    
    def search_by_text(
        self,
        query: str,
        document_id: Optional[str] = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Search images by text query
        
        Args:
            query: Search query text
            document_id: Optional - limit search to specific document
            limit: Maximum number of results
            
        Returns:
            List of matching images with metadata
        """
        # Generate query embedding
        query_embedding = self._get_query_embedding(query)
        
        # Build filter
        search_filter = None
        if document_id:
            search_filter = qdrant_models.Filter(
                must=[
                    qdrant_models.FieldCondition(
                        key="document_id",
                        match=qdrant_models.MatchValue(value=document_id)
                    )
                ]
            )
        
        # Search in Qdrant
        results = self.qdrant.search(
            collection_name=self.COLLECTION_NAME,
            query_vector=query_embedding,
            query_filter=search_filter,
            limit=limit
        )
        
        # Fetch full metadata from MongoDB
        search_results = []
        for hit in results:
            image_id = hit.payload.get("image_id")
            metadata = self.images_collection.find_one({"image_id": image_id})
            
            if metadata:
                # Remove MongoDB _id and convert to serializable format
                metadata.pop("_id", None)
                search_results.append({
                    "score": hit.score,
                    "image_id": image_id,
                    "document_id": metadata.get("document_id"),
                    "document_name": metadata.get("document_name"),
                    "page_number": metadata.get("page_number"),
                    "description": metadata.get("description"),
                    "image_type": metadata.get("image_type"),
                    "thumbnail": metadata.get("thumbnail_base64")
                })
        
        return search_results
    
    async def search_by_image(
        self,
        image_data: bytes,
        document_id: Optional[str] = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Search similar images using an uploaded image
        
        Args:
            image_data: Image bytes
            document_id: Optional - limit search to specific document
            limit: Maximum number of results
            
        Returns:
            List of matching images with metadata
        """
        # Convert bytes to PIL Image
        image = Image.open(io.BytesIO(image_data))
        
        # Analyze uploaded image with Gemini to get description
        analysis = await self._analyze_single_image_with_gemini(image)
        
        # Create search text from description + image_type only
        description = analysis.get('description', '')
        image_type = analysis.get('drawing_type', 'unknown')
        search_text = f"{description} [Type: {image_type}]"
        
        # Use text search with the image description
        return self.search_by_text(search_text, document_id, limit)
    
    def get_document_images(
        self,
        document_id: str,
        page_number: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Get all images for a document
        
        Args:
            document_id: Document ID
            page_number: Optional - get images from specific page only
            
        Returns:
            List of image metadata
        """
        query = {"document_id": document_id}
        if page_number:
            query["page_number"] = page_number
        
        images = list(self.images_collection.find(query).sort("page_number", 1))
        
        # Clean up for JSON serialization
        for img in images:
            img.pop("_id", None)
        
        return images
    
    def get_image_by_id(self, image_id: str) -> Optional[Dict[str, Any]]:
        """Get single image by ID"""
        image = self.images_collection.find_one({"image_id": image_id})
        if image:
            image.pop("_id", None)
        return image
    
    def delete_document(self, document_id: str) -> Dict[str, Any]:
        """
        Delete all data for a document (including images and chatbots)
        
        Args:
            document_id: Document ID to delete
            
        Returns:
            Deletion results
        """
        # Get all images for this document
        images = list(self.images_collection.find({"document_id": document_id}))
        
        # Delete from Qdrant
        image_ids = [img["image_id"] for img in images]
        point_ids = [int(hashlib.md5(iid.encode()).hexdigest()[:8], 16) for iid in image_ids]
        
        if point_ids:
            self.qdrant.delete(
                collection_name=self.COLLECTION_NAME,
                points_selector=qdrant_models.PointIdsList(points=point_ids)
            )
        
        # Delete image files
        for img in images:
            try:
                path = Path(img.get("full_image_path", ""))
                if path.exists():
                    path.unlink()
            except Exception:
                pass
        
        # Delete from MongoDB
        images_deleted = self.images_collection.delete_many({"document_id": document_id})
        chatbots_deleted = self.chatbots_collection.delete_many({"document_id": document_id})
        self.documents_collection.delete_one({"document_id": document_id})
        
        return {
            "document_id": document_id,
            "images_deleted": images_deleted.deleted_count,
            "chatbots_deleted": chatbots_deleted.deleted_count,
            "success": True
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get service statistics"""
        total_documents = self.documents_collection.count_documents({})
        total_images = self.images_collection.count_documents({})
        total_chatbots = self.chatbots_collection.count_documents({})
        
        # Get image types distribution
        pipeline = [
            {"$group": {"_id": "$image_type", "count": {"$sum": 1}}},
            {"$sort": {"count": -1}}
        ]
        image_types = list(self.images_collection.aggregate(pipeline))
        
        return {
            "total_documents": total_documents,
            "total_images": total_images,
            "total_chatbots": total_chatbots,
            "image_types": {item["_id"]: item["count"] for item in image_types},
            "embedding_provider": self.embedding_config.provider,
            "embedding_model": self.embedding_config.model,
            "embedding_dimension": self.embedding_dim
        }
    
    # ==================== DOCUMENT CRUD ====================
    
    def create_document(self, document: Document) -> Dict[str, Any]:
        """Create a new document"""
        self.documents_collection.update_one(
            {"document_id": document.document_id},
            {"$set": document.to_dict()},
            upsert=True
        )
        return {"success": True, "document_id": document.document_id}
    
    def get_document(self, document_id: str) -> Optional[Dict[str, Any]]:
        """Get document by ID"""
        doc = self.documents_collection.find_one({"document_id": document_id})
        if doc:
            doc.pop("_id", None)
        return doc
    
    def get_document_by_url(self, url: str) -> Optional[Dict[str, Any]]:
        """Get document by URL"""
        doc = self.documents_collection.find_one({"url": url})
        if doc:
            doc.pop("_id", None)
        return doc
    
    def list_documents(self, limit: int = 100, skip: int = 0) -> List[Dict[str, Any]]:
        """List all documents"""
        docs = list(
            self.documents_collection.find()
            .sort("created_at", -1)
            .skip(skip)
            .limit(limit)
        )
        for doc in docs:
            doc.pop("_id", None)
        return docs
    
    def update_document(self, document_id: str, updates: Dict[str, Any]) -> Dict[str, Any]:
        """Update a document"""
        updates["updated_at"] = datetime.utcnow()
        result = self.documents_collection.update_one(
            {"document_id": document_id},
            {"$set": updates}
        )
        return {
            "success": result.modified_count > 0,
            "document_id": document_id
        }
    
    # ==================== CHATBOT CRUD ====================
    
    def create_chatbot(self, chatbot: Chatbot) -> Dict[str, Any]:
        """Create a new chatbot session"""
        self.chatbots_collection.update_one(
            {"chatbot_id": chatbot.chatbot_id},
            {"$set": chatbot.to_dict()},
            upsert=True
        )
        return {"success": True, "chatbot_id": chatbot.chatbot_id}
    
    def get_chatbot(self, chatbot_id: str) -> Optional[Dict[str, Any]]:
        """Get chatbot by ID"""
        chatbot = self.chatbots_collection.find_one({"chatbot_id": chatbot_id})
        if chatbot:
            chatbot.pop("_id", None)
        return chatbot
    
    def get_chatbots_by_document(self, document_id: str) -> List[Dict[str, Any]]:
        """Get all chatbots for a document"""
        chatbots = list(
            self.chatbots_collection.find({"document_id": document_id})
            .sort("created_at", -1)
        )
        for chatbot in chatbots:
            chatbot.pop("_id", None)
        return chatbots
    
    def list_chatbots(self, limit: int = 100, skip: int = 0) -> List[Dict[str, Any]]:
        """List all chatbots"""
        chatbots = list(
            self.chatbots_collection.find()
            .sort("created_at", -1)
            .skip(skip)
            .limit(limit)
        )
        for chatbot in chatbots:
            chatbot.pop("_id", None)
        return chatbots
    
    def update_chatbot(self, chatbot_id: str, updates: Dict[str, Any]) -> Dict[str, Any]:
        """Update a chatbot"""
        updates["updated_at"] = datetime.utcnow()
        result = self.chatbots_collection.update_one(
            {"chatbot_id": chatbot_id},
            {"$set": updates}
        )
        return {
            "success": result.modified_count > 0,
            "chatbot_id": chatbot_id
        }
    
    def add_chat_message(
        self,
        chatbot_id: str,
        question: str,
        answer: str
    ) -> Dict[str, Any]:
        """Add a message to chatbot history"""
        message = {
            "question": question,
            "answer": answer,
            "timestamp": datetime.utcnow().isoformat()
        }
        result = self.chatbots_collection.update_one(
            {"chatbot_id": chatbot_id},
            {
                "$push": {"chat_history": message},
                "$set": {"updated_at": datetime.utcnow()}
            }
        )
        return {
            "success": result.modified_count > 0,
            "chatbot_id": chatbot_id
        }
    
    def clear_chat_history(self, chatbot_id: str) -> Dict[str, Any]:
        """Clear chat history for a chatbot"""
        result = self.chatbots_collection.update_one(
            {"chatbot_id": chatbot_id},
            {
                "$set": {
                    "chat_history": [],
                    "updated_at": datetime.utcnow()
                }
            }
        )
        return {
            "success": result.modified_count > 0,
            "chatbot_id": chatbot_id
        }
    
    def delete_chatbot(self, chatbot_id: str) -> Dict[str, Any]:
        """Delete a chatbot"""
        result = self.chatbots_collection.delete_one({"chatbot_id": chatbot_id})
        return {
            "success": result.deleted_count > 0,
            "chatbot_id": chatbot_id
        }
    
    def delete_chatbots_by_document(self, document_id: str) -> Dict[str, Any]:
        """Delete all chatbots for a document"""
        result = self.chatbots_collection.delete_many({"document_id": document_id})
        return {
            "success": True,
            "document_id": document_id,
            "deleted_count": result.deleted_count
        }


# Singleton instance
_image_search_service: Optional[ImageSearchService] = None


def get_image_search_service() -> ImageSearchService:
    """Get or create the ImageSearchService singleton"""
    global _image_search_service
    
    if _image_search_service is None:
        from dotenv import load_dotenv
        load_dotenv()
        
        # Load embedding config from environment
        embedding_config = EmbeddingConfig.from_env()
        
        # Support both naming conventions for MongoDB
        mongodb_uri = os.getenv("DATABASE_URL") or os.getenv("MONGODB_URI", "mongodb://localhost:27017")
        mongodb_db = os.getenv("DATABASE_NAME") or os.getenv("MONGODB_DB", "cad_assistant")
        
        _image_search_service = ImageSearchService(
            gemini_api_key=os.getenv("GEMINI_API_KEY"),
            embedding_config=embedding_config,
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            cohere_api_key=os.getenv("COHERE_API_KEY"),
            qdrant_url=os.getenv("QDRANT_URL", "localhost"),
            qdrant_port=int(os.getenv("QDRANT_PORT", "6333")),
            mongodb_uri=mongodb_uri,
            mongodb_db=mongodb_db,
            images_dir=os.getenv("IMAGES_DIR", "./extracted_images")
        )
    
    return _image_search_service
