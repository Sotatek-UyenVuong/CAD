from flask import Flask, request, jsonify, send_from_directory, send_file
import uuid
from flask_cors import CORS
import google.generativeai as genai
import os
import re
import asyncio
from dotenv import load_dotenv
import tempfile
import json
from pathlib import Path
from functools import wraps
from datetime import datetime, timedelta
import bcrypt
import jwt

# Get absolute path of this script's directory
BASE_DIR = Path(__file__).resolve().parent

# Load .env from project directory
load_dotenv(dotenv_path=BASE_DIR / '.env')
api_key = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=api_key)

app = Flask(__name__, static_folder=str(BASE_DIR))
CORS(app)

# JWT Secret Key
JWT_SECRET = os.getenv("JWT_SECRET", "your-secret-key-change-in-production")
JWT_EXPIRATION_HOURS = 24

# Store active chat sessions (for Gemini chat)
chat_sessions = {}

# MongoDB connection for auth
_mongo_client = None
_auth_db = None

def get_auth_db():
    """Get MongoDB database for authentication"""
    global _mongo_client, _auth_db
    if _auth_db is None:
        try:
            from pymongo import MongoClient
            mongodb_uri = os.getenv("DATABASE_URL") or os.getenv("MONGODB_URI", "mongodb://localhost:27017")
            mongodb_db = os.getenv("DATABASE_NAME") or os.getenv("MONGODB_DB", "cad_assistant")
            _mongo_client = MongoClient(mongodb_uri)
            _auth_db = _mongo_client[mongodb_db]
            
            # Create indexes
            _auth_db.users.create_index("email", unique=True)
            print("✅ Auth database connected")
        except Exception as e:
            print(f"⚠️ Auth database connection failed: {e}")
            return None
    return _auth_db


def generate_token(user_id: str, email: str) -> str:
    """Generate JWT token for user"""
    payload = {
        "user_id": user_id,
        "email": email,
        "exp": datetime.utcnow() + timedelta(hours=JWT_EXPIRATION_HOURS),
        "iat": datetime.utcnow()
    }
    return jwt.encode(payload, JWT_SECRET, algorithm="HS256")


def verify_token(token: str) -> dict:
    """Verify JWT token and return payload"""
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=["HS256"])
        return payload
    except jwt.ExpiredSignatureError:
        return None
    except jwt.InvalidTokenError:
        return None


def auth_required(f):
    """Decorator to require authentication"""
    @wraps(f)
    def decorated(*args, **kwargs):
        token = None
        
        # Get token from header
        if 'Authorization' in request.headers:
            auth_header = request.headers['Authorization']
            if auth_header.startswith('Bearer '):
                token = auth_header.split(' ')[1]
        
        if not token:
            return jsonify({"success": False, "error": "Token required"}), 401
        
        payload = verify_token(token)
        if not payload:
            return jsonify({"success": False, "error": "Invalid or expired token"}), 401
        
        # Add user info to request
        request.user_id = payload.get("user_id")
        request.user_email = payload.get("email")
        
        return f(*args, **kwargs)
    return decorated

# Image search service (lazy loaded)
_image_search_service = None


def get_image_search():
    """Lazy load image search service"""
    global _image_search_service
    if _image_search_service is None:
        try:
            from image_search_service import ImageSearchService, EmbeddingConfig
            
            # Load embedding config from environment
            embedding_config = EmbeddingConfig.from_env()
            
            # Support both naming conventions for MongoDB
            mongodb_uri = os.getenv("DATABASE_URL") or os.getenv("MONGODB_URI", "mongodb://localhost:27017")
            mongodb_db = os.getenv("DATABASE_NAME") or os.getenv("MONGODB_DB", "cad_assistant")
            
            _image_search_service = ImageSearchService(
                gemini_api_key=api_key,
                embedding_config=embedding_config,
                openai_api_key=os.getenv("OPENAI_API_KEY"),
                cohere_api_key=os.getenv("COHERE_API_KEY"),
                qdrant_url=os.getenv("QDRANT_URL", "localhost"),
                qdrant_port=int(os.getenv("QDRANT_PORT", "6333")),
                mongodb_uri=mongodb_uri,
                mongodb_db=mongodb_db,
                images_dir=str(BASE_DIR / "extracted_images")
            )
        except Exception as e:
            print(f"⚠️ Image search service not available: {e}")
            return None
    return _image_search_service


def async_route(f):
    """Decorator to run async functions in Flask routes"""
    @wraps(f)
    def wrapper(*args, **kwargs):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(f(*args, **kwargs))
        finally:
            loop.close()
    return wrapper


class ChatSession:
    # Maximum number of Q&A pairs to keep in history (to save tokens)
    MAX_HISTORY = 10
    
    SYSTEM_INSTRUCTION = """You are an expert in architectural CAD drawings and construction documents.

Your expertise includes:
- Analyzing architectural floor plans, elevations, sections, and technical drawings
- Japanese architectural standards and notation (図面番号, 図面名称, etc.)
- Extracting information about rooms, dimensions, symbols, and equipment
- Counting symbols: doors, windows, lighting fixtures, electrical equipment, fire safety
- Providing room dimensions and spatial analysis
- Explaining technical annotations and specifications

When answering:
- Be specific and cite page numbers using ONLY this exact format: [page X] where X is the page number
  Example: "The entrance hall [page 2] has a total area of 45.6 m²"
  Example: "Fire alarm system [page 11][page 12] includes 24 detectors"
- Provide exact counts when counting symbols or equipment
- Include dimensions with proper units (mm, m, m²)
- Use both Japanese and English terms when appropriate
- Structure answers clearly with bullet points using • character
- If information is not clearly visible, state your confidence level
- Format lists with bullet points (•) for better readability

IMPORTANT CITATION RULES:
✓ Correct: [page 2], [page 11][page 12], [page 5][page 8]
✗ Wrong: [pages 11-12], [pages 2-8], [pages 2,8], (page 2), page 2, on page 2

ALWAYS use separate [page X] tags for each page - NEVER use ranges or comma-separated lists!
Each page must have its own individual [page X] citation so users can click to navigate!"""

    def __init__(self, file_path, filename):
        self.file_path = file_path
        self.filename = filename
        self.uploaded_file = None
        self.chat = None
        self.chat_history = []
        self.document_id = None
        self.created_at = datetime.now().isoformat()
        self.pdf_path = file_path  # Store PDF path for page rendering
        self.setup()
    
    def setup(self):
        """Initialize the chat session"""
        # Upload PDF to Gemini
        self.uploaded_file = genai.upload_file(self.file_path)
        self._create_chat_session()
    
    def _create_chat_session(self, history=None):
        """Create or rebuild chat session with optional history"""
        model = genai.GenerativeModel(
            "models/gemini-2.5-pro",
            system_instruction=self.SYSTEM_INSTRUCTION
        )
        
        # Convert history to Gemini format if provided
        gemini_history = []
        if history:
            for item in history:
                gemini_history.append({
                    "role": "user",
                    "parts": [item["question"]]
                })
                gemini_history.append({
                    "role": "model", 
                    "parts": [item["answer"]]
                })
        
        self.chat = model.start_chat(history=gemini_history)
    
    def _trim_history_if_needed(self):
        """Trim history to MAX_HISTORY and rebuild chat session"""
        if len(self.chat_history) > self.MAX_HISTORY:
            print(f"📝 Trimming chat history from {len(self.chat_history)} to {self.MAX_HISTORY} messages")
            # Keep only the most recent MAX_HISTORY messages
            self.chat_history = self.chat_history[-self.MAX_HISTORY:]
            # Rebuild chat session with trimmed history
            self._create_chat_session(self.chat_history)
    
    def send_message(self, message):
        """Send message and get response"""
        try:
            # Trim history if it exceeds MAX_HISTORY
            self._trim_history_if_needed()
            
            # First message includes the PDF
            if len(self.chat_history) == 0:
                response = self.chat.send_message([self.uploaded_file, message])
            else:
                response = self.chat.send_message(message)
            
            answer = response.text
            
            # Save to history
            self.chat_history.append({
                "question": message,
                "answer": answer
            })
            
            return {
                "success": True,
                "answer": answer,
                "citations": self.extract_citations(answer)
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def extract_citations(self, text):
        """Extract page numbers and drawing references from text"""
        citations = []
        
        # Extract [page X] citations
        page_matches = re.finditer(r'\[page (\d+)\]', text, re.IGNORECASE)
        for match in page_matches:
            page_num = int(match.group(1))
            if page_num not in citations:
                citations.append(page_num)
        
        return sorted(citations)


# ==================== BASIC ROUTES ====================

@app.route('/')
def index():
    """Serve the main HTML file"""
    return send_from_directory(str(BASE_DIR), 'chat_interface.html')


@app.route('/chat_interface.js')
def serve_js():
    """Serve the JavaScript file"""
    return send_from_directory(str(BASE_DIR), 'chat_interface.js')


# ==================== AUTH ROUTES ====================

@app.route('/api/auth/register', methods=['POST'])
def register():
    """Register a new user"""
    try:
        data = request.get_json()
        
        email = data.get('email', '').strip().lower()
        password = data.get('password', '')
        name = data.get('name', '').strip()
        
        # Validation
        if not email or not password:
            return jsonify({"success": False, "error": "Email and password required"}), 400
        
        if len(password) < 6:
            return jsonify({"success": False, "error": "Password must be at least 6 characters"}), 400
        
        if not name:
            name = email.split('@')[0]
        
        db = get_auth_db()
        if db is None:
            return jsonify({"success": False, "error": "Database not available"}), 503
        
        # Check if user exists
        if db.users.find_one({"email": email}):
            return jsonify({"success": False, "error": "Email already registered"}), 409
        
        # Hash password (decode to string for storage)
        password_hash = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
        
        # Create user
        user_id = str(uuid.uuid4())
        user = {
            "_id": user_id,
            "email": email,
            "name": name,
            "password_hash": password_hash,
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat()
        }
        
        db.users.insert_one(user)
        
        # Generate token
        token = generate_token(user_id, email)
        
        print(f"✅ User registered: {email}")
        
        return jsonify({
            "success": True,
            "token": token,
            "user": {
                "id": user_id,
                "email": email,
                "name": name,
                "created_at": user["created_at"]
            }
        })
        
    except Exception as e:
        import traceback
        print(f"Registration error: {e}")
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/auth/login', methods=['POST'])
def login():
    """Login user"""
    try:
        data = request.get_json()
        
        email = data.get('email', '').strip().lower()
        password = data.get('password', '')
        
        if not email or not password:
            return jsonify({"success": False, "error": "Email and password required"}), 400
        
        db = get_auth_db()
        if db is None:
            return jsonify({"success": False, "error": "Database not available"}), 503
        
        # Find user
        user = db.users.find_one({"email": email})
        if not user:
            return jsonify({"success": False, "error": "Invalid email or password"}), 401
        
        # Check password (handle both string and bytes stored hash)
        stored_hash = user['password_hash']
        if isinstance(stored_hash, str):
            stored_hash = stored_hash.encode('utf-8')
        
        if not bcrypt.checkpw(password.encode('utf-8'), stored_hash):
            return jsonify({"success": False, "error": "Invalid email or password"}), 401
        
        # Generate token
        token = generate_token(user['_id'], email)
        
        print(f"✅ User logged in: {email}")
        
        return jsonify({
            "success": True,
            "token": token,
            "user": {
                "id": user['_id'],
                "email": user['email'],
                "name": user['name'],
                "created_at": user['created_at']
            }
        })
        
    except Exception as e:
        import traceback
        print(f"Login error: {e}")
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/auth/me', methods=['GET'])
@auth_required
def get_current_user():
    """Get current user info"""
    try:
        db = get_auth_db()
        if db is None:
            return jsonify({"success": False, "error": "Database not available"}), 503
        
        user = db.users.find_one({"_id": request.user_id})
        if not user:
            return jsonify({"success": False, "error": "User not found"}), 404
        
        return jsonify({
            "success": True,
            "user": {
                "id": user['_id'],
                "email": user['email'],
                "name": user['name'],
                "created_at": user['created_at']
            }
        })
        
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/auth/logout', methods=['POST'])
@auth_required
def logout():
    """Logout user (client should delete token)"""
    return jsonify({"success": True, "message": "Logged out successfully"})


# ==================== CHAT ROUTES ====================

@app.route('/api/upload', methods=['POST'])
@async_route
async def upload_file():
    """
    Handle PDF upload and process images for search
    
    Flow:
    1. Save PDF file
    2. Process all pages with Gemini (1 API call)
    3. Generate embeddings and store
    4. Return success when complete
    """
    try:
        if 'file' not in request.files:
            return jsonify({"success": False, "error": "No file provided"}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({"success": False, "error": "No file selected"}), 400
        
        if not file.filename.endswith('.pdf'):
            return jsonify({"success": False, "error": "File must be a PDF"}), 400
        
        # Save file
        import hashlib
        temp_dir = tempfile.gettempdir()
        temp_path = os.path.join(temp_dir, file.filename)
        file.save(temp_path)
        
        # Generate document_id
        document_id = hashlib.md5(f"{file.filename}_{datetime.utcnow().isoformat()}".encode()).hexdigest()
        
        print(f"📤 Processing upload: {file.filename}")
        
        # Process images for search (wait until complete)
        image_search = get_image_search()
        processing_result = None
        
        if image_search:
            try:
                processing_result = await image_search.process_document(
                    pdf_path=temp_path,
                    document_name=file.filename,
                    document_id=document_id
                )
                print(f"✅ Image processing complete: {processing_result['images_processed']} pages indexed")
            except Exception as e:
                print(f"⚠️ Image processing failed: {e}")
                return jsonify({
                    "success": False,
                    "error": f"Failed to process document: {str(e)}"
                }), 500
        else:
            return jsonify({
                "success": False,
                "error": "Image search service not available. Check Qdrant and MongoDB."
            }), 503
        
        # Create chat session for the uploaded PDF
        session_id = str(uuid.uuid4())
        try:
            chat_sessions[session_id] = ChatSession(temp_path, file.filename)
            chat_sessions[session_id].document_id = document_id
        except Exception as e:
            print(f"⚠️ Chat session creation failed: {e}")
            # Continue without chat session - image search will still work
        
        return jsonify({
            "success": True,
            "session_id": session_id,
            "document_id": document_id,
            "file_name": file.filename,
            "total_pages": processing_result.get("total_pages", 0) if processing_result else 0,
            "images_processed": processing_result.get("images_processed", 0) if processing_result else 0,
            "message": "Document uploaded and indexed successfully!"
        })
        
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/chat', methods=['POST'])
def chat():
    """Handle chat messages"""
    try:
        data = request.json
        session_id = data.get('session_id')
        message = data.get('message')
        
        if not session_id or session_id not in chat_sessions:
            return jsonify({"success": False, "error": "Invalid session"}), 400
        
        if not message:
            return jsonify({"success": False, "error": "No message provided"}), 400
        
        # Get chat session and send message
        session = chat_sessions[session_id]
        result = session.send_message(message)
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/history/<session_id>', methods=['GET'])
def get_history(session_id):
    """Get chat history for a session"""
    try:
        if session_id not in chat_sessions:
            return jsonify({"success": False, "error": "Invalid session"}), 400
        
        session = chat_sessions[session_id]
        
        return jsonify({
            "success": True,
            "history": session.chat_history,
            "filename": session.filename
        })
        
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/clear/<session_id>', methods=['POST'])
def clear_history(session_id):
    """Clear chat history for a session"""
    try:
        if session_id not in chat_sessions:
            return jsonify({"success": False, "error": "Invalid session"}), 400
        
        session = chat_sessions[session_id]
        
        # Clear history and recreate chat session
        session.chat_history = []
        session._create_chat_session()
        
        return jsonify({
            "success": True,
            "message": "Chat history cleared"
        })
        
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/save/<session_id>', methods=['POST'])
def save_history(session_id):
    """Save chat history to JSON file"""
    try:
        if session_id not in chat_sessions:
            return jsonify({"success": False, "error": "Invalid session"}), 400
        
        session = chat_sessions[session_id]
        
        if not session.chat_history:
            return jsonify({"success": False, "error": "No history to save"}), 400
        
        # Create filename with timestamp
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"chat_history_{timestamp}.json"
        
        # Save to project directory (absolute path)
        filepath = BASE_DIR / filename
        
        # Save to file
        history_data = {
            "filename": session.filename,
            "timestamp": timestamp,
            "total_messages": len(session.chat_history),
            "chat_history": session.chat_history
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(history_data, f, ensure_ascii=False, indent=2)
        
        return jsonify({
            "success": True,
            "filename": filename,
            "message": "History saved successfully"
        })
        
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/sessions', methods=['GET'])
def list_sessions():
    """List all active sessions"""
    sessions = []
    for session_id, session in chat_sessions.items():
        sessions.append({
            "session_id": session_id,
            "file_name": session.filename,  # Match frontend expectation
            "message_count": len(session.chat_history),
            "document_id": getattr(session, 'document_id', None),
            "created_at": getattr(session, 'created_at', None) or datetime.now().isoformat()
        })
    
    return jsonify({
        "success": True,
        "sessions": sessions
    })


@app.route('/api/sessions/<session_id>', methods=['DELETE'])
def delete_session(session_id):
    """Delete a session and its associated data"""
    global chat_sessions
    
    try:
        # Get document_id from session before deleting
        document_id = None
        if session_id in chat_sessions:
            document_id = getattr(chat_sessions[session_id], 'document_id', None)
            del chat_sessions[session_id]
        
        # Try to delete from image search database as well
        try:
            image_search = get_image_search()
            if image_search and document_id:
                # Delete document and all associated images from Qdrant/MongoDB
                image_search.delete_document(document_id)
        except Exception as e:
            print(f"Warning: Could not delete document data for session {session_id}: {e}")
        
        print(f"✅ Session deleted: {session_id}")
        
        return jsonify({
            "success": True,
            "message": f"Session {session_id} deleted"
        })
        
    except Exception as e:
        print(f"Delete session error: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


# ==================== DOCUMENT ROUTES ====================

@app.route('/api/document-info/<session_id>', methods=['GET'])
def get_document_info(session_id):
    """Get document information including total pages"""
    global chat_sessions
    
    if session_id not in chat_sessions:
        return jsonify({
            "success": False,
            "error": "Session not found"
        }), 404
    
    session = chat_sessions[session_id]
    
    # Count PDF pages
    total_pages = 0
    if session.pdf_path and os.path.exists(session.pdf_path):
        try:
            import fitz  # PyMuPDF
            doc = fitz.open(session.pdf_path)
            total_pages = doc.page_count
            doc.close()
        except Exception as e:
            print(f"Error counting pages: {e}")
            total_pages = 0
    
    return jsonify({
        "success": True,
        "session_id": session_id,
        "file_name": session.filename,
        "total_pages": total_pages
    })


@app.route('/api/page-image/<session_id>/<int:page_number>', methods=['GET'])
def get_page_image(session_id, page_number):
    """Get a specific page as an image"""
    global chat_sessions
    
    if session_id not in chat_sessions:
        return jsonify({"success": False, "error": "Session not found"}), 404
    
    session = chat_sessions[session_id]
    
    if not session.pdf_path or not os.path.exists(session.pdf_path):
        return jsonify({"success": False, "error": "PDF file not found"}), 404
    
    try:
        import fitz  # PyMuPDF
        from io import BytesIO
        
        doc = fitz.open(session.pdf_path)
        
        # Page numbers are 1-indexed from frontend
        page_idx = page_number - 1
        if page_idx < 0 or page_idx >= doc.page_count:
            doc.close()
            return jsonify({"success": False, "error": "Page not found"}), 404
        
        page = doc[page_idx]
        
        # Render page to image (2x resolution for clarity)
        mat = fitz.Matrix(2, 2)
        pix = page.get_pixmap(matrix=mat)
        
        # Convert to PNG
        img_bytes = pix.tobytes("png")
        doc.close()
        
        return send_file(
            BytesIO(img_bytes),
            mimetype='image/png',
            as_attachment=False
        )
        
    except Exception as e:
        print(f"Error rendering page: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/download/<session_id>', methods=['GET'])
def download_document(session_id):
    """Download the original PDF document"""
    global chat_sessions
    
    if session_id not in chat_sessions:
        return jsonify({"success": False, "error": "Session not found"}), 404
    
    session = chat_sessions[session_id]
    
    if not session.pdf_path or not os.path.exists(session.pdf_path):
        return jsonify({"success": False, "error": "PDF file not found"}), 404
    
    return send_file(
        session.pdf_path,
        mimetype='application/pdf',
        as_attachment=True,
        download_name=session.filename
    )


# ==================== IMAGE SEARCH ROUTES ====================

@app.route('/api/image-search/text', methods=['POST'])
def search_images_by_text():
    """
    Search images by text query
    
    Request body:
    {
        "query": "electrical outlets near door",
        "document_id": "optional - limit to specific document",
        "limit": 10
    }
    """
    try:
        image_search = get_image_search()
        if not image_search:
            return jsonify({
                "success": False,
                "error": "Image search service not available. Please ensure Qdrant and MongoDB are running."
            }), 503
        
        data = request.json
        query = data.get('query')
        
        if not query:
            return jsonify({"success": False, "error": "No query provided"}), 400
        
        document_id = data.get('document_id')
        limit = data.get('limit', 10)
        
        results = image_search.search_by_text(query, document_id, limit)
        
        return jsonify({
            "success": True,
            "query": query,
            "results": results,
            "count": len(results)
        })
        
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/image-search/image', methods=['POST'])
@async_route
async def search_images_by_image():
    """
    Search similar images using an uploaded image
    
    Supports both:
    - multipart/form-data with 'image' file
    - JSON with 'image' as base64 string
    Optional: 'document_id' to limit search, 'limit' for max results
    """
    try:
        image_search = get_image_search()
        if not image_search:
            return jsonify({
                "success": False,
                "error": "Image search service not available. Please ensure Qdrant and MongoDB are running."
            }), 503
        
        image_data = None
        document_id = None
        limit = 10
        
        # Check if JSON request with base64 image
        if request.is_json:
            data = request.json
            base64_image = data.get('image')
            if not base64_image:
                return jsonify({"success": False, "error": "No image provided"}), 400
            
            # Decode base64 image
            import base64
            try:
                image_data = base64.b64decode(base64_image)
            except Exception as e:
                return jsonify({"success": False, "error": f"Invalid base64 image: {str(e)}"}), 400
            
            document_id = data.get('document_id')
            limit = int(data.get('limit', 10))
        
        # Check if multipart form data
        elif 'image' in request.files:
            image_file = request.files['image']
            image_data = image_file.read()
            document_id = request.form.get('document_id')
            limit = int(request.form.get('limit', 10))
        
        else:
            return jsonify({"success": False, "error": "No image provided"}), 400
        
        results = await image_search.search_by_image(image_data, document_id, limit)
        
        return jsonify({
            "success": True,
            "results": results,
            "total_results": len(results)
        })
        
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/image-search/document/<document_id>', methods=['GET'])
def get_document_images(document_id):
    """
    Get all images for a specific document
    
    Query params:
    - page: Optional page number filter
    """
    try:
        image_search = get_image_search()
        if not image_search:
            return jsonify({
                "success": False,
                "error": "Image search service not available"
            }), 503
        
        page_number = request.args.get('page', type=int)
        
        images = image_search.get_document_images(document_id, page_number)
        
        return jsonify({
            "success": True,
            "document_id": document_id,
            "images": images,
            "count": len(images)
        })
        
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/image-search/image/<image_id>', methods=['GET'])
def get_image_details(image_id):
    """Get detailed information about a specific image"""
    try:
        image_search = get_image_search()
        if not image_search:
            return jsonify({
                "success": False,
                "error": "Image search service not available"
            }), 503
        
        image = image_search.get_image_by_id(image_id)
        
        if not image:
            return jsonify({"success": False, "error": "Image not found"}), 404
        
        return jsonify({
            "success": True,
            "image": image
        })
        
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/image-search/stats', methods=['GET'])
def get_image_search_stats():
    """Get image search service statistics"""
    try:
        image_search = get_image_search()
        if not image_search:
            return jsonify({
                "success": False,
                "error": "Image search service not available"
            }), 503
        
        stats = image_search.get_statistics()
        
        return jsonify({
            "success": True,
            "statistics": stats
        })
        
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/image-search/document/<document_id>', methods=['DELETE'])
def delete_document_images(document_id):
    """Delete all images for a document"""
    try:
        image_search = get_image_search()
        if not image_search:
            return jsonify({
                "success": False,
                "error": "Image search service not available"
            }), 503
        
        result = image_search.delete_document(document_id)
        
        return jsonify({
            "success": True,
            "result": result
        })
        
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# ==================== CHATBOT ROUTES ====================

@app.route('/api/chatbots', methods=['GET'])
@auth_required
def list_chatbots():
    """List all chatbots for the current user"""
    try:
        image_search = get_image_search()
        if not image_search:
            return jsonify({
                "success": False,
                "error": "Service not available"
            }), 503
        
        # Get chatbots for this user
        chatbots = list(
            image_search.chatbots_collection.find({"user_id": request.user_id})
            .sort("created_at", -1)
        )
        
        for chatbot in chatbots:
            chatbot.pop("_id", None)
        
        return jsonify({
            "success": True,
            "chatbots": chatbots
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/chatbots', methods=['POST'])
@auth_required
def create_chatbot():
    """Create a new chatbot"""
    try:
        data = request.get_json()
        
        chatbot_id = data.get('chatbot_id') or str(uuid.uuid4())
        document_id = data.get('document_id')
        document_name = data.get('document_name', '')
        session_id = data.get('session_id', document_id)
        
        if not document_id:
            return jsonify({"success": False, "error": "document_id required"}), 400
        
        image_search = get_image_search()
        if not image_search:
            return jsonify({
                "success": False,
                "error": "Service not available"
            }), 503
        
        # Check if chatbot already exists for this user and document
        existing = image_search.chatbots_collection.find_one({
            "user_id": request.user_id,
            "document_id": document_id
        })
        
        if existing:
            existing.pop("_id", None)
            return jsonify({
                "success": True,
                "chatbot": existing,
                "message": "Chatbot already exists"
            })
        
        chatbot_data = {
            "chatbot_id": chatbot_id,
            "user_id": request.user_id,
            "document_id": document_id,
            "document_name": document_name,
            "session_id": session_id,
            "messages": [],
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat()
        }
        
        image_search.chatbots_collection.insert_one(chatbot_data)
        chatbot_data.pop("_id", None)
        
        print(f"✅ Chatbot created: {chatbot_id} for user {request.user_id}")
        
        return jsonify({
            "success": True,
            "chatbot": chatbot_data
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/chatbots/<chatbot_id>', methods=['GET'])
@auth_required
def get_chatbot(chatbot_id):
    """Get a specific chatbot"""
    try:
        image_search = get_image_search()
        if not image_search:
            return jsonify({
                "success": False,
                "error": "Service not available"
            }), 503
        
        chatbot = image_search.chatbots_collection.find_one({
            "chatbot_id": chatbot_id,
            "user_id": request.user_id
        })
        
        if not chatbot:
            return jsonify({"success": False, "error": "Chatbot not found"}), 404
        
        chatbot.pop("_id", None)
        
        return jsonify({
            "success": True,
            "chatbot": chatbot
        })
        
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/chatbots/<chatbot_id>/messages', methods=['POST'])
@auth_required
def add_chatbot_message(chatbot_id):
    """Add a message to chatbot history"""
    try:
        data = request.get_json()
        message = data.get('message')
        
        if not message:
            return jsonify({"success": False, "error": "message required"}), 400
        
        image_search = get_image_search()
        if not image_search:
            return jsonify({
                "success": False,
                "error": "Service not available"
            }), 503
        
        result = image_search.chatbots_collection.update_one(
            {
                "chatbot_id": chatbot_id,
                "user_id": request.user_id
            },
            {
                "$push": {"messages": message},
                "$set": {"updated_at": datetime.utcnow().isoformat()}
            }
        )
        
        if result.matched_count == 0:
            return jsonify({"success": False, "error": "Chatbot not found"}), 404
        
        return jsonify({
            "success": True,
            "message": "Message added"
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/chatbots/<chatbot_id>/clear', methods=['POST'])
@auth_required
def clear_chatbot_messages(chatbot_id):
    """Clear all messages in a chatbot"""
    try:
        image_search = get_image_search()
        if not image_search:
            return jsonify({
                "success": False,
                "error": "Service not available"
            }), 503
        
        result = image_search.chatbots_collection.update_one(
            {
                "chatbot_id": chatbot_id,
                "user_id": request.user_id
            },
            {
                "$set": {
                    "messages": [],
                    "updated_at": datetime.utcnow().isoformat()
                }
            }
        )
        
        if result.matched_count == 0:
            return jsonify({"success": False, "error": "Chatbot not found"}), 404
        
        print(f"✅ Chatbot messages cleared: {chatbot_id}")
        
        return jsonify({
            "success": True,
            "message": "Messages cleared"
        })
        
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/chatbots/<chatbot_id>', methods=['DELETE'])
@auth_required
def delete_chatbot(chatbot_id):
    """Delete a chatbot"""
    try:
        image_search = get_image_search()
        if not image_search:
            return jsonify({
                "success": False,
                "error": "Service not available"
            }), 503
        
        result = image_search.chatbots_collection.delete_one({
            "chatbot_id": chatbot_id,
            "user_id": request.user_id
        })
        
        if result.deleted_count == 0:
            return jsonify({"success": False, "error": "Chatbot not found"}), 404
        
        print(f"✅ Chatbot deleted: {chatbot_id}")
        
        return jsonify({
            "success": True,
            "message": "Chatbot deleted"
        })
        
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# ==================== STATIC FILES ====================

@app.route('/extracted_images/<path:filename>')
def serve_extracted_image(filename):
    """Serve extracted images"""
    images_dir = BASE_DIR / "extracted_images"
    return send_from_directory(str(images_dir), filename)


if __name__ == '__main__':
    print("="*70)
    print("🏗️  CAD Document Chat Assistant - Web Interface")
    print("="*70)
    print("📡 Server starting on http://localhost:5006")
    print("📄 Open your browser and navigate to http://localhost:5006")
    print("")
    print("📊 Image Search Features:")
    print("   • POST /api/image-search/text  - Search by text")
    print("   • POST /api/image-search/image - Search by image")
    print("   • GET  /api/image-search/stats - View statistics")
    print("="*70)
    app.run(debug=True, host='0.0.0.0', port=5006)
