import os
import uuid
from flask import Flask, request, render_template, jsonify, send_from_directory, Response
from utils import ask_groq_stream, perform_web_search, clear_chat_history, execute_mcp_tools, MCP_SERVERS, rag_system, process_uploaded_file
import json
import logging

logger = logging.getLogger(__name__)

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

ALLOWED_EXTENSIONS = {'txt', 'pdf', 'csv', 'json', 'doc', 'docx'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/")
def index():
    session_id = str(uuid.uuid4())
    return render_template("index.html", session_id=session_id)

@app.route("/static/<path:filename>")
def serve_static(filename):
    return send_from_directory('static', filename)

@app.route("/clear-history", methods=["POST"])
def clear_history():
    data = request.json
    session_id = data.get("session_id", "")
    
    if not session_id:
        return jsonify({"error": "Missing session_id"}), 400
    
    try:
        clear_chat_history(session_id)
        return jsonify({"status": "success", "message": "Chat history cleared"})
    except Exception as e:
        logger.error(f"Error clearing chat history: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/mcp/servers", methods=["GET"])
def get_mcp_servers():
    """Get available MCP servers"""
    try:
        return jsonify({"servers": MCP_SERVERS, "status": "success"})
    except Exception as e:
        logger.error(f"Error getting MCP servers: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/rag/upload", methods=["POST"])
def rag_upload():
    """Upload documents to RAG system"""
    try:
        # Check if a file was uploaded
        if 'file' not in request.files:
            return jsonify({"error": "No file provided"}), 400
        
        file = request.files['file']
        
        # Check if file is selected
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400
        
        # Check if file type is allowed
        if file and allowed_file(file.filename):
            # Save the file
            filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filename)
            
            # Process the file
            text_content = process_uploaded_file(filename)
            
            if not text_content:
                return jsonify({"error": "Could not extract text from file"}), 400
            
            # Initialize RAG system with the text
            rag_system.create_vectorstore([text_content], file.filename)
            rag_system.create_qa_chain()
            
            return jsonify({
                "status": "success", 
                "message": "Document processed by RAG system",
                "filename": file.filename
            })
        else:
            return jsonify({"error": "File type not allowed"}), 400
            
    except Exception as e:
        logger.error(f"Error uploading file to RAG: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/rag/url", methods=["POST"])
def rag_url():
    """Process URL for RAG system"""
    try:
        data = request.json
        url = data.get("url", "")
        
        if not url:
            return jsonify({"error": "No URL provided"}), 400
        
        # Process the URL
        text_content = process_uploaded_file(url, is_url=True)
        
        if not text_content:
            return jsonify({"error": "Could not extract text from URL"}), 400
        
        # Initialize RAG system with the text
        rag_system.create_vectorstore([text_content], url)
        rag_system.create_qa_chain()
        
        return jsonify({
            "status": "success", 
            "message": "URL content processed by RAG system",
            "url": url
        })
            
    except Exception as e:
        logger.error(f"Error processing URL for RAG: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/rag/query", methods=["POST"])
def rag_query():
    """Query the RAG system directly"""
    try:
        data = request.json
        question = data.get("question", "")
        
        if not question:
            return jsonify({"error": "No question provided"}), 400
        
        result = rag_system.query_rag(question)
        
        return jsonify({
            "status": "success", 
            "result": result["result"],
            "sources": [doc.metadata.get("source", "Unknown") for doc in result.get("source_documents", [])]
        })
    except Exception as e:
        logger.error(f"Error querying RAG: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/ask-stream", methods=["GET", "POST"])
def ask_stream():
    if request.method == "GET":
        # Handle GET request (for EventSource)
        question = request.args.get('question', '')
        session_id = request.args.get('session_id', '')
        use_web_search = request.args.get('web_search', 'true').lower() == 'true'
        context = request.args.get('context', '')
    else:
        # Handle POST request
        data = request.json
        question = data.get("question", "")
        session_id = data.get("session_id", "")
        use_web_search = data.get("web_search", True)
        context = data.get("context", "")
    
    if not question:
        return jsonify({"error": "Missing question"}), 400
    
    def generate():
        try:
            # Perform web search only if web search is enabled
            web_results = []
            if use_web_search:
                web_results = perform_web_search(question)
            
            # Stream response from Groq with MCP and RAG integration
            for chunk in ask_groq_stream(session_id, question, web_results, use_web_search, context):
                yield f"data: {json.dumps(chunk)}\n\n"
        except Exception as e:
            logger.error(f"Error in ask-stream: {str(e)}")
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
    
    return Response(generate(), mimetype='text/event-stream')

if __name__ == "__main__":
    # Create necessary directories if they don't exist
    os.makedirs("chroma_db", exist_ok=True)
    os.makedirs("uploads", exist_ok=True)
    app.run(debug=True)