import os
import uuid
from flask import Flask, request, render_template, jsonify
from utils import add_to_vector_db, search_documents, ask_groq
from werkzeug.utils import secure_filename

UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {"pdf", "txt", "csv"}

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/")
def index():
    session_id = str(uuid.uuid4())
    return render_template("index.html", session_id=session_id)

@app.route("/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return jsonify({"error": "No file"}), 400
    file = request.files["file"]
    if file.filename == "" or not allowed_file(file.filename):
        return jsonify({"error": "Invalid file"}), 400
    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(file_path)
    add_to_vector_db(file_path)
    return jsonify({"message": "File uploaded and processed successfully"})

@app.route("/ask", methods=["POST"])
def ask():
    data = request.json
    question = data.get("question", "")
    session_id = data.get("session_id", "")
    if not question or not session_id:
        return jsonify({"error": "Missing question or session_id"}), 400
    docs = search_documents(question)
    context = "\n".join(docs)
    answer = ask_groq(session_id, context, question)
    return jsonify({"answer": answer})

if __name__ == "__main__":
    app.run(debug=True)
