"""
Sticky2Jira - Main Flask Application
Browser-based local application for extracting text from sticky notes and creating Jira issues.
"""

import os
import webbrowser
import logging
from logging.handlers import RotatingFileHandler
from threading import Timer

from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit
from flask_cors import CORS
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
app.config["SECRET_KEY"] = os.urandom(24)
app.config["UPLOAD_FOLDER"] = "uploads"
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024  # 5MB max upload size

# Enable CORS
CORS(app)

# Initialize SocketIO
socketio = SocketIO(app, cors_allowed_origins="*")

# Configure logging
app.logger.setLevel(logging.INFO)

# File handler - app.log (5MB max, 3 backups)
file_handler = RotatingFileHandler("app.log", maxBytes=5 * 1024 * 1024, backupCount=3)
file_handler.setFormatter(
    logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
)
app.logger.addHandler(file_handler)

# Error file handler - errors.log
error_handler = RotatingFileHandler(
    "errors.log", maxBytes=5 * 1024 * 1024, backupCount=3
)
error_handler.setLevel(logging.ERROR)
error_handler.setFormatter(
    logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s\n%(pathname)s:%(lineno)d\n%(exc_info)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
)
app.logger.addHandler(error_handler)

# Console handler (terminal output)
console_handler = logging.StreamHandler()
console_handler.setFormatter(
    logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
)
app.logger.addHandler(console_handler)

app.logger.info("Sticky2Jira application starting...")


# ============================================================================
# Routes - Main Application
# ============================================================================


@app.route("/")
def index():
    """Main application page."""
    app.logger.info("Serving index page")
    return render_template("index.html")


# ============================================================================
# API Endpoints - Jira Configuration
# ============================================================================


@app.route("/api/jira/test-connection", methods=["POST"])
def test_jira_connection():
    """Test Jira connection with provided credentials."""
    try:
        data = request.json
        if not data:
            return jsonify({"success": False, "error": "No data provided"}), 400

        app.logger.info(f"Testing Jira connection to {data.get('server_url')}")

        # TODO: Implement Jira connection test using jira_service

        return jsonify(
            {
                "success": True,
                "message": "Connection successful",
                "server_title": data.get("server_url", "Jira Server"),
            }
        )
    except Exception as e:
        app.logger.error(f"Jira connection test failed: {str(e)}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/jira/settings", methods=["GET"])
def get_jira_settings():
    """Retrieve saved Jira settings."""
    try:
        app.logger.info("Retrieving Jira settings")

        from services.session_manager import get_jira_settings as load_settings

        settings = load_settings()

        return jsonify({"success": True, "settings": settings or {}})
    except Exception as e:
        app.logger.error(f"Failed to retrieve Jira settings: {str(e)}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/jira/settings", methods=["POST"])
def save_jira_settings():
    """Save Jira connection settings."""
    try:
        data = request.json
        if not data:
            return jsonify({"success": False, "error": "No data provided"}), 400

        app.logger.info(f"Saving Jira settings for {data.get('server_url')}")

        from services.session_manager import save_jira_settings as persist_settings

        persist_settings(data)

        return jsonify({"success": True, "message": "Settings saved"})
    except Exception as e:
        app.logger.error(f"Failed to save Jira settings: {str(e)}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/jira/projects", methods=["GET"])
def get_jira_projects():
    """List available Jira projects."""
    try:
        app.logger.info("Fetching Jira projects")

        # TODO: Implement project listing using jira_service

        return jsonify({"success": True, "projects": []})
    except Exception as e:
        app.logger.error(f"Failed to fetch Jira projects: {str(e)}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/jira/issue-types/<project_key>", methods=["GET"])
def get_issue_types(project_key):
    """Get issue types for a specific project."""
    try:
        app.logger.info(f"Fetching issue types for project {project_key}")

        # TODO: Implement issue type discovery using jira_service

        return jsonify({"success": True, "issue_types": []})
    except Exception as e:
        app.logger.error(f"Failed to fetch issue types: {str(e)}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500


# ============================================================================
# API Endpoints - OCR Processing
# ============================================================================


@app.route("/api/ocr/upload", methods=["POST"])
def upload_image():
    """Upload sticky note image for OCR processing."""
    try:
        if "file" not in request.files:
            return jsonify({"success": False, "error": "No file provided"}), 400

        file = request.files["file"]
        if file.filename == "":
            return jsonify({"success": False, "error": "No file selected"}), 400

        # Save uploaded file
        filename = file.filename
        if not filename:
            return jsonify({"success": False, "error": "Invalid filename"}), 400

        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(filepath)

        app.logger.info(f"Image uploaded: {filename}")

        return jsonify({"success": True, "filename": filename, "filepath": filepath})
    except Exception as e:
        app.logger.error(f"Image upload failed: {str(e)}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/ocr/process", methods=["POST"])
def process_ocr():
    """Start OCR processing on uploaded image."""
    try:
        data = request.json
        if not data:
            return jsonify({"success": False, "error": "No data provided"}), 400

        filename = data.get("filename")
        if not filename:
            return jsonify({"success": False, "error": "No filename provided"}), 400

        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)

        if not os.path.exists(filepath):
            return jsonify({"success": False, "error": "File not found"}), 404

        app.logger.info(f"Starting OCR processing for {filename}")

        # Import OCR service
        from services.ocr_service import process_image_async
        import threading

        # Process in background thread
        def process_thread():
            try:
                process_image_async(filepath, socketio, callback_event="ocr_progress")
            except Exception as e:
                app.logger.error(f"OCR thread failed: {str(e)}", exc_info=True)
                socketio.emit(
                    "ocr_progress",
                    {"status": "error", "message": f"OCR processing failed: {str(e)}"},
                )

        thread = threading.Thread(target=process_thread)
        thread.daemon = True
        thread.start()

        return jsonify({"success": True, "message": "OCR processing started"})
    except Exception as e:
        app.logger.error(f"OCR processing failed: {str(e)}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500


# ============================================================================
# API Endpoints - Color Mapping
# ============================================================================


@app.route("/api/mapping/save", methods=["POST"])
def save_color_mapping():
    """Save color-to-issue-type mappings."""
    try:
        app.logger.info("Saving color mappings")

        # TODO: Implement mapping persistence using session_manager

        return jsonify({"success": True, "message": "Mappings saved"})
    except Exception as e:
        app.logger.error(f"Failed to save mappings: {str(e)}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/mapping/load", methods=["GET"])
def load_color_mapping():
    """Load saved color mappings for current project."""
    try:
        project_key = request.args.get("project_key")
        app.logger.info(f"Loading color mappings for {project_key}")

        # TODO: Implement mapping retrieval using session_manager

        return jsonify({"success": True, "mappings": {}})
    except Exception as e:
        app.logger.error(f"Failed to load mappings: {str(e)}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500


# ============================================================================
# API Endpoints - Issues Management
# ============================================================================


@app.route("/api/issues/preview", methods=["GET"])
def get_issues_preview():
    """Get current session's issues for review."""
    try:
        app.logger.info("Retrieving issues preview")

        # TODO: Implement issue retrieval using session_manager

        return jsonify({"success": True, "issues": []})
    except Exception as e:
        app.logger.error(f"Failed to retrieve issues: {str(e)}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/issues/bulk-edit", methods=["POST"])
def bulk_edit_issues():
    """Update multiple issues before import."""
    try:
        data = request.json
        if not data:
            return jsonify({"success": False, "error": "No data provided"}), 400

        issues = data.get("issues", [])
        app.logger.info(f"Bulk editing {len(issues)} issues")

        # TODO: Implement bulk update using session_manager

        return jsonify({"success": True, "message": f"{len(issues)} issues updated"})
    except Exception as e:
        app.logger.error(f"Bulk edit failed: {str(e)}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/issues/import", methods=["POST"])
def import_issues():
    """Import issues to Jira."""
    try:
        app.logger.info("Starting Jira import")

        # TODO: Implement Jira import using jira_service in background thread
        # Emit progress via SocketIO: import_progress events

        return jsonify({"success": True, "message": "Import started"})
    except Exception as e:
        app.logger.error(f"Import failed: {str(e)}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500


# ============================================================================
# API Endpoints - Session Management
# ============================================================================


@app.route("/api/session/new", methods=["POST"])
def new_session():
    """Start a new import session (clears existing issues)."""
    try:
        app.logger.info("Starting new session - truncating issues table")

        # TODO: Implement session reset using session_manager

        return jsonify({"success": True, "message": "New session started"})
    except Exception as e:
        app.logger.error(f"Session reset failed: {str(e)}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/session/history", methods=["GET"])
def get_import_history():
    """Retrieve import history."""
    try:
        app.logger.info("Retrieving import history")

        # TODO: Implement history retrieval using session_manager

        return jsonify({"success": True, "history": []})
    except Exception as e:
        app.logger.error(f"Failed to retrieve history: {str(e)}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500


# ============================================================================
# SocketIO Event Handlers
# ============================================================================


@socketio.on("connect")
def handle_connect():
    """Handle client connection."""
    app.logger.info("Client connected via SocketIO")
    emit("connected", {"message": "Connected to Sticky2Jira server"})


@socketio.on("disconnect")
def handle_disconnect():
    """Handle client disconnection."""
    app.logger.info("Client disconnected from SocketIO")


# ============================================================================
# Application Startup
# ============================================================================


def open_browser():
    """Open default browser to application URL."""
    url = f"http://{HOST}:{PORT}"
    app.logger.info(f"Opening browser to {url}")
    webbrowser.open(url)


if __name__ == "__main__":
    # Configuration
    HOST = os.getenv("FLASK_HOST", "127.0.0.1")
    PORT = int(os.getenv("FLASK_PORT", 5000))
    DEBUG = os.getenv("FLASK_DEBUG", "False").lower() == "true"

    # Ensure upload directory exists
    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

    # Initialize encryption key on startup (avoids delay on first settings load)
    from services.crypto_utils import _get_or_create_key

    _get_or_create_key()

    app.logger.info(f"Starting Flask server on {HOST}:{PORT} (debug={DEBUG})")

    # Auto-launch browser after 1.5 seconds
    Timer(1.5, open_browser).start()

    # Run application
    socketio.run(app, host=HOST, port=PORT, debug=DEBUG)
