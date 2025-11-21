"""
Sticky2Jira - Main Flask Application
Browser-based local application for extracting text from sticky notes and creating Jira issues.
"""

import os
import webbrowser
import logging
from logging.handlers import RotatingFileHandler
from threading import Timer

from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_socketio import SocketIO, emit
from flask_cors import CORS
from dotenv import load_dotenv

# Import services
from services import session_manager

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


@app.route("/uploads/<filename>")
def serve_upload(filename):
    """Serve uploaded images."""
    try:
        return send_from_directory(app.config["UPLOAD_FOLDER"], filename)
    except Exception as e:
        app.logger.error(f"Failed to serve file {filename}: {str(e)}")
        return jsonify({"success": False, "error": "File not found"}), 404


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

        # Load Jira settings
        settings = session_manager.get_jira_settings()
        if not settings:
            return jsonify(
                {"success": False, "error": "Jira connection not configured"}
            ), 400

        # Initialize Jira service
        from services.jira_service import JiraService

        jira_svc = JiraService(
            server_url=settings["server_url"],
            api_token=settings["api_token"],
        )

        # Get projects
        projects = jira_svc.get_projects()
        app.logger.info(f"Retrieved {len(projects)} projects")

        return jsonify({"success": True, "projects": projects})
    except Exception as e:
        app.logger.error(f"Failed to fetch Jira projects: {str(e)}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/field-defaults", methods=["GET"])
def get_all_field_defaults():
    """Get all field defaults configurations."""
    try:
        project_key = request.args.get("project_key")
        configs = session_manager.get_all_field_defaults_configs(project_key)
        return jsonify({"success": True, "configs": configs})
    except Exception as e:
        app.logger.error(f"Failed to fetch field defaults: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/field-defaults/<project_key>/<issue_type>", methods=["GET"])
def get_field_defaults(project_key, issue_type):
    """Get field defaults for a specific project/issue type."""
    try:
        defaults = session_manager.get_field_defaults_config(project_key, issue_type)
        return jsonify({"success": True, "field_defaults": defaults})
    except Exception as e:
        app.logger.error(f"Failed to fetch field defaults: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/field-defaults", methods=["POST"])
def save_field_defaults():
    """Save field defaults configuration for a project/issue type."""
    try:
        data = request.json
        if not data:
            return jsonify({"success": False, "error": "No data provided"}), 400

        project_key = data.get("project_key")
        issue_type = data.get("issue_type")
        field_defaults = data.get("field_defaults", {})

        if not project_key or not issue_type:
            return jsonify(
                {"success": False, "error": "project_key and issue_type required"}
            ), 400

        session_manager.save_field_defaults_config(
            project_key, issue_type, field_defaults
        )
        return jsonify({"success": True})
    except Exception as e:
        app.logger.error(f"Failed to save field defaults: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/field-defaults/<project_key>/<issue_type>", methods=["DELETE"])
def delete_field_defaults(project_key, issue_type):
    """Delete field defaults configuration."""
    try:
        session_manager.delete_field_defaults_config(project_key, issue_type)
        return jsonify({"success": True})
    except Exception as e:
        app.logger.error(f"Failed to delete field defaults: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/jira/issue-types/<project_key>", methods=["GET"])
def get_issue_types(project_key):
    """Get issue types for a specific project."""
    try:
        app.logger.info(f"Fetching issue types for project {project_key}")

        # Load Jira settings
        settings = session_manager.get_jira_settings()
        if not settings:
            return jsonify(
                {"success": False, "error": "Jira connection not configured"}
            ), 400

        # Initialize Jira service
        from services.jira_service import JiraService

        jira_svc = JiraService(
            server_url=settings["server_url"],
            api_token=settings["api_token"],
        )

        # Get issue types for project
        issue_types = jira_svc.get_issue_types(project_key)
        app.logger.info(f"Retrieved {len(issue_types)} issue types for {project_key}")

        return jsonify({"success": True, "issue_types": issue_types})
    except Exception as e:
        app.logger.error(f"Failed to fetch issue types: {str(e)}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/jira/fields", methods=["GET"])
def get_jira_fields():
    """Get all Jira fields with metadata for field defaults configuration."""
    try:
        project_key = request.args.get("project_key")
        issue_type = request.args.get("issue_type", "Task")

        if not project_key:
            return jsonify(
                {"success": False, "error": "project_key parameter required"}
            ), 400

        app.logger.info(f"Fetching fields for {project_key}/{issue_type}")

        # Load Jira settings
        settings = session_manager.get_jira_settings()
        if not settings:
            return jsonify(
                {"success": False, "error": "Jira connection not configured"}
            ), 400

        # Initialize Jira service
        from services.jira_service import JiraService

        jira_svc = JiraService(
            server_url=settings["server_url"],
            api_token=settings["api_token"],
        )

        # Get create metadata for the project/issue type
        field_meta = jira_svc.get_create_fields(project_key, issue_type)

        # Format fields for UI
        formatted_fields = []
        for field_id, field_info in field_meta.items():
            # Skip standard fields that we already handle
            if field_id in ["project", "issuetype", "summary", "description"]:
                continue

            field_data = {
                "id": field_id,
                "name": field_info.get("name"),
                "required": field_info.get("required", False),
                "type": field_info.get("schema", {}).get("type"),
                "custom": field_info.get("schema", {}).get("custom"),
            }

            # Extract allowed values if present (already converted to dicts)
            allowed_values = field_info.get("allowedValues")
            if allowed_values:
                field_data["allowedValues"] = []
                for val in allowed_values:
                    field_data["allowedValues"].append(
                        {
                            "id": val.get("id"),
                            "name": val.get("name") or val.get("value"),
                            "value": val.get("value"),
                        }
                    )

            formatted_fields.append(field_data)

        app.logger.info(
            f"Retrieved {len(formatted_fields)} fields for {project_key}/{issue_type}"
        )

        return jsonify({"success": True, "fields": formatted_fields})
    except Exception as e:
        app.logger.error(f"Failed to fetch fields: {str(e)}", exc_info=True)
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
        app.logger.error(f"Failed to save mappings: {str(e)}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500


# ============================================================================
# API Endpoints - Jira Import
# ============================================================================


@app.route("/api/jira/import", methods=["POST"])
def import_to_jira():
    """Import issues to Jira."""
    try:
        data = request.json
        if not data or "issues" not in data:
            return jsonify({"success": False, "error": "No issues provided"}), 400

        issues = data["issues"]
        app.logger.info(f"Starting import of {len(issues)} issues to Jira")

        # Get Jira settings
        settings = session_manager.get_jira_settings()
        if not settings:
            return (
                jsonify({"success": False, "error": "Jira not configured"}),
                400,
            )

        # Initialize Jira service
        from services.jira_service import JiraService

        jira_service = JiraService(
            server_url=settings["server_url"], api_token=settings["api_token"]
        )

        # Get all field defaults configurations for efficient lookup
        all_configs = session_manager.get_all_field_defaults_configs()
        field_defaults_by_type = {
            f"{config['project_key']}_{config['issue_type']}": config["field_defaults"]
            for config in all_configs
        }

        # Import issues
        results = []
        created_count = 0
        updated_count = 0
        failed_count = 0

        for idx, issue in enumerate(issues):
            try:
                # Emit progress
                socketio.emit(
                    "import_progress",
                    {
                        "percent": int((idx / len(issues)) * 100),
                        "current": idx + 1,
                        "total": len(issues),
                        "status": "processing",
                        "message": f"Importing issue {idx + 1}/{len(issues)}...",
                    },
                )

                # Build issue data - start with required fields
                issue_data = {
                    "project": {"key": issue["project_key"]},
                    "summary": issue["summary"] or "No summary",
                    "issuetype": {"name": issue["issue_type"]},
                }

                # Add description if provided
                description = issue.get("description", "").strip()
                if description:
                    issue_data["description"] = description

                # Apply field defaults for this specific project/issue type combination
                lookup_key = f"{issue['project_key']}_{issue['issue_type']}"
                field_defaults = field_defaults_by_type.get(lookup_key, {})

                if field_defaults:
                    for field_id, field_value in field_defaults.items():
                        if field_id not in issue_data:  # Don't override existing values
                            issue_data[field_id] = field_value
                    app.logger.info(
                        f"Applied field defaults for {issue['project_key']}/{issue['issue_type']}"
                    )
                else:
                    app.logger.warning(
                        f"No field defaults configured for {issue['project_key']}/{issue['issue_type']}"
                    )

                # Check if issue already has a Jira key (update vs create)
                existing_issue_key = issue.get("issue_key")
                issue_id = issue.get("db_id") or issue.get(
                    "id"
                )  # Try db_id first, fallback to id

                if existing_issue_key:
                    # Update existing issue
                    jira_service.update_issue(
                        issue_key=existing_issue_key,
                        summary=issue_data.get("summary"),
                        description=issue_data.get("description"),
                        additional_fields={
                            k: v
                            for k, v in issue_data.items()
                            if k
                            not in ["project", "summary", "issuetype", "description"]
                        },
                    )
                    issue_key = existing_issue_key
                    updated_count += 1
                    status = "updated"
                    app.logger.info(f"Updated issue {issue_key}")
                else:
                    # Create new issue
                    created_issue = jira_service.client.create_issue(fields=issue_data)
                    issue_key = created_issue.key

                    # Save issue_key to database
                    if issue_id:
                        session_manager.update_issue_key(issue_id, issue_key)
                        app.logger.info(
                            f"Saved issue_key {issue_key} to database for issue {issue_id}"
                        )

                    created_count += 1
                    status = "created"
                    app.logger.info(f"Created issue {issue_key}")

                results.append(
                    {
                        "success": True,
                        "issue_key": issue_key,
                        "summary": issue["summary"],
                        "status": status,
                    }
                )

            except Exception as e:
                error_msg = str(e)
                app.logger.error(f"Failed to create issue: {error_msg}")

                # Try to extract meaningful error from Jira API response
                try:
                    import json as json_lib

                    json_start = error_msg.find("{")
                    if json_start != -1:
                        error_payload = json_lib.loads(error_msg[json_start:])
                        errors = error_payload.get("errors", {})
                        if errors:
                            error_msg = "; ".join(
                                [f"{k}: {v}" for k, v in errors.items()]
                            )
                        elif error_payload.get("errorMessages"):
                            error_msg = "; ".join(error_payload["errorMessages"])
                except Exception:
                    pass

                results.append(
                    {
                        "success": False,
                        "error": error_msg,
                        "summary": issue.get("summary", "Unknown"),
                    }
                )
                failed_count += 1

        # Final progress update
        socketio.emit(
            "import_progress",
            {
                "percent": 100,
                "current": len(issues),
                "total": len(issues),
                "status": "complete",
                "message": f"Import complete! Created {created_count}, updated {updated_count}, {failed_count} failed.",
                "created": created_count,
                "updated": updated_count,
                "failed": failed_count,
                "results": results,
            },
        )

        return jsonify(
            {
                "success": True,
                "created": created_count,
                "updated": updated_count,
                "failed": failed_count,
                "total": len(issues),
                "results": results,
            }
        )

    except Exception as e:
        app.logger.error(f"Import failed: {str(e)}", exc_info=True)
        socketio.emit(
            "import_progress",
            {"status": "error", "message": f"Import failed: {str(e)}"},
        )
        return jsonify({"success": False, "error": str(e)}), 500


# ============================================================================
# API Endpoints - Session Management
# ============================================================================


@app.route("/api/issues/preview", methods=["GET"])
def get_issues_preview():
    """Get current session's issues for review."""
    try:
        app.logger.info("Retrieving issues preview")

        # Get all issues from database
        issues = session_manager.get_all_issues()

        # Transform to frontend format
        issues_data = [
            {
                "db_id": issue["id"],
                "id": issue["region_id"],
                "color_hex": issue["color_hex"],
                "color_name": "unknown",  # TODO: derive from color_hex
                "text": issue["summary"],
                "linked_text": issue["description"],
                "issue_type": issue["issue_type"],
                "project_key": issue["project_key"],
                "issue_key": issue["issue_key"],
                "confidence": issue["confidence"],
                "image_filename": issue.get("image_filename", ""),
                "bbox": {
                    "x": issue["bbox_x"],
                    "y": issue["bbox_y"],
                    "width": issue["bbox_width"],
                    "height": issue["bbox_height"],
                },
            }
            for issue in issues
        ]

        return jsonify({"success": True, "issues": issues_data})
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


@app.route("/api/issues/bulk-update-mapping", methods=["POST"])
def bulk_update_mapping():
    """Update project_key and issue_type for multiple issues after color mapping."""
    try:
        data = request.json
        if not data:
            return jsonify({"success": False, "error": "No data provided"}), 400

        updates = data.get("updates", [])
        app.logger.info(f"Updating mappings for {len(updates)} issues")

        updated_count = 0
        for update in updates:
            issue_id = update.get("id")
            if not issue_id:
                continue

            update_data = {}
            if "project_key" in update:
                update_data["project_key"] = update["project_key"]
            if "issue_type" in update:
                update_data["issue_type"] = update["issue_type"]

            if update_data and session_manager.update_issue(issue_id, update_data):
                updated_count += 1

        app.logger.info(f"Updated {updated_count} issue mappings")
        return jsonify(
            {
                "success": True,
                "message": f"Updated {updated_count} issue mappings",
                "updated": updated_count,
            }
        )
    except Exception as e:
        app.logger.error(f"Bulk update mapping failed: {str(e)}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/issues/<int:issue_id>", methods=["PUT"])
def update_single_issue(issue_id):
    """Update a single issue field (summary or description)."""
    try:
        data = request.json
        if not data:
            return jsonify({"success": False, "error": "No data provided"}), 400

        app.logger.info(f"Updating issue {issue_id}: {data}")

        if session_manager.update_issue(issue_id, data):
            return jsonify({"success": True, "message": "Issue updated"})
        else:
            return jsonify({"success": False, "error": "Update failed"}), 500
    except Exception as e:
        app.logger.error(f"Update issue failed: {str(e)}", exc_info=True)
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

        # Clear all issues from database
        session_manager.truncate_issues()

        app.logger.info("New session started - all issues cleared")
        return jsonify(
            {"success": True, "message": "New session started - all issues cleared"}
        )
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
