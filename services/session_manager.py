"""
Session Manager - SQLite database operations for sticky2jira.
Handles issues, color mappings, Jira settings, and import history persistence.
"""

import sqlite3
import logging
from datetime import datetime
from typing import List, Dict, Optional, Any
from contextlib import contextmanager

logger = logging.getLogger(__name__)

DB_PATH = "session.db"

# ============================================================================
# Database Schema
# ============================================================================

SCHEMA_SQL = """
-- Issues table: Stores extracted sticky note content before/after Jira import
CREATE TABLE IF NOT EXISTS issues (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    image_filename TEXT NOT NULL,
    region_id INTEGER NOT NULL,
    color_hex TEXT NOT NULL,
    summary TEXT NOT NULL,
    description TEXT,
    issue_type TEXT,
    project_key TEXT,
    issue_key TEXT,  -- Populated after Jira import (e.g., 'A935-156')
    confidence REAL,
    bbox_x INTEGER,
    bbox_y INTEGER,
    bbox_width INTEGER,
    bbox_height INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Color mappings table: User preferences for color-to-issue-type mapping
CREATE TABLE IF NOT EXISTS color_mappings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    project_key TEXT NOT NULL,
    color_hex TEXT NOT NULL,
    issue_type TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(color_hex, project_key)
);

-- Jira settings table: Connection credentials and defaults (single row)
CREATE TABLE IF NOT EXISTS jira_settings (
    id INTEGER PRIMARY KEY CHECK (id = 1),  -- Enforce single row
    server_url TEXT NOT NULL,
    api_token BLOB NOT NULL,  -- Encrypted API token
    default_project_key TEXT,
    field_defaults TEXT,  -- JSON string of field defaults
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Import history table: Audit log of Jira import operations
CREATE TABLE IF NOT EXISTS import_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    project_key TEXT NOT NULL,
    issues_created INTEGER DEFAULT 0,
    issues_updated INTEGER DEFAULT 0,
    issues_failed INTEGER DEFAULT 0,
    image_filename TEXT,
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    status TEXT,  -- 'success', 'partial', 'failed'
    error_message TEXT
);

-- Indices for performance
CREATE INDEX IF NOT EXISTS idx_issues_project_key ON issues(project_key);
CREATE INDEX IF NOT EXISTS idx_issues_issue_key ON issues(issue_key);
CREATE INDEX IF NOT EXISTS idx_issues_image_filename ON issues(image_filename);
CREATE INDEX IF NOT EXISTS idx_issues_created_at ON issues(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_color_mappings_project ON color_mappings(project_key);
CREATE INDEX IF NOT EXISTS idx_import_history_project ON import_history(project_key);
CREATE INDEX IF NOT EXISTS idx_import_history_timestamp ON import_history(completed_at DESC);
"""


# ============================================================================
# Database Connection Context Manager
# ============================================================================


@contextmanager
def get_db_connection():
    """
    Context manager for database connections.
    Ensures proper connection cleanup and transaction handling.
    """
    conn = None
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row  # Enable dict-like access
        yield conn
        conn.commit()
    except Exception as e:
        if conn:
            conn.rollback()
        logger.error(f"Database error: {str(e)}", exc_info=True)
        raise
    finally:
        if conn:
            conn.close()


# ============================================================================
# Database Initialization
# ============================================================================


def init_database():
    """Create database schema if not exists."""
    try:
        with get_db_connection() as conn:
            conn.executescript(SCHEMA_SQL)
        logger.info("Database schema initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize database: {str(e)}", exc_info=True)
        raise


# ============================================================================
# Issues CRUD Operations
# ============================================================================


def create_issue(issue_data: Dict[str, Any]) -> int:
    """
    Insert a new issue into the database.

    Args:
        issue_data: Dictionary with keys matching issues table columns

    Returns:
        ID of the newly created issue
    """
    required_fields = ["image_filename", "region_id", "color_hex", "summary"]
    for field in required_fields:
        if field not in issue_data:
            raise ValueError(f"Missing required field: {field}")

    with get_db_connection() as conn:
        cursor = conn.execute(
            """
            INSERT INTO issues (
                image_filename, region_id, color_hex, summary, description,
                issue_type, project_key, issue_key, confidence,
                bbox_x, bbox_y, bbox_width, bbox_height
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                issue_data.get("image_filename"),
                issue_data.get("region_id"),
                issue_data.get("color_hex"),
                issue_data.get("summary"),
                issue_data.get("description"),
                issue_data.get("issue_type"),
                issue_data.get("project_key"),
                issue_data.get("issue_key"),
                issue_data.get("confidence"),
                issue_data.get("bbox_x"),
                issue_data.get("bbox_y"),
                issue_data.get("bbox_width"),
                issue_data.get("bbox_height"),
            ),
        )
        issue_id = cursor.lastrowid
        if issue_id is None:
            raise RuntimeError("Failed to retrieve issue ID after insert")
        logger.info(f"Created issue {issue_id}: {issue_data.get('summary')}")
        return issue_id


def bulk_create_issues(issues: List[Dict[str, Any]]) -> int:
    """
    Insert multiple issues efficiently.

    Args:
        issues: List of issue dictionaries

    Returns:
        Number of issues created
    """
    if not issues:
        return 0

    with get_db_connection() as conn:
        conn.executemany(
            """
            INSERT INTO issues (
                image_filename, region_id, color_hex, summary, description,
                issue_type, project_key, issue_key, confidence,
                bbox_x, bbox_y, bbox_width, bbox_height
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    issue.get("image_filename"),
                    issue.get("region_id"),
                    issue.get("color_hex"),
                    issue.get("summary"),
                    issue.get("description"),
                    issue.get("issue_type"),
                    issue.get("project_key"),
                    issue.get("issue_key"),
                    issue.get("confidence"),
                    issue.get("bbox_x"),
                    issue.get("bbox_y"),
                    issue.get("bbox_width"),
                    issue.get("bbox_height"),
                )
                for issue in issues
            ],
        )
    logger.info(f"Bulk created {len(issues)} issues")
    return len(issues)


def get_all_issues() -> List[Dict[str, Any]]:
    """Retrieve all issues from the current session."""
    with get_db_connection() as conn:
        cursor = conn.execute(
            """
            SELECT * FROM issues
            ORDER BY created_at DESC, region_id ASC
            """
        )
        return [dict(row) for row in cursor.fetchall()]


def get_issues_by_image(image_filename: str) -> List[Dict[str, Any]]:
    """Retrieve all issues for a specific image."""
    with get_db_connection() as conn:
        cursor = conn.execute(
            """
            SELECT * FROM issues
            WHERE image_filename = ?
            ORDER BY region_id ASC
            """,
            (image_filename,),
        )
        return [dict(row) for row in cursor.fetchall()]


def update_issue(issue_id: int, updates: Dict[str, Any]) -> bool:
    """
    Update an existing issue.

    Args:
        issue_id: ID of the issue to update
        updates: Dictionary of field:value pairs to update

    Returns:
        True if update successful
    """
    if not updates:
        return False

    # Build dynamic UPDATE query
    set_clause = ", ".join([f"{field} = ?" for field in updates.keys()])
    values = list(updates.values()) + [datetime.now(), issue_id]

    with get_db_connection() as conn:
        conn.execute(
            f"""
            UPDATE issues
            SET {set_clause}, updated_at = ?
            WHERE id = ?
            """,
            values,
        )
    logger.info(f"Updated issue {issue_id}: {list(updates.keys())}")
    return True


def bulk_update_issues(updates: List[Dict[str, Any]]) -> int:
    """
    Update multiple issues efficiently.

    Args:
        updates: List of dicts with 'id' and field updates

    Returns:
        Number of issues updated
    """
    if not updates:
        return 0

    count = 0
    with get_db_connection() as conn:
        for update in updates:
            issue_id = update.pop("id", None)
            if not issue_id:
                continue

            set_clause = ", ".join([f"{field} = ?" for field in update.keys()])
            values = list(update.values()) + [datetime.now(), issue_id]

            conn.execute(
                f"""
                UPDATE issues
                SET {set_clause}, updated_at = ?
                WHERE id = ?
                """,
                values,
            )
            count += 1

    logger.info(f"Bulk updated {count} issues")
    return count


def update_issue_key(issue_id: int, issue_key: str) -> bool:
    """
    Update issue_key after successful Jira import.

    Args:
        issue_id: Database ID of the issue
        issue_key: Jira issue key (e.g., 'A935-156')

    Returns:
        True if update successful
    """
    with get_db_connection() as conn:
        conn.execute(
            """
            UPDATE issues
            SET issue_key = ?, updated_at = ?
            WHERE id = ?
            """,
            (issue_key, datetime.now(), issue_id),
        )
    logger.info(f"Updated issue {issue_id} with Jira key {issue_key}")
    return True


def truncate_issues() -> None:
    """Delete all issues (used when starting a new import session)."""
    with get_db_connection() as conn:
        conn.execute("DELETE FROM issues")
    logger.info("Truncated issues table for new session")


# ============================================================================
# Color Mappings CRUD Operations
# ============================================================================


def save_color_mapping(project_key: str, color_hex: str, issue_type: str) -> int:
    """
    Save or update color-to-issue-type mapping.

    Args:
        project_key: Jira project key
        color_hex: Sticky note color (e.g., '#FFFF00')
        issue_type: Jira issue type name

    Returns:
        ID of the mapping
    """
    with get_db_connection() as conn:
        cursor = conn.execute(
            """
            INSERT INTO color_mappings (project_key, color_hex, issue_type)
            VALUES (?, ?, ?)
            ON CONFLICT(color_hex, project_key)
            DO UPDATE SET issue_type = ?, updated_at = ?
            """,
            (project_key, color_hex, issue_type, issue_type, datetime.now()),
        )
        mapping_id = cursor.lastrowid
        if mapping_id is None:
            raise RuntimeError("Failed to retrieve mapping ID after insert")
    logger.info(f"Saved color mapping: {color_hex} -> {issue_type} for {project_key}")
    return mapping_id


def get_color_mappings(project_key: str) -> Dict[str, str]:
    """
    Retrieve color mappings for a project.

    Args:
        project_key: Jira project key

    Returns:
        Dictionary {color_hex: issue_type}
    """
    with get_db_connection() as conn:
        cursor = conn.execute(
            """
            SELECT color_hex, issue_type
            FROM color_mappings
            WHERE project_key = ?
            """,
            (project_key,),
        )
        return {row["color_hex"]: row["issue_type"] for row in cursor.fetchall()}


# ============================================================================
# Jira Settings CRUD Operations
# ============================================================================


def save_jira_settings(settings: Dict[str, str]) -> bool:
    """
    Save Jira connection settings (single row table).
    API token is encrypted before storage.

    Args:
        settings: Dictionary with keys: server_url, api_token, default_project_key

    Returns:
        True if save successful
    """
    required = ["server_url", "api_token"]
    for field in required:
        if field not in settings:
            raise ValueError(f"Missing required field: {field}")

    # Encrypt API token
    from services.crypto_utils import encrypt_token
    import json

    encrypted_token = encrypt_token(settings["api_token"])

    # Serialize field_defaults to JSON if provided
    field_defaults_json = None
    if "field_defaults" in settings and settings["field_defaults"]:
        field_defaults_json = json.dumps(settings["field_defaults"])

    with get_db_connection() as conn:
        conn.execute(
            """
            INSERT INTO jira_settings (id, server_url, api_token, default_project_key, field_defaults, field_defaults_project, field_defaults_issue_type)
            VALUES (1, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id)
            DO UPDATE SET 
                server_url = ?,
                api_token = ?,
                default_project_key = ?,
                field_defaults = ?,
                field_defaults_project = ?,
                field_defaults_issue_type = ?,
                updated_at = ?
            """,
            (
                settings["server_url"],
                encrypted_token,
                settings.get("default_project_key"),
                field_defaults_json,
                settings.get("field_defaults_project"),
                settings.get("field_defaults_issue_type"),
                settings["server_url"],
                encrypted_token,
                settings.get("default_project_key"),
                field_defaults_json,
                settings.get("field_defaults_project"),
                settings.get("field_defaults_issue_type"),
                datetime.now(),
            ),
        )
    logger.info(f"Saved Jira settings for {settings['server_url']} (token encrypted)")
    return True


def get_jira_settings() -> Optional[Dict[str, Any]]:
    """
    Retrieve Jira connection settings.
    API token is decrypted using Windows DPAPI after loading.

    Returns:
        Dictionary with settings or None if not configured
    """
    with get_db_connection() as conn:
        cursor = conn.execute("SELECT * FROM jira_settings WHERE id = 1")
        row = cursor.fetchone()

        if not row:
            return None

        settings = dict(row)

        # Decrypt API token
        from services.crypto_utils import decrypt_token, is_encrypted
        import json

        encrypted_token = settings["api_token"]

        # Handle migration: check if token is already encrypted or plaintext
        if isinstance(encrypted_token, bytes) and is_encrypted(encrypted_token):
            # Already encrypted - decrypt it
            try:
                settings["api_token"] = decrypt_token(encrypted_token)
            except Exception as e:
                logger.error(f"Failed to decrypt API token: {e}")
                raise ValueError(
                    "Unable to decrypt API token. It may have been encrypted by a different user."
                )
        elif isinstance(encrypted_token, str):
            # Legacy plaintext token - migrate to encrypted
            logger.warning("Migrating plaintext API token to encrypted storage")
            plaintext_token = encrypted_token
            settings["api_token"] = plaintext_token

            # Re-save to encrypt it
            save_jira_settings(settings)
        else:
            # Bytes but not encrypted (shouldn't happen, but handle gracefully)
            logger.warning("Unknown token format, treating as plaintext")
            settings["api_token"] = (
                encrypted_token.decode("utf-8")
                if isinstance(encrypted_token, bytes)
                else str(encrypted_token)
            )

        # Parse field_defaults JSON if present
        field_defaults = {}
        if settings.get("field_defaults"):
            try:
                field_defaults = json.loads(settings["field_defaults"])
            except Exception:
                field_defaults = {}
        settings["field_defaults"] = field_defaults

        # Include field_defaults_project and field_defaults_issue_type
        settings["field_defaults_project"] = settings.get("field_defaults_project")
        settings["field_defaults_issue_type"] = settings.get(
            "field_defaults_issue_type"
        )

        return settings


# ============================================================================
# Field Defaults Configuration CRUD Operations
# ============================================================================


def save_field_defaults_config(
    project_key: str, issue_type: str, field_defaults: Dict[str, Any]
) -> bool:
    """
    Save field defaults configuration for a specific project/issue type.

    Args:
        project_key: Jira project key
        issue_type: Issue type name (e.g., 'Story', 'Task', 'Bug')
        field_defaults: Dictionary of field defaults

    Returns:
        True if successful
    """
    import json

    field_defaults_json = json.dumps(field_defaults) if field_defaults else "{}"

    with get_db_connection() as conn:
        conn.execute(
            """
            INSERT INTO field_defaults_config (project_key, issue_type, field_defaults)
            VALUES (?, ?, ?)
            ON CONFLICT(project_key, issue_type)
            DO UPDATE SET 
                field_defaults = ?,
                updated_at = CURRENT_TIMESTAMP
            """,
            (project_key, issue_type, field_defaults_json, field_defaults_json),
        )
    logger.info(f"Saved field defaults for {project_key}/{issue_type}")
    return True


def get_field_defaults_config(
    project_key: str, issue_type: str
) -> Optional[Dict[str, Any]]:
    """
    Get field defaults for a specific project/issue type.

    Args:
        project_key: Jira project key
        issue_type: Issue type name

    Returns:
        Dictionary of field defaults or None
    """
    import json

    with get_db_connection() as conn:
        cursor = conn.execute(
            """
            SELECT field_defaults 
            FROM field_defaults_config 
            WHERE project_key = ? AND issue_type = ?
            """,
            (project_key, issue_type),
        )
        row = cursor.fetchone()

    if not row:
        return None

    try:
        return json.loads(row["field_defaults"])
    except Exception:
        return {}


def get_all_field_defaults_configs(
    project_key: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Get all field defaults configurations, optionally filtered by project.

    Args:
        project_key: Optional project key to filter by

    Returns:
        List of configuration dictionaries with project_key, issue_type, and field_defaults
    """
    import json

    with get_db_connection() as conn:
        if project_key:
            cursor = conn.execute(
                """
                SELECT project_key, issue_type, field_defaults 
                FROM field_defaults_config 
                WHERE project_key = ?
                ORDER BY issue_type
                """,
                (project_key,),
            )
        else:
            cursor = conn.execute(
                """
                SELECT project_key, issue_type, field_defaults 
                FROM field_defaults_config 
                ORDER BY project_key, issue_type
                """
            )
        rows = cursor.fetchall()

    configs = []
    for row in rows:
        try:
            field_defaults = json.loads(row["field_defaults"])
        except Exception:
            field_defaults = {}

        configs.append(
            {
                "project_key": row["project_key"],
                "issue_type": row["issue_type"],
                "field_defaults": field_defaults,
            }
        )

    return configs


def delete_field_defaults_config(project_key: str, issue_type: str) -> bool:
    """
    Delete field defaults configuration for a specific project/issue type.

    Args:
        project_key: Jira project key
        issue_type: Issue type name

    Returns:
        True if successful
    """
    with get_db_connection() as conn:
        conn.execute(
            """
            DELETE FROM field_defaults_config 
            WHERE project_key = ? AND issue_type = ?
            """,
            (project_key, issue_type),
        )
    logger.info(f"Deleted field defaults for {project_key}/{issue_type}")
    return True


# ============================================================================
# Import History CRUD Operations
# ============================================================================


def create_import_record(project_key: str, image_filename: str) -> int:
    """
    Create new import history record (started state).

    Args:
        project_key: Jira project key
        image_filename: Source image filename

    Returns:
        ID of the import record
    """
    with get_db_connection() as conn:
        cursor = conn.execute(
            """
            INSERT INTO import_history (project_key, image_filename, started_at, status)
            VALUES (?, ?, ?, 'in_progress')
            """,
            (project_key, image_filename, datetime.now()),
        )
        record_id = cursor.lastrowid
        if record_id is None:
            raise RuntimeError("Failed to retrieve record ID after insert")
    logger.info(f"Created import record {record_id} for {project_key}")
    return record_id


def complete_import_record(
    record_id: int,
    created: int,
    updated: int,
    failed: int,
    error_message: Optional[str] = None,
) -> bool:
    """
    Mark import record as completed with results.

    Args:
        record_id: Import history record ID
        created: Number of issues created
        updated: Number of issues updated
        failed: Number of issues failed
        error_message: Error details if any failures

    Returns:
        True if update successful
    """
    status = (
        "success" if failed == 0 else ("partial" if created + updated > 0 else "failed")
    )

    with get_db_connection() as conn:
        conn.execute(
            """
            UPDATE import_history
            SET issues_created = ?,
                issues_updated = ?,
                issues_failed = ?,
                completed_at = ?,
                status = ?,
                error_message = ?
            WHERE id = ?
            """,
            (
                created,
                updated,
                failed,
                datetime.now(),
                status,
                error_message,
                record_id,
            ),
        )
    logger.info(
        f"Completed import record {record_id}: {created} created, {updated} updated, {failed} failed"
    )
    return True


def get_import_history(limit: int = 50) -> List[Dict[str, Any]]:
    """
    Retrieve recent import history.

    Args:
        limit: Maximum number of records to return

    Returns:
        List of import history records
    """
    with get_db_connection() as conn:
        cursor = conn.execute(
            """
            SELECT * FROM import_history
            ORDER BY completed_at DESC
            LIMIT ?
            """,
            (limit,),
        )
        return [dict(row) for row in cursor.fetchall()]


# ============================================================================
# Initialization
# ============================================================================

# Initialize database on module import
try:
    init_database()
except Exception as e:
    logger.error(f"Failed to initialize database on startup: {str(e)}")
