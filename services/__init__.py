"""
Services package for sticky2jira.
Contains business logic modules for OCR, Jira integration, and session management.
"""

from .session_manager import (
    init_database,
    create_issue,
    bulk_create_issues,
    get_all_issues,
    update_issue,
    bulk_update_issues,
    truncate_issues,
    save_color_mapping,
    get_color_mappings,
    save_jira_settings,
    get_jira_settings,
    create_import_record,
    complete_import_record,
    get_import_history,
)

from .jira_service import JiraService, create_jira_service

from .ocr_service import OCRService, create_ocr_service, process_image_async

__all__ = [
    # Session Manager
    "init_database",
    "create_issue",
    "bulk_create_issues",
    "get_all_issues",
    "update_issue",
    "bulk_update_issues",
    "truncate_issues",
    "save_color_mapping",
    "get_color_mappings",
    "save_jira_settings",
    "get_jira_settings",
    "create_import_record",
    "complete_import_record",
    "get_import_history",
    # Jira Service
    "JiraService",
    "create_jira_service",
    # OCR Service
    "OCRService",
    "create_ocr_service",
    "process_image_async",
]
