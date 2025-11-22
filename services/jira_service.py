"""
Jira Service - Handles Jira API operations for sticky2jira.
Connection management, field discovery, issue creation/update with progress callbacks.
"""

import logging
import urllib3
from typing import Dict, List, Optional, Any, Callable

from jira import JIRA
from jira.exceptions import JIRAError

# Disable SSL warnings for internal Jira servers without valid certificates
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

logger = logging.getLogger(__name__)


# ============================================================================
# Jira Client Management
# ============================================================================


class JiraService:
    """Manages Jira API interactions and issue operations."""

    def __init__(self, server_url: str, api_token: str):
        """
        Initialize Jira service.

        Args:
            server_url: Jira server URL (e.g., 'https://company.atlassian.net')
            api_token: Jira API token
        """
        self.server_url = server_url.rstrip("/")
        self.api_token = api_token
        self._client: Optional[JIRA] = None

    @property
    def client(self) -> JIRA:
        """Lazy-loaded Jira client with connection reuse."""
        if self._client is None:
            try:
                self._client = JIRA(
                    server=self.server_url,
                    token_auth=self.api_token,
                    options={
                        "verify": False
                    },  # Disable SSL verification for internal Jira servers
                )
                logger.info(f"Connected to Jira server: {self.server_url}")
            except JIRAError as e:
                logger.error(
                    f"Jira connection failed: {e.status_code} - {e.text}", exc_info=True
                )
                raise ConnectionError(f"Failed to connect to Jira: {e.text}")
            except Exception as e:
                logger.error(
                    f"Unexpected error connecting to Jira: {str(e)}", exc_info=True
                )
                raise
        return self._client

    def test_connection(self) -> Dict[str, Any]:
        """
        Test Jira connection and retrieve server info.

        Returns:
            Dictionary with connection status and server info
        """
        try:
            info = self.client.server_info()
            logger.info(f"Jira connection successful: {info.get('serverTitle')}")
            return {
                "success": True,
                "server_title": info.get("serverTitle"),
                "version": info.get("version"),
                "base_url": info.get("baseUrl"),
            }
        except JIRAError as e:
            logger.error(f"Connection test failed: {e.status_code} - {e.text}")
            return {
                "success": False,
                "error": f"Jira API error: {e.text}",
                "status_code": e.status_code,
            }
        except Exception as e:
            logger.error(f"Connection test failed: {str(e)}", exc_info=True)
            return {"success": False, "error": str(e)}

    # ============================================================================
    # Project and Issue Type Discovery
    # ============================================================================

    def get_projects(self) -> List[Dict[str, str]]:
        """
        Retrieve all accessible Jira projects.

        Returns:
            List of projects with keys: key, name, id
        """
        try:
            projects = self.client.projects()
            result = [{"key": p.key, "name": p.name, "id": p.id} for p in projects]
            logger.info(f"Retrieved {len(result)} Jira projects")
            return result
        except JIRAError as e:
            logger.error(f"Failed to retrieve projects: {e.text}")
            raise

    def get_issue_types(self, project_key: str) -> List[Dict[str, str]]:
        """
        Get available issue types for a project.

        Args:
            project_key: Jira project key (e.g., 'A935')

        Returns:
            List of issue types with keys: id, name, description
        """
        try:
            project = self.client.project(project_key)
            issue_types = project.issueTypes
            result = [
                {
                    "id": it.id,
                    "name": it.name,
                    "description": getattr(it, "description", ""),
                }
                for it in issue_types
            ]
            logger.info(f"Retrieved {len(result)} issue types for {project_key}")
            return result
        except JIRAError as e:
            logger.error(f"Failed to retrieve issue types for {project_key}: {e.text}")
            raise

    # ============================================================================
    # Field Discovery
    # ============================================================================

    def get_create_fields(self, project_key: str, issue_type: str) -> Dict[str, Any]:
        """
        Discover all available fields for creating an issue.

        Args:
            project_key: Jira project key
            issue_type: Issue type name (e.g., 'Task', 'Bug')

        Returns:
            Dictionary of field metadata
        """
        try:
            # For Jira 10.3.9+, use project_issue_fields
            # First get the issue type ID
            issue_types = self.get_issue_types(project_key)
            issue_type_id = None
            for it in issue_types:
                if it["name"] == issue_type:
                    issue_type_id = it["id"]
                    break

            if not issue_type_id:
                raise ValueError(
                    f'Issue type "{issue_type}" not found in {project_key}'
                )

            # Get fields using the newer API
            fields = self.client.project_issue_fields(project_key, issue_type_id)

            # Convert to the expected format
            field_meta = {}
            for field in fields:
                field_id = getattr(field, "id", None) or getattr(field, "fieldId", None)
                if field_id:
                    # Convert schema (PropertyHolder) to dict
                    schema = getattr(field, "schema", None)
                    schema_dict = {}
                    if schema:
                        schema_dict = {
                            "type": getattr(schema, "type", None),
                            "custom": getattr(schema, "custom", None),
                        }

                    # Convert allowedValues (PropertyHolder objects) to dicts
                    allowed_values = getattr(field, "allowedValues", None)
                    allowed_values_list = None
                    if allowed_values:
                        allowed_values_list = []
                        for val in allowed_values:
                            # PropertyHolder objects - use getattr to extract properties
                            val_dict = {
                                "id": getattr(val, "id", None),
                                "name": getattr(val, "name", None),
                                "value": getattr(val, "value", None),
                            }
                            allowed_values_list.append(val_dict)

                    field_meta[field_id] = {
                        "name": getattr(field, "name", ""),
                        "required": getattr(field, "required", False),
                        "schema": schema_dict,
                        "allowedValues": allowed_values_list,
                    }

            # Fetch team options for team fields that don't have allowedValues
            self._fetch_team_options(project_key, field_meta)

            logger.info(
                f"Retrieved {len(field_meta)} fields for {project_key}/{issue_type}"
            )
            return field_meta
        except JIRAError as e:
            logger.error(f"Failed to retrieve field metadata: {e.text}")
            raise

    def _fetch_team_options(self, project_key: str, field_meta: Dict[str, Any]) -> None:
        """
        Fetch team options for team custom fields using the customfield options API.

        Args:
            project_key: Jira project key
            field_meta: Field metadata dictionary (modified in-place)
        """
        try:
            for field_id, field_info in field_meta.items():
                # Skip if already has allowedValues
                if field_info.get("allowedValues"):
                    continue

                # Check if this is a team field based on name or custom type
                field_name = field_info.get("name", "").lower()
                schema = field_info.get("schema", {})
                custom_type = schema.get("custom", "")

                if "team" in field_name or "com.atlassian.teams" in custom_type:
                    logger.info(
                        f"Attempting to fetch team options for {field_id} ({field_info['name']})"
                    )

                    # Try multiple API endpoints to get team data
                    teams = self._fetch_customfield_options(field_id)

                    if teams:
                        field_info["allowedValues"] = teams
                        logger.info(
                            f"Successfully loaded {len(teams)} teams for {field_id}"
                        )
                    else:
                        logger.warning(
                            f"Could not fetch teams for {field_id} - user will need to enter team ID manually"
                        )

        except Exception as e:
            logger.warning(f"Failed to fetch team options: {e}")

    def _fetch_customfield_options(
        self, field_id: str
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Fetch options for a custom field using various Jira API endpoints.

        Args:
            field_id: Custom field ID (e.g., 'customfield_10912')

        Returns:
            List of option dictionaries with id, name, and value
        """
        try:
            # Method 1: Try the customfield options endpoint
            url = f"{self.server_url}/rest/api/2/customFieldOption/{field_id}"
            if self.client._session is None:  # type: ignore[attr-defined]
                return []
            response = self.client._session.get(url)  # type: ignore[attr-defined]

            if response.status_code == 200:
                data = response.json()
                options = []

                if isinstance(data, dict) and "values" in data:
                    for option in data["values"]:
                        options.append(
                            {
                                "id": str(option.get("id")),
                                "name": option.get("value") or option.get("name"),
                                "value": option.get("value") or option.get("name"),
                            }
                        )
                elif isinstance(data, list):
                    for option in data:
                        options.append(
                            {
                                "id": str(option.get("id")),
                                "name": option.get("value") or option.get("name"),
                                "value": option.get("value") or option.get("name"),
                            }
                        )

                if options:
                    return options

            # Method 2: Try field-specific API
            url = f"{self.server_url}/rest/api/2/field/{field_id}/option"
            if self.client._session is None:  # type: ignore[attr-defined]
                return []
            response = self.client._session.get(url)  # type: ignore[attr-defined]

            if response.status_code == 200:
                data = response.json()
                options = []

                values = data.get("values", []) if isinstance(data, dict) else data
                for option in values:
                    options.append(
                        {
                            "id": str(option.get("id")),
                            "name": option.get("value") or option.get("name"),
                            "value": option.get("value") or option.get("name"),
                        }
                    )

                if options:
                    return options

            logger.debug(f"No options found for {field_id} via API endpoints")
            return None

        except Exception as e:
            logger.debug(f"Failed to fetch options for {field_id}: {e}")
            return None

    # ============================================================================
    # Issue Creation and Update
    # ============================================================================

    def create_issue(
        self,
        project_key: str,
        issue_type: str,
        summary: str,
        description: Optional[str] = None,
        additional_fields: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Create a new Jira issue.

        Args:
            project_key: Jira project key
            issue_type: Issue type name
            summary: Issue summary/title
            description: Issue description (optional)
            additional_fields: Extra field values (optional)

        Returns:
            Dictionary with issue_key, id, and url
        """
        try:
            # Build base field structure
            fields = {
                "project": {"key": project_key},
                "issuetype": {"name": issue_type},
                "summary": summary,
            }

            # Add description if provided
            if description:
                fields["description"] = description

            # Merge additional fields
            if additional_fields:
                fields.update(self._normalize_field_values(additional_fields))

            # Create issue
            issue = self.client.create_issue(fields=fields)

            result = {
                "issue_key": issue.key,
                "id": issue.id,
                "url": f"{self.server_url}/browse/{issue.key}",
            }

            logger.info(f"Created issue {issue.key}: {summary}")
            return result
        except JIRAError as e:
            # Translate field IDs to names for user-friendly error messages
            error_msg = self._translate_field_errors(e.text)
            logger.error(f"Failed to create issue: {error_msg}")
            raise ValueError(error_msg)

    def update_issue(
        self,
        issue_key: str,
        summary: Optional[str] = None,
        description: Optional[str] = None,
        additional_fields: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Update an existing Jira issue.

        Args:
            issue_key: Jira issue key (e.g., 'A935-156')
            summary: New summary (optional)
            description: New description (optional)
            additional_fields: Extra field updates (optional)

        Returns:
            Dictionary with issue_key and url
        """
        try:
            # Build update fields
            fields = {}
            if summary:
                fields["summary"] = summary
            if description:
                fields["description"] = description
            if additional_fields:
                fields.update(self._normalize_field_values(additional_fields))

            # Update issue
            issue = self.client.issue(issue_key)
            issue.update(fields=fields)

            result = {
                "issue_key": issue_key,
                "url": f"{self.server_url}/browse/{issue_key}",
            }

            logger.info(f"Updated issue {issue_key}")
            return result
        except JIRAError as e:
            error_msg = self._translate_field_errors(e.text)
            logger.error(f"Failed to update issue {issue_key}: {error_msg}")
            raise ValueError(error_msg)

    def create_or_update_issue(
        self, issue_data: Dict[str, Any], progress_callback: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """
        Create new issue or update existing one based on issue_key presence.

        Args:
            issue_data: Dictionary with project_key, issue_type, summary, description, issue_key (optional)
            progress_callback: Optional callback for progress reporting

        Returns:
            Dictionary with issue_key, id, url, and status ('created' or 'updated')
        """
        issue_key = issue_data.get("issue_key")

        try:
            if issue_key:
                # Update existing issue
                result = self.update_issue(
                    issue_key=issue_key,
                    summary=issue_data.get("summary"),
                    description=issue_data.get("description"),
                    additional_fields=issue_data.get("additional_fields"),
                )
                result["status"] = "updated"

                if progress_callback:
                    progress_callback(issue_key=issue_key, status="updated")
            else:
                # Create new issue
                result = self.create_issue(
                    project_key=issue_data["project_key"],
                    issue_type=issue_data["issue_type"],
                    summary=issue_data["summary"],
                    description=issue_data.get("description"),
                    additional_fields=issue_data.get("additional_fields"),
                )
                result["status"] = "created"

                if progress_callback:
                    progress_callback(issue_key=result["issue_key"], status="created")

            return result
        except Exception as e:
            logger.error(f"Failed to create/update issue: {str(e)}")
            if progress_callback:
                progress_callback(error=str(e), status="failed")
            raise

    # ============================================================================
    # Bulk Import Operations
    # ============================================================================

    def bulk_import_issues(
        self, issues: List[Dict[str, Any]], progress_callback: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """
        Import multiple issues with continue-on-failure behavior.

        Args:
            issues: List of issue dictionaries
            progress_callback: Callback function(current, total, issue_key, status, error)

        Returns:
            Summary dictionary with counts and errors
        """
        total = len(issues)
        created = 0
        updated = 0
        failed = 0
        errors = []

        for idx, issue_data in enumerate(issues, start=1):
            try:
                result = self.create_or_update_issue(issue_data)

                if result["status"] == "created":
                    created += 1
                else:
                    updated += 1

                # Report progress
                if progress_callback:
                    progress_callback(
                        current=idx,
                        total=total,
                        issue_key=result["issue_key"],
                        status=result["status"],
                        url=result["url"],
                    )
            except Exception as e:
                failed += 1
                error_info = {
                    "summary": issue_data.get("summary", "Unknown"),
                    "error": str(e),
                }
                errors.append(error_info)

                logger.error(f"Failed to import issue {idx}/{total}: {str(e)}")

                # Report failure
                if progress_callback:
                    progress_callback(
                        current=idx, total=total, status="failed", error=str(e)
                    )

        summary = {
            "total": total,
            "created": created,
            "updated": updated,
            "failed": failed,
            "errors": errors,
        }

        logger.info(
            f"Bulk import completed: {created} created, {updated} updated, {failed} failed"
        )
        return summary

    # ============================================================================
    # Helper Methods
    # ============================================================================

    def _normalize_field_values(self, fields: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize field values to Jira API format.
        Handles complex field types (user, select, multi-select, teams, etc.)
        """
        normalized = {}

        for field_id, value in fields.items():
            if value is None:
                continue

            # Already in correct format (e.g., {"id": "123"} for teams/options)
            if isinstance(value, dict):
                normalized[field_id] = value
            # Handle multi-select fields
            elif isinstance(value, list):
                normalized[field_id] = [
                    {"value": v} if isinstance(v, str) else v for v in value
                ]
            # Simple string/number values
            else:
                normalized[field_id] = value

        return normalized

    def _translate_field_errors(self, error_text: str | None) -> str:
        """
        Translate Jira API field errors to user-friendly messages.
        Converts field IDs to field names where possible.
        """
        if not error_text:
            return "Unknown error occurred"

        # Basic translation patterns
        if "is required" in error_text.lower():
            # Extract field name from error
            return error_text.replace("Field", "Required field:")

        if "does not exist" in error_text.lower():
            return error_text.replace("does not exist", "not found in project")

        return error_text


# ============================================================================
# Convenience Factory Function
# ============================================================================


def create_jira_service(server_url: str, api_token: str) -> JiraService:
    """
    Factory function to create JiraService instance.

    Args:
        server_url: Jira server URL
        api_token: Jira API token

    Returns:
        JiraService instance
    """
    return JiraService(server_url, api_token)
