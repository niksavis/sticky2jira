# sticky2jira - Image-to-Jira Issue Converter

**Created:** November 21, 2025  
**Last Updated:** November 21, 2025  
**Project Location:** `C:\Development\sticky2jira\`  
**Status:** Phase 1 MVP Complete ✅ | Phase 2 In Progress

## Project Overview

A browser-based local application that extracts text from sticky note board images using OCR and creates/updates Jira issues—designed for non-technical users with zero cloud dependencies and no manual JSON editing.

## Current Implementation Status

### ✅ Phase 1 MVP - COMPLETE

- Core infrastructure with automated setup (install.bat, launch.bat)
- Flask application with WebSocket (SocketIO) support
- Jira integration with field defaults per issue type (Story, Task, Bug)
- OCR pipeline using PaddleOCR (100% local processing)
- Browser UI with 7-tab wizard workflow
- Issue review with inline editing
- Import to Jira with real-time progress
- **Update vs Create logic**: Issues with `issue_key` are updated, not recreated
- **Issue tracking**: `issue_key` displayed in Issue Review tab with Jira link
- Encrypted credential storage (Fernet AES-128)
- Multi-type field defaults configuration UI

### 🚧 Known Limitations (To Be Addressed in Phase 2+)

- No manual region drawing (OCR detection only)
- No preprocessing presets (auto-detect only)
- No Quick Fix Grid for bulk corrections
- No card view (table view only)
- No bulk operations (select multiple → edit)
- No session export/import (JSON)
- No import history viewer
- No "Add Images" mode (new session only)
- No visual indicators (🆕/✏️/✅/⚠️) for issue status
- No CSV error export
- No retry failed imports

## Core Requirements

- **100% Local Processing**: No cloud services - all OCR happens locally (confidential data)
- **Zero Technical Knowledge**: No JSON editing, no command-line complexity
- **Prerequisites**: Python 3.8+ runtime and browser only
- **Distribution**: Git repo + automated setup script (no packaged .exe initially)

## User Workflow

### First-Time Setup

1. Clone repo → run `install.bat` (one command)
2. Script creates venv, installs dependencies including PaddleOCR (pure Python, no external binaries)
3. Browser opens automatically to `http://localhost:5000`
4. **Setup Tab**: Enter Jira credentials (URL, token, project_key)
   - Click "Connect & Fetch Fields"
   - System fetches field metadata from Jira
   - Set default values for mandatory custom fields
   - Configuration saved to SQLite

### Image-to-Issues Flow

1. **Upload Tab**: Drag/drop sticky note board image (auto-resizes to max 2000px)

2. **OCR Review Tab**:
   - Background processing with real-time progress
   - Canvas overlay shows detected sticky regions (color-coded rectangles)
   - **Adjust Detection Panel**: HSV tolerance, min/max region size, proximity threshold sliders
   - **Manual Correction**: Click-to-edit text, draw rectangles for missed stickies
   - **Quick Fix Grid**: Thumbnail + textarea for rapid bulk corrections
   - "Re-run OCR" button

3. **Mapping Tab**: Assign issue types (Story/Task/Bug) to detected sticky colors
   - Optional: Enable description mode (linked stickies become descriptions)

4. **Issue Review Tab**:
   - **Table View** (DataTables.js) or **Card View** (Bootstrap cards)
   - Inline editing: summary, description, project_key, issuetype
   - **Bulk Operations**: Select rows → set project/type, delete
   - **Visual Indicators**: 🆕 new, ✏️ update, ✅ valid, ⚠️ missing fields
   - Multi-project support with grouped rows

5. **Import Tab**:
   - Click "Import to Jira" → real-time progress
   - **Critical**: Issues with issue_key are updated, not created (prevents duplicates)
   - Results with clickable Jira links
   - Success/error summary

6. **Multi-Image Workflow**:
   - "New Import": clears session, starts fresh
   - "Add Images": accumulates issues from multiple images
   - Session export/import as JSON

## Technology Stack

### Backend

- Flask + Flask-SocketIO (WebSocket for real-time updates)
- Python 3.8+
- SQLite (session.db)
- Threading (background OCR/import)

### OCR & Image Processing

- PaddleOCR 2.7+ (pure Python OCR engine)
- PaddlePaddle 2.6+ (deep learning framework)
- OpenCV (cv2)
- Pillow
- NumPy

### Frontend

- Vanilla JavaScript
- Bootstrap 5 (self-hosted)
- DataTables.js (self-hosted)
- HTML5 Canvas
- **No CDN dependencies**

### Jira Integration

- jira library (Python)

## Project Structure

```text
C:\Development\sticky2jira\
├── plan.md (this file)
├── README.md (user-facing setup instructions)
├── install.bat (setup automation)
├── download_libs.bat (frontend library downloader)
├── launch.bat (auto-generated startup script, git-ignored)
├── app.py (Flask main + SocketIO)
├── requirements.in (source dependencies)
├── requirements.txt (compiled, git-tracked)
├── .gitignore
├── .venv/ (Python virtual environment, git-ignored)
├── .encryption_key (Fernet key for API tokens, git-ignored)
├── bin/ (git-ignored, reserved for future use)
├── services/
│   ├── __init__.py
│   ├── crypto_utils.py (AES-128 encryption for API tokens)
│   ├── jira_service.py (Jira API integration)
│   ├── ocr_service.py (OpenCV + PaddleOCR)
│   └── session_manager.py (SQLite CRUD)
├── templates/
│   └── index.html (SPA with all tabs)
├── static/
│   ├── libs/ (git-ignored, downloaded by download_libs.bat)
│   │   ├── bootstrap-5.3.0/
│   │   ├── datatables-1.13/
│   │   ├── jquery-3.7.0/
│   │   └── socket.io-4.5.4/
│   ├── css/
│   │   └── app.css
│   ├── js/
│   │   └── app.js (all UI logic in single file)
│   └── favicon.ico
├── uploads/ (temporary, git-ignored)
├── session.db (SQLite, git-ignored)
├── jira_templates/ (generated field metadata, git-ignored)
├── app.log (git-ignored)
└── errors.log (git-ignored)
```

**`.gitignore` Contents:**

Standard Python .gitignore (created via VS Code) with project-specific additions:

```text
# Project-specific (auto-appended by install.bat)
bin/
.encryption_key
uploads/
session.db
jira_templates/
app.log
errors.log
launch.bat
static/libs/
.github/
```

**Environment Requirements:**

- Python 3.8+ (verify with `python --version`)
- Windows OS (PowerShell 5.1+)
- Internet connection for initial setup (Python package downloads)
- Modern browser (Chrome, Edge, Firefox)

## Implementation Plan

### 1. Core Infrastructure

Create project structure with automated setup implementing `requirements.in` listing core dependencies (Flask, Flask-SocketIO, paddleocr, paddlepaddle, opencv-python, pillow, jira, numpy), `install.bat` that creates Python virtual environment (`.venv`) using `python -m venv .venv`, activates it, installs pip-tools, compiles `requirements.txt` via `pip-compile requirements.in`, installs dependencies from `requirements.txt` (including PaddleOCR and PaddlePaddle - pure Python, no external binaries needed), writes config to `.env`, appends project-specific entries to `.gitignore` (bin/, .env, uploads/, session.db, jira_templates/), creates `launch.bat` (activates `.venv` and starts Flask server), and displays success message.

### 2. Flask Application with WebSocket

Build `app.py` serving SPA from `templates/index.html`, configuring Flask (debug=False for production, host='127.0.0.1', port=5000), implementing REST endpoints (`POST /api/jira/connect`, `POST /api/ocr/upload`, `POST /api/ocr/process`, `POST /api/mapping/save`, `GET /api/issues`, `POST /api/issues/bulk-edit`, `POST /api/import/start`), SocketIO event handlers (`ocr_progress`, `ocr_complete`, `import_progress`, `import_complete`), SQLite session management via `services/session_manager.py`, logging configuration (file and console output), CORS handling for localhost, and auto-launching browser to `http://localhost:5000`.

### 3. Jira Integration

Create `services/jira_service.py` with JIRA client initialization/authentication, template generation logic with field discovery, field merging with defaults, create/update operations with issue_key detection, progress callback system emitting SocketIO events, and issue_key persistence to `session.db`.

### 4. OCR Pipeline

Implement `services/ocr_service.py` using OpenCV HSV color segmentation (configurable thresholds: HSV tolerance=20, min_size=50, proximity=100 defaults), k-means dominant color extraction, PaddleOCR text extraction with confidence scores, spatial proximity analysis for description linking, background threading with SocketIO progress events `{percent, current_region, total_regions, preview_url}`, cancellation support, and return JSON `[{id, color_hex, text, bbox, confidence, linked_to[]}]`.

**OCR Accuracy Optimization:**

- Default auto-detect with medium sensitivity (immediate results)
- Prominent "Adjust Detection" collapsible panel (visual indicator when non-default)
- Quick Fix Grid as fastest correction path (thumbnail + textarea)
- Consider preprocessing presets (Low Light, High Contrast, Handwriting) as one-click alternatives

### 5. Browser UI - Wizard Workflow

Build interface using self-hosted Bootstrap 5.3.0 and DataTables.js 1.13.x in `static/libs/`, implementing:

- Tabbed navigation (Setup → Upload → OCR Review → Mapping → Issue Review → Import → Results)
- Jira connection form with test button
- Drag-drop image upload with Canvas API resize to max 2000px
- Interactive canvas overlay (color-coded rectangles, click-to-edit, manual region drawing)
- "Adjust Detection" panel with sliders and live preview
- Quick Fix Grid for bulk corrections
- Color-to-issuetype mapping wizard

### 6. Issue Review with Multi-Project Support

Build review interface with DataTables.js inline cell editing, grouped expandable rows by project_key, bulk-edit toolbar (checkbox selection → set project/type dropdowns, delete action), visual indicators (🆕 new, ✏️ update, ✅ valid, ⚠️ missing fields from `jira_templates/`), column filters, toggle between table and card views, validation against templates.

**Multi-Project UX Enhancements:**

- Smart defaults (pre-fill project_key from Jira settings)
- Visual project grouping (color-coded rows/expandable sections)
- "Select All in Project" checkbox for group operations
- Project statistics sidebar (issue count per project/type)
- Undo capability for bulk edits before import

### 7. Session Management

Implement `services/session_manager.py` with SQLite schema:

**issues** table: id, project_key, issuetype, summary, description, issue_key, color_hex, image_filename, bbox_json, ocr_confidence, created_at, updated_at (indices on: project_key, issue_key, image_filename)

**color_mappings** table: color_hex, issuetype, project_key (unique constraint on: color_hex, project_key)

**jira_settings** table: id, server_url, username, api_token (BLOB, encrypted), default_project_key, created_at, updated_at (singleton table with id=1 check constraint)

**import_history** table: timestamp, project_key, created_count, updated_count, error_log_json (index on: timestamp DESC)

Implement "New Import" (TRUNCATE issues) vs "Add Images" (INSERT with tracking), multi-project support, session export/import JSON, and endpoints `/api/session/clear`, `/api/session/export`, `/api/session/import`.

### 8. Testing & Validation

Document image requirements in README (min 1024px, PNG/JPG, max 5MB), provide visual guidelines for description linking, test end-to-end with real Jira instance, verify encryption of API tokens, and validate cross-platform compatibility (Windows focus, but encryption is platform-agnostic).

## API Endpoints

### REST

- `POST /api/jira/connect` - Test connection, fetch issue types
- `POST /api/jira/fetch-templates` - Generate field metadata
- `POST /api/jira/set-defaults` - Save field defaults
- `POST /api/ocr/upload` - Upload image, get preview
- `POST /api/ocr/process` - Start OCR with settings
- `POST /api/ocr/cancel` - Cancel OCR task
- `POST /api/ocr/manual-region` - Force OCR on user-drawn rectangle
- `POST /api/mapping/save` - Save color→issuetype mappings
- `GET /api/mapping/list` - Retrieve mappings
- `GET /api/issues` - Get all issues (grouped by project)
- `POST /api/issues/add` - Add single issue
- `PUT /api/issues/:id` - Update issue
- `DELETE /api/issues/:id` - Delete issue
- `POST /api/issues/bulk-edit` - Bulk update selected
- `POST /api/import/start` - Start import
- `POST /api/import/cancel` - Cancel import
- `POST /api/session/clear` - Clear all issues
- `GET /api/session/export` - Download JSON
- `POST /api/session/import` - Upload JSON

### WebSocket/SocketIO Events

**Client → Server:**

```javascript
socket.emit('ocr_start', {image_id, settings});
socket.emit('ocr_cancel', {task_id});
socket.emit('import_start', {});
socket.emit('import_cancel', {import_id});
```

**Server → Client:**

```javascript
socket.on('ocr_progress', {percent, current_region, total_regions, preview_url});
socket.on('ocr_complete', {regions: [{id, color_hex, text, bbox, confidence, linked_to[]}, ...]});
socket.on('ocr_error', {error});
socket.on('import_progress', {current, total, project_key, issue_key, status, message, jira_link});
socket.on('import_complete', {created_count, updated_count, errors: []});
socket.on('import_error', {error});
```

## Error Handling & User Feedback

**Implementation Requirements:**

- Translate Jira API errors to user-friendly messages (field ID → field name using templates)
- Inline validation tooltips on Issue Review tab
- Separate recoverable vs fatal errors in import results
- Continue-on-failure with error collection
- CSV export of errors for debugging
- "Retry Failed" button to re-attempt only errored issues
- Test on clean Windows machine without dev tools

## Key Design Decisions

1. **Local-first**: No cloud APIs (confidentiality requirement)
2. **No JSON editing**: All configuration via UI forms
3. **Browser-based**: Leverage existing browser (Flask over PyQt)
4. **SQLite persistence**: Enable multi-image accumulation
5. **SocketIO real-time**: Feedback for long operations
6. **Reuse proven code**: Apply established patterns for reliability
7. **Self-contained**: Git clone + install.bat (no complex deployment)
8. **Progressive enhancement**: Basic flow first, advanced features later

---

## Next Steps for Implementation

1. **Initial Setup**: Run through step 1 (Core Infrastructure) to create folder structure, install.bat, requirements.in, and compiled requirements.txt
2. **Backend Foundation**: Implement steps 2-3 (Flask app + Jira service) to establish server and Jira connectivity
3. **OCR Pipeline**: Complete step 4 to enable image processing capabilities
4. **Frontend Development**: Build UI components in steps 5-6 for user interaction
5. **Integration**: Connect frontend to backend via REST API and SocketIO
6. **Testing**: Create test images and validate end-to-end workflow with real Jira instance
7. **Polish**: Implement error handling, validation, and UX enhancements

**Development Order Priority:**

- **Phase 1 (MVP)**: ✅ **COMPLETE** - Steps 1-4, all 7 tabs functional, Jira connection, OCR processing, color mapping, issue review, import with progress, encrypted credentials, multi-type field defaults, update vs create logic
- **Phase 2 (Current - OCR Enhancements)**: Manual region drawing, preprocessing presets, Quick Fix Grid, OCR parameter tuning UI
- **Phase 3 (Multi-Project & Bulk Ops)**: Bulk operations (select rows → edit/delete), card view toggle, project grouping, undo capability
- **Phase 4 (Session Management)**: Export/import JSON, import history viewer, "Add Images" mode (accumulate from multiple images)
- **Phase 5 (UI Polish)**: Visual indicators (🆕 new/✏️ update/✅ valid/⚠️ errors), inline validation tooltips, CSV error export
- **Phase 6 (Advanced Features)**: Retry failed imports, preprocessing presets, performance tuning, field validation against templates

**Next Immediate Priorities (Phase 2):**

1. Manual region drawing on canvas (missed stickies)
2. Quick Fix Grid (thumbnail + textarea for rapid corrections)
3. Preprocessing presets (Low Light, High Contrast, Handwriting)
4. OCR parameter sliders UI (HSV tolerance, min/max size, proximity)
5. Re-run OCR button with adjusted settings

**Additional Files Created:**

- ✅ `LICENSE` (internal use license)
- ✅ `README.md` (concise setup guide with quickstart, troubleshooting, security notes)
- ✅ `.github/copilot-instructions.md` (AI agent development guidelines, git-ignored)
