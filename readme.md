# <img src="static/favicon.ico" width="30" height="30" alt="Sticky2Jira" style="vertical-align: text-bottom;"> Sticky2Jira

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Version](https://img.shields.io/badge/version-1.1.0-green.svg)](https://github.com/niksavis/sticky2jira)

Browser-based local application that extracts text from sticky note images using OCR and creates/updates Jira issues. **100% local processing - no cloud APIs.**

## Features

- **100% Local Processing** - No cloud APIs, all OCR processing happens on your machine using PaddleOCR (pure Python)
- **11-Color Detection** - Automatically detects red, orange, yellow, lime, green, cyan, blue, violet, pink, gray, and black sticky notes
- **Spatial Linking** - Links description stickies to nearby summary stickies
- **Jira Integration** - Creates new issues or updates existing ones (prevents duplicates)
- **Manual Issue Creation** - Add issues during review without OCR using the "Add Issue" button in the Issue Review tab
- **Mobile Responsive** - Optimized card-based layout for mobile devices with touch-friendly controls
- **Interactive UI** - Step-by-step wizard interface with real-time progress and autocomplete inputs
- **Secure** - API tokens encrypted with AES-128, stored locally in SQLite
- **Offline Capable** - All frontend assets self-hosted, works without internet after setup

## Requirements

- **Python 3.8+** - Required for Flask backend
- **Windows OS** - Batch scripts designed for Windows
- **Internet connection** - Only needed for initial setup (downloading Python packages) and Jira import
- **Jira Access** - Valid Jira URL and API token required for importing issues (OCR works offline)

## Installation

1. **Clone/download this repository** to your local machine

2. **Run the installation script:**

   ```cmd
   install.bat
   ```

   This will:
   - Create Python virtual environment (`.venv`)
   - Install all Python dependencies including PaddleOCR (pure Python, no binaries)
   - Download frontend libraries (Bootstrap, jQuery, DataTables, Socket.IO)

3. **Wait for installation to complete** (may take 5-10 minutes for PaddleOCR and dependencies)

4. **Run the application:**

   ```cmd
   launch.bat
   ```

   Browser opens automatically to `http://localhost:5000`

## Quick Start

1. **Upload**: Drop sticky note image → OCR runs automatically
2. **OCR Review**: View detected regions with confidence scores → Proceed to Mapping
3. **Mapping**: Assign sticky colors to issue types (Story/Bug/Task) → Proceed to Issue Review
4. **Issue Review**: Review/edit OCR-detected issues → Use "Add Issue" button to create additional manual issues
5. **Import**: Click "Import" button to send all issues to Jira
6. **Results**: View import summary with clickable Jira issue links

**Setup:** Click ⚙️ icon in header → Enter Jira URL and API token → Test → Save  
**Tips:** 
- Use "Add Issue" button in Issue Review tab to create issues without OCR (marked with light blue background)
- Issues with `issue_key` are updated (no duplicates)
- Mobile users see optimized card layout with visual editing indicators (yellow background = editable fields)
- Click "New Session" to clear all data and start fresh

## Advanced Configuration

**OCR Parameters** (`services/ocr_service.py`):

```python
DEFAULT_HSV_TOLERANCE = 20   # Color sensitivity
DEFAULT_MIN_SIZE = 1500      # Min sticky size (pixels)
DEFAULT_PROXIMITY = 100      # Text clustering distance
```

**Server Settings** (optional `.env` file):

By default, the app runs on `http://127.0.0.1:5000`. To customize Flask settings, create a `.env` file in the project root:

```bash
FLASK_HOST=127.0.0.1
FLASK_PORT=5000       # Change if port 5000 is in use
FLASK_DEBUG=False
```

*Note: `.env` file is optional - the app works without it using default values.*

**Supported Colors:** Red, orange, yellow, lime, green, cyan, blue, violet, pink, gray, black (11 total)

## Project Structure

```text
sticky2jira/
├── app.py                      # Flask server + SocketIO
├── __version__.py              # Version info (1.0.0)
├── install.bat                 # One-command setup
├── services/                   # Business logic
│   ├── ocr_service.py          # PaddleOCR + OpenCV (11 colors)
│   ├── jira_service.py         # Jira API integration
│   ├── session_manager.py      # SQLite operations
│   └── crypto_utils.py         # AES-128 encryption
├── templates/index.html        # 7-tab wizard UI
├── static/                     # Frontend assets (self-hosted)
│   ├── js/app.js               # Vanilla JS + SocketIO
│   ├── css/app.css             # Custom styles
│   └── libs/                   # Bootstrap, jQuery, DataTables
├── tests/                      # Pytest suite (15 tests)
├── tools/                      # Development utilities
└── test_images/                # OCR validation images
```

## Troubleshooting

| Issue              | Solution                                                                   |
| ------------------ | -------------------------------------------------------------------------- |
| **Blank page**     | Run `download_libs.bat` to get Bootstrap/jQuery/DataTables                 |
| **OCR errors**     | Reinstall: `.venv\Scripts\activate` then `pip install --upgrade paddleocr` |
| **Poor detection** | Use higher resolution images (300+ DPI), good lighting                     |
| **Jira fails**     | Check URL format: `https://company.atlassian.net` (no trailing slash)      |
| **Port conflict**  | Create `.env` with `FLASK_PORT=5001`                                       |
| **Duplicates**     | Click "New Import" to clear session before new images                      |

## Security & Privacy

- **Encrypted credentials**: Jira API tokens encrypted with AES-128 Fernet, key stored in `.encryption_key` (git-ignored)
- **100% local processing**: All OCR runs on your machine with PaddleOCR (pure Python, no external binaries or cloud APIs)
- **Localhost only**: Flask server binds to `127.0.0.1` (not accessible from network)
- **Session data**: Stored in `session.db` SQLite database (git-ignored)

## Technical Stack

- **Backend**: Flask + Flask-SocketIO, SQLite, Python 3.8+
- **OCR**: PaddleOCR 2.7+ (pure Python), OpenCV, NumPy
- **Frontend**: Vanilla JS, Bootstrap 5.3.0, DataTables.js, Socket.IO (all self-hosted)
- **Jira**: jira Python library
- **Security**: cryptography (Fernet/AES-128 encryption)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) - Pure Python OCR engine
- [Flask](https://flask.palletsprojects.com/) - Web framework
- [Bootstrap](https://getbootstrap.com/) - Frontend framework

---

**Version:** 1.1.0 | **Status:** Production Ready ✅  
**Last Updated:** November 22, 2025 | **License:** MIT
