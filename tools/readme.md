# Development Tools

This folder contains utility scripts for testing and validating sticky2jira functionality.

## Available Tools

### `analyze_test_images.py`

**Purpose:** Validate OCR parameter configurations against test images.

**When to use:**

- Testing new sticky note images before deploying to production
- Troubleshooting detection issues (under/over detection)
- Finding optimal parameters for different sticky note styles/colors
- Before deploying to new teams with different sticky note preferences

**Usage:**

```bash
# Activate virtual environment first
.\.venv\Scripts\activate

# Run the analysis tool
python tools\analyze_test_images.py
```

**Output:**

- Console: Detailed results comparing different parameter configurations
- `test_output/ocr_parameter_analysis.json`: Machine-readable results

**Configuration:**
Edit the `test_cases` list at the bottom of the script to add your own test images:

```python
test_cases = [
    {
        "path": "test_images/your_image.png",
        "expected": 10,  # Expected number of stickies
        "description": "Your test description"
    }
]
```

**Interpretation:**

- ✅ = Configuration achieves 100% detection (matches expected count)
- ❌ = Mismatch (under/over detection)
- ℹ️ = No expected count provided (informational only)

If current production settings don't work for your images, the tool will suggest which alternative configuration performs better.

## Adding New Tools

Place utility scripts in this folder to keep the project root clean. Update this README when adding new tools.

**Naming convention:**

- Use descriptive names: `validate_*.py`, `analyze_*.py`, `debug_*.py`
- Include comprehensive docstring explaining purpose and usage
- Add command-line help text for user-facing tools
