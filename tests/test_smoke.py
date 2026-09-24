import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# basic test
# 1. checks that the basic project structure and Python files are valid before running more complicated tests
def test_python_files_compile():
    """Catch Python syntax errors."""
    python_files = list(ROOT.glob("*.py"))
    
    src_dir = ROOT / "src"
    if src_dir.exists():
        python_files.extend(src_dir.rglob("*.py"))

    assert python_files, "No Python source files found"

    for path in python_files:
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

# 2. This checks that you have a non-empty requirements.txt
def test_requirements_exists():
    """Check that requirements.txt exists and is not empty."""
    requirements = ROOT / "requirements.txt"
    assert requirements.exists(), "requirements.txt is missing"
    assert requirements.read_text(encoding="utf-8").strip(), \
        "requirements.txt is empty"

def test_third_party_dependencies_import():
    """Verify required third-party packages are installed."""
    from google import genai
    from google.genai import types
    from google.genai.errors import ServerError

    import cv2
    import ultralytics
    import uuid
    import json
    import asyncio
    import logging

        
def test_application_modules_import():
    """Verify application modules can be imported."""
    from src.tools import trigger_alert
    import src.methods
    from src.local_validator import LocalEdgeValidator
    from src.audit_logger_pg import PGAuditLogger