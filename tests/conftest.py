# tests/conftest.py
# Pytest configuration: ensures the project root is on sys.path so that
# `import src` always works regardless of how pytest is invoked.

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
