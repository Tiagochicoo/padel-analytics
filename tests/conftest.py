"""Shared test fixtures."""
import sys
from pathlib import Path

# Ensure project root is on sys.path for `from src...` imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
