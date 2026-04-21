"""
conftest.py — permite `pytest` ejecutarse desde la raíz del proyecto
sin instalar el paquete.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
