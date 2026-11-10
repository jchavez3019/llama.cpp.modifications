import sys
from pathlib import Path

# The fit module is a plain script (no package), so make it importable.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
