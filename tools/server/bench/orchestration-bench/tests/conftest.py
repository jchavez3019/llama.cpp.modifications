import sys
from pathlib import Path

# The benchmark modules are plain scripts (no package), so make them importable.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
