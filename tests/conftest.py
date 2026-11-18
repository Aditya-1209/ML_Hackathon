import sys
from pathlib import Path

# Add src/ to sys.path so test discovery and IDEs resolve pathsnap_ml seamlessly
src_path = str(Path(__file__).resolve().parent.parent / "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)
