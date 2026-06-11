from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocess import preprocess_swat

shapes = preprocess_swat(
    PROJECT_ROOT / "data" / "swat" / "raw",
    PROJECT_ROOT / "data" / "swat" / "processed",
)

print(shapes)