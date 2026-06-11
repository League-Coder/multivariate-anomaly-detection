import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

cmd = [
    sys.executable,
    str(PROJECT_ROOT / "code" / "1_train_model.py"),
    "--machine", "swat",
    "--data-dir", "data/swat/processed",
    "--output-dir", "models/tranad/swat",
    "--epochs", "10",
]

print("Running:")
print(" ".join(cmd))

subprocess.run(cmd, check=True)