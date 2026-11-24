"""
DermAI Root Launcher
Runs the FastAPI backend from the root directory.
"""
import os
import sys
import subprocess

if __name__ == "__main__":
    backend_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "backend")
    os.chdir(backend_dir)
    sys.exit(subprocess.call([sys.executable, "run.py"]))
