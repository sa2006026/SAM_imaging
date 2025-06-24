#!/usr/bin/env python3
"""Startup script for SAM Droplet segmentation server."""

import sys
import subprocess
import argparse
from pathlib import Path

def run_flask_dev():
    """Run Flask development server."""
    print("Starting Flask development server...")
    subprocess.run([sys.executable, "app.py"])

def run_waitress():
    """Run Waitress server (Windows compatible)."""
    print("Starting Waitress server...")
    subprocess.run([sys.executable, "-m", "waitress", "--host=0.0.0.0", "--port=9487", "app:app"])

def run_gunicorn():
    """Run Gunicorn server (Linux/Mac)."""
    print("Starting Gunicorn server...")
    subprocess.run(["gunicorn", "--bind", "0.0.0.0:9487", "--workers", "1", "--threads", "2", "--timeout", "300", "app:app"])

def install_dependencies():
    """Install required dependencies."""
    print("Installing dependencies...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements_server.txt"])

def main():
    parser = argparse.ArgumentParser(description="Start SAM Droplet segmentation server")
    parser.add_argument("--mode", choices=["dev", "waitress", "gunicorn"], default="dev",
                       help="Server mode: dev (Flask), waitress (Windows prod), or gunicorn (Linux/Mac prod)")
    parser.add_argument("--install", action="store_true",
                       help="Install dependencies before starting")
    
    args = parser.parse_args()
    
    # Change to script directory
    script_dir = Path(__file__).parent
    import os
    os.chdir(script_dir)
    
    if args.install:
        install_dependencies()
    
    if args.mode == "dev":
        run_flask_dev()
    elif args.mode == "waitress":
        run_waitress()
    elif args.mode == "gunicorn":
        run_gunicorn()

if __name__ == "__main__":
    main() 