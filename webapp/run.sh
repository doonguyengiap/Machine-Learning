#!/usr/bin/env bash
# Simple helper to run the Flask app locally

# Ensure we are in the project root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# Activate virtual environment
if [ -d ".venv" ]; then
    source .venv/bin/activate
else
    echo "Warning: .venv not found. Running with system python."
fi

export FLASK_APP=webapp.app
export FLASK_ENV=development
export PYTHONPATH=$PROJECT_ROOT

echo "Starting Flask app from $PROJECT_ROOT..."
python webapp/app.py
