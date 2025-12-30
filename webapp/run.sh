#!/usr/bin/env bash
# Simple helper to run the Flask app locally
export FLASK_APP=webapp.app
export FLASK_ENV=development
python webapp/app.py
