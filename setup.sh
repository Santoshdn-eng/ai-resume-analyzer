#!/bin/bash
# setup.sh — one-shot environment setup for AI Resume Analyser and Job Match Platform
# Run this from the project root: bash setup.sh

set -e

echo "Setting up AI Resume Analyser and Job Match Platform..."

# 1. Select Python binary (prefer 3.12 or 3.11 with pre-built binary wheels)
if command -v python3.12 &> /dev/null; then
  PYTHON_BIN="python3.12"
elif command -v python3.11 &> /dev/null; then
  PYTHON_BIN="python3.11"
else
  PYTHON_BIN="python3"
fi

echo "Using Python interpreter: $PYTHON_BIN"

rm -rf venv
$PYTHON_BIN -m venv venv
source venv/bin/activate

# 2. Install backend dependencies
pip install --upgrade pip setuptools wheel
pip install -r backend/requirements.txt

# 3. Download the spaCy language model
python -m spacy download en_core_web_sm

# 4. Init git if not already present
if [ ! -d ".git" ]; then
  git init
  cat > .gitignore << 'EOF'
venv/
__pycache__/
*.pyc
backend/uploads/
.DS_Store
EOF
fi

echo ""
echo "Setup complete."
echo "Run the app with:      source venv/bin/activate && cd backend && python app.py"
echo "Or with Docker:        docker compose up --build"
echo ""
