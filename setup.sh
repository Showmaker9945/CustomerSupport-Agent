#!/bin/bash
# Quick start script for CustomerSupport-Agent

set -e

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║     CustomerSupport-Agent - Quick Start Script               ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# Check Python version
echo "🔍 Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "✓ Python $python_version found"
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
    echo "✓ Virtual environment created"
    echo ""
fi

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source venv/bin/activate
echo "✓ Virtual environment activated"
echo ""

# Install dependencies
if [ ! -f ".deps_installed" ]; then
    echo "📥 Installing dependencies..."
    pip install -q --upgrade pip
    pip install -q -r requirements.txt
    echo "✓ Dependencies installed"
    touch .deps_installed
    echo ""
else
    echo "✓ Dependencies already installed"
    echo ""
fi

# Download NLP data
echo "📚 Downloading NLP data..."
python -m textblob.download_corpora > /dev/null 2>&1 || echo "  (TextBlob corpora already downloaded)"
echo "✓ NLP data ready"
echo ""

# Check .env file
if [ ! -f ".env" ]; then
    echo "⚙️  Creating .env file..."
    cp .env.example .env
    echo ""
    echo "⚠️  IMPORTANT: Edit .env and add your OPENAI_API_KEY"
    echo "   nano .env"
    echo ""
    echo "   Then run this script again."
    exit 1
fi

# Check if API key is set
if grep -q "your_openai_api_key_here" .env || grep -q "OPENAI_API_KEY=.*$" .env; then
    echo "⚠️  Please set your OPENAI_API_KEY in .env file:"
    echo "   OPENAI_API_KEY=sk-your-actual-key-here"
    echo ""
    echo "   Then run this script again."
    exit 1
fi

echo "✓ Configuration looks good!"
echo ""

# Create data directories
echo "📁 Creating data directories..."
mkdir -p data/knowledge_base data/chroma_db data/user_memory
echo "✓ Directories created"
echo ""

# Run tests (optional)
read -p "Run tests? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "🧪 Running tests..."
    pytest tests/unit/ -v --tb=no --no-header -q || true
    echo ""
fi

# Ask how to start
echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "🚀 Setup complete! Choose how to start the server:"
echo ""
echo "  1. Development mode (auto-reload):"
echo "     uvicorn src.api.main:app --reload"
echo ""
echo "  2. Production mode:"
echo "     uvicorn src.api.main:app --host 0.0.0.0 --port 8000"
echo ""
echo "  3. With Docker:"
echo "     docker-compose up"
echo ""
echo "  4. Run tests:"
echo "     pytest"
echo ""
echo "═══════════════════════════════════════════════════════════════"
echo ""
echo "📖 API will be available at:"
echo "   - REST: http://localhost:8000"
echo "   - WebSocket: ws://localhost:8000/ws/chat/{user_id}"
echo "   - Docs: http://localhost:8000/docs"
echo ""
