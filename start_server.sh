#!/bin/bash

# CAD Document Chat Assistant - Startup Script
# This script starts the Flask server in a tmux session

SESSION_NAME="cad_server"

echo "╔════════════════════════════════════════════════════════════════════╗"
echo "║      🏗️  CAD DOCUMENT CHAT ASSISTANT - WEB INTERFACE           ║"
echo "╚════════════════════════════════════════════════════════════════════╝"
echo ""

# Check if tmux is installed
if ! command -v tmux &> /dev/null; then
    echo "❌ tmux is not installed!"
    echo "Please install tmux:"
    echo "  Ubuntu/Debian: sudo apt install tmux"
    echo "  CentOS/RHEL: sudo yum install tmux"
    echo "  macOS: brew install tmux"
    exit 1
fi

# Check if .env file exists
if [ ! -f .env ]; then
    echo "⚠️  WARNING: .env file not found!"
    echo ""
    echo "Please create a .env file with your API key:"
    echo "  GEMINI_API_KEY=your_api_key_here"
    echo ""
    echo "Get your API key from: https://makersuite.google.com/app/apikey"
    echo ""
    exit 1
fi

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed!"
    echo "Please install Python 3.8 or higher"
    exit 1
fi

# Check if requirements are installed
echo "📦 Checking dependencies..."
python3 -c "import flask" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "📥 Installing dependencies..."
    pip3 install -r requirements.txt
    if [ $? -ne 0 ]; then
        echo "❌ Failed to install dependencies"
        exit 1
    fi
else
    echo "✅ Dependencies are installed"
fi

echo ""

# Check if tmux session already exists
if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "⚠️  Tmux session '$SESSION_NAME' already exists!"
    echo ""
    echo "Options:"
    echo "  1. Attach to existing session: tmux attach -t $SESSION_NAME"
    echo "  2. Kill existing session: tmux kill-session -t $SESSION_NAME"
    echo "  3. Run this script again after killing the session"
    echo ""
    exit 1
fi

echo "╔════════════════════════════════════════════════════════════════════╗"
echo "║  🚀 Starting server in tmux session...                            ║"
echo "║                                                                    ║"
echo "║  📡 Server will be available at: http://localhost:5005            ║"
echo "║  🌐 Open your browser and navigate to the URL above               ║"
echo "║                                                                    ║"
echo "║  📺 Tmux session name: $SESSION_NAME                              ║"
echo "║                                                                    ║"
echo "║  Useful commands:                                                  ║"
echo "║    • View server logs: tmux attach -t $SESSION_NAME               ║"
echo "║    • Detach from tmux: Press Ctrl+B then D                        ║"
echo "║    • Stop server: tmux kill-session -t $SESSION_NAME              ║"
echo "╚════════════════════════════════════════════════════════════════════╝"
echo ""

# Create tmux session and start the Flask server
tmux new-session -d -s "$SESSION_NAME" "cd $(pwd) && python3 app.py"

echo "✅ Server started successfully in tmux session!"
echo ""
echo "To view server logs, run:"
echo "  tmux attach -t $SESSION_NAME"
echo ""

