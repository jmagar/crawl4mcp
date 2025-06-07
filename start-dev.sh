#!/bin/bash

# Start development environment for Crawl4MCP
echo "Starting Crawl4MCP Development Environment..."

# Function to cleanup background processes
cleanup() {
    echo "Stopping services..."
    kill $MCP_PID 2>/dev/null
    exit 0
}

# Set trap to cleanup on script exit
trap cleanup SIGINT SIGTERM

# Start MCP Server
echo "Starting MCP Server on port 9130..."
cd "$(dirname "$0")"
python -m src.app &
MCP_PID=$!

# Wait a moment for the server to start
sleep 3

echo ""
echo "🚀 Development environment started!"
echo "📡 MCP Server: http://localhost:9130"
echo ""
echo "Press Ctrl+C to stop the server"

# Wait for background processes
wait