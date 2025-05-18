FROM python:3.12-slim

ARG PORT=8051

WORKDIR /app
ENV PYTHONPATH /app

# Install uv
RUN pip install uv

# Install git
RUN apt-get update && apt-get install -y git

# Copy the MCP server files
COPY . .

# Install packages directly to the system (no virtual environment)
# Combining commands to reduce Docker layers
RUN uv pip install --system -e ".[visualization]" && \
    crawl4ai-setup && \
    python -m nltk.downloader stopwords

EXPOSE ${PORT}

# Command to run the MCP server
CMD ["python", "-m", "src.app"]