FROM python:3.12

ARG PORT=8051

WORKDIR /app
ENV PYTHONPATH /app

# Install uv
RUN pip install uv

# Install git
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# Copy the MCP server files
COPY . .

# Install packages directly to the system (no virtual environment)
RUN uv pip install --system -e "." && \
    crawl4ai-setup && \
    python -m nltk.downloader stopwords && \
    echo "Listing /usr/local/bin:" && \
    ls -l /usr/local/bin

EXPOSE ${PORT}

# Command to run the MCP server
CMD ["/usr/local/bin/python3", "-m", "src.app"]