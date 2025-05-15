# Crawl4AI RAG MCP Server

## Why this project exists
This project exists to provide AI agents and AI coding assistants with advanced web crawling and RAG (Retrieval Augmented Generation) capabilities. It serves as a bridge between web content and AI systems, allowing them to search, extract, and utilize information from websites.

## What problems it solves
- Enables AI agents to access and process web content that's not in their training data
- Provides a structured way to crawl websites and extract meaningful content
- Stores content in a vector database (currently Qdrant) for semantic search capabilities
- Implements RAG to enhance AI responses with up-to-date information from the web

## How it should work
The server provides tools for:
1. Crawling single web pages
2. Smart crawling of websites (handling sitemaps, text files, or recursive crawling)
3. Retrieving available sources for query filtering
4. Performing RAG queries on the stored content

The system intelligently chunks content, generates embeddings (using a self-hosted model like BAAI/bge-large-en-v1.5 via a TEI server), stores them in Qdrant, and provides semantic search capabilities. 