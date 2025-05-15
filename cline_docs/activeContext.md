# Current Task: System Stabilization, Qdrant Warning Investigation & Feature Exploration

## What we're working on now
- Confirming the stability and performance of the Qdrant/TEI-based embedding and RAG system.
- Investigating the Qdrant client parsing warnings that occur at server startup.
- Considering and discussing potential new features to enhance the server's capabilities.

## Recent changes
- **Completed Migration:** Successfully migrated the vector database from Supabase to Qdrant.
- **Switched Embedding Source:** Changed from OpenAI API embeddings to a self-hosted TEI server providing `BAAI/bge-large-en-v1.5` embeddings.
- **Configuration Updates:** `pyproject.toml`, `.env` file, and `docker-compose.yml` updated for Qdrant and TEI.
- **Code Refactoring:** `src/utils.py` and `src/crawl4ai_mcp.py` were significantly refactored to use Qdrant and the TEI server, adopting asynchronous patterns.
- **Troubleshooting - Payload Errors:** Resolved "413 Payload Too Large" errors from the TEI server by:
    - Disabling the OpenAI-based contextual summarization feature.
    - Adjusting `smart_chunk_markdown` size, currently set to 750 characters.
- **Troubleshooting - MCP Server Error:** Addressed a `RuntimeError: Received request before initialization was complete` by ensuring sufficient server startup time before client interaction.

## Next steps
1.  **Address Qdrant Client Warnings:** Determine the cause of the Pydantic validation errors (`Error parsing server response for collection 'crawl4ai_mcp'`) during startup. This will likely involve checking Qdrant server and `qdrant-client` library versions for compatibility.
2.  **Comprehensive Testing:** Perform broader testing with diverse websites and query types to ensure overall system robustness.
3.  **Evaluate Chunking/Embedding Strategy:** Confirm that the 750-character chunking (without summaries) provides good RAG performance. Consider if client-side token counting or more sophisticated chunking would be beneficial if issues arise.
4.  **Feature Prioritization & Implementation:** Based on discussion, select and implement any desired new features (e.g., proxy support, PDF/screenshot capture, refined progress tracking for crawls).
5.  **Memory Bank Review:** Ensure all Memory Bank documents are fully up-to-date and accurate post-migration. 