# Logging System for Crawl4AI MCP Server

This document describes the logging system implementation for the Crawl4AI MCP server.

## Overview

The system implements a comprehensive logging solution with the following features:

- Dual output to both console and log file
- Configurable log levels via environment variables
- Rotating log files with size limits
- Standard log formatting with timestamps and context information
- Error tracking with proper exception handling
- Consistent application-wide logging patterns

## Configuration

Logging is configured through environment variables in the `.env` file:

```
# Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
LOG_LEVEL=INFO

# Log filename 
LOG_FILENAME=crawl4mcp.log
```

## Log Directories

Log files are automatically stored in a `logs` directory created in the project root. The system will create this directory automatically if it doesn't exist.

## Log Format

Two different formats are used:

### Console Format
```
YYYY-MM-DD HH:MM:SS - module_name - LEVEL - Message text
```

### File Format (more detailed)
```
YYYY-MM-DD HH:MM:SS - module_name - LEVEL - filename:line_number - Message text
```

## Usage in Code

To use logging in any module:

1. Import the logger:
```python
from ..utils.logging_utils import get_logger

# Initialize logger with the current module name
logger = get_logger(__name__)
```

2. Use the logger with appropriate log levels:
```python
logger.debug("Detailed information for debugging")
logger.info("General information about normal operations")
logger.warning("Warning about potential issues")
logger.error("Error information for actual problems")
logger.critical("Critical issues requiring immediate attention")
```

3. For exceptions, use:
```python
try:
    # Some code that might raise an exception
    result = some_operation()
except Exception as e:
    logger.error(f"Operation failed: {e}", exc_info=True)
    # Handle the error
```

## Logging Best Practices

- Use appropriate log levels based on importance
- Include contextual information in log messages (e.g., file paths, query parameters, URLs)
- For expensive logging operations, check log level first:
  ```python
  if logger.isEnabledFor(logging.DEBUG):
      logger.debug(f"Expensive operation result: {expensive_operation()}")
  ```
- Include useful context but avoid logging sensitive information
- For RAG queries, log both the query parameters and results summary

## Initialization

Logging is initialized early in the application lifecycle in `mcp_setup.py` to ensure all components have access to properly configured logging.

## Log File Rotation

Log files are automatically rotated when they reach 10MB, with 5 backup files maintained.

## Changing Log Levels Dynamically

While the initial log level is set via environment variables, it can be changed dynamically using the `set_level()` function:

```python
from .utils.logging_utils import set_level

set_level("DEBUG")  # Temporarily increase verbosity
```

This affects all loggers and handlers consistently. 