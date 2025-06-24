"""
Logging utility functions for the Crawl4AI MCP server.
"""
import os
import sys
import logging
from pathlib import Path
from logging.handlers import RotatingFileHandler
from typing import List, Optional
import asyncio

# Import RichHandler for pretty console logging
from rich.logging import RichHandler

def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with the specified name.
    
    Args:
        name: Name of the logger, typically __name__ of the calling module
        
    Returns:
        Configured logger instance
    """
    return logging.getLogger(name)

def setup_logging() -> None:
    """
    Set up logging configuration based on environment variables.
    Logs to both console (using RichHandler) and a rotating file.
    
    Environment variables used:
    - LOG_LEVEL: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    - LOG_FILENAME: Name of the log file
    """
    log_level_str = os.getenv("LOG_LEVEL", "INFO").upper()
    log_levels = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL
    }
    log_level = log_levels.get(log_level_str, logging.INFO)

    # Create base logger
    # Note: It's common to configure the root logger, or configure specific loggers.
    # Here, we are configuring the root logger. If you want to restrict rich formatting
    # to only your application's loggers, you might get your app's top-level logger instead
    # and add handlers to it, rather than logging.basicConfig or logging.getLogger().addHandler.
    # For simplicity and broad effect, we'll configure the root logger's handlers.
    
    # Define a standard formatter for the file logger
    file_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)"
    )

    # Create a directory for logs if it doesn't exist
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    log_filename = os.getenv("LOG_FILENAME", "crawl4mcp.log")
    log_file_path = log_dir / log_filename

    # File Handler (Rotating)
    file_handler = RotatingFileHandler(
        log_file_path, maxBytes=10*1024*1024, backupCount=5  # 10 MB per file, 5 backups
    )
    file_handler.setFormatter(file_formatter)
    file_handler.setLevel(log_level) # Set level for file handler

    # Console Handler (RichHandler for pretty output)
    # RichHandler provides its own excellent default formatting and coloring.
    # You can customize it further if needed (e.g., rich_tracebacks=True, keywords)
    console_handler = RichHandler(
        level=log_level, 
        show_time=True, 
        show_level=True, 
        show_path=True, # Shows filename:lineno
        markup=True # Enable Rich's markup for log messages
    )
    # No need to set a formatter for RichHandler unless you want to override its defaults significantly.
    # console_handler.setFormatter(logging.Formatter("%(message)s")) # Example if you wanted minimal rich output

    # Get the root logger and add handlers
    # It's important to manage handlers carefully to avoid duplicate logs if setup_logging is called multiple times
    # or if libraries also configure the root logger.
    root_logger = logging.getLogger() # Get the root logger
    
    # Clear existing handlers from the root logger to avoid duplicates if setup_logging is called again
    # or if other parts of the application/libraries might add handlers.
    if root_logger.hasHandlers():
        root_logger.handlers.clear()
        
    root_logger.setLevel(log_level) # Set level on the root logger itself
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)

    # Optionally, quiet down other verbose loggers if needed
    # logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    # logging.getLogger("uvicorn.error").setLevel(logging.WARNING)
    # logging.getLogger("httpx").setLevel(logging.WARNING)

    # Test message to confirm logger setup
    # Initializing a logger here to demonstrate its usage post-setup
    # initial_logger = get_logger("logging_setup") # Use a specific name
    # initial_logger.info(f"Logging initialized. Level: {log_level_str}. File: {log_file_path}")
    # initial_logger.debug("This is a debug message from logging_setup.")

# Call setup_logging once when this module is imported, if desired as a global setup.
# However, it's better to call it explicitly from your main application entry point
# to ensure it's done at the right time.
# setup_logging() # Commented out for explicit call in mcp_setup.py

def set_level(level: str) -> None:
    """
    Dynamically set the logging level for the root logger.
    
    Args:
        level: String representation of log level ('DEBUG', 'INFO', etc.)
    """
    log_level_str = level.upper()
    log_levels = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL
    }
    log_level = log_levels.get(log_level_str, logging.INFO)
    
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Also update all handlers to maintain consistent levels
    for handler in root_logger.handlers:
        handler.setLevel(log_level)
    
    root_logger.info(f"Logging level changed to {log_level_str}")

class LogAccessor:
    """
    Provides access to the server's log file contents.
    Intended to be instantiated and made available via LifespanContext.
    """
    def __init__(self, log_directory: str = "logs", default_log_filename: Optional[str] = None):
        """
        Initialize the LogAccessor.

        Args:
            log_directory: The directory where log files are stored. Defaults to "logs".
            default_log_filename: The default log filename to use if LOG_FILENAME environment
                                  variable is not set. Defaults to "crawl4mcp.log" if None.
        """
        self.log_dir = Path(log_directory)
        # If LOG_FILENAME env var is set, it takes precedence.
        # Otherwise, use the default_log_filename passed during instantiation (e.g., from mcp_setup).
        # If neither, default to 'crawl4mcp.log'.
        self.log_filename = os.getenv("LOG_FILENAME", default_log_filename or "crawl4mcp.log")
        self.log_file_path = self.log_dir / self.log_filename
        self._ensure_log_dir_exists()

    def _ensure_log_dir_exists(self):
        """Ensures the log directory exists."""
        try:
            self.log_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            # Use a basic print here if logger itself isn't ready or causes issues
            print(f"[LogAccessor._ensure_log_dir_exists] Critical error: Could not create log directory {self.log_dir}: {e}", file=sys.stderr)

    async def get_last_log_lines(self, num_lines: int = 150) -> List[str]:
        """
        Retrieves the last N lines from the configured log file.

        Args:
            num_lines: The number of lines to retrieve from the end of the log file.
        
        Returns:
            A list of strings, where each string is a log line. 
            Returns an empty list if the file doesn't exist or an error occurs.
        """
        # Ensure the path is up-to-date if LOG_FILENAME could change at runtime (though unlikely for this setup)
        current_log_filename = os.getenv("LOG_FILENAME", self.log_filename) # Re-check env var
        if current_log_filename != self.log_filename:
            self.log_filename = current_log_filename
            self.log_file_path = self.log_dir / self.log_filename

        # Check file existence asynchronously if possible, or keep sync if it's quick
        # For this case, a quick sync check before threading is fine.
        if not self.log_file_path.exists() or not self.log_file_path.is_file():
            # Logger might not be fully set up or could be part of the issue, so direct print for critical path
            print(f"[LogAccessor.get_last_log_lines] Log file not found or is not a file: {self.log_file_path}", file=sys.stderr)
            return []

        try:
            # Define a helper synchronous function for file operations
            def _read_log_lines():
                with open(self.log_file_path, 'r', encoding='utf-8') as f:
                    all_lines = f.readlines()
                    return all_lines[-num_lines:]
            
            return await asyncio.to_thread(_read_log_lines)
        except Exception as e:
            print(f"[LogAccessor.get_last_log_lines] Error reading log file {self.log_file_path}: {e}", file=sys.stderr)
            return []

# Ensure the file ends with a newline for linters 