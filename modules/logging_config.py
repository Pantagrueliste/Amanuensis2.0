"""
Logging configuration for Amanuensis 2.0
"""

import os
import logging
import logging.handlers
import tempfile
from pathlib import Path


def setup_logging(level=logging.INFO, log_dir=None):
    """
    Configure logging for the application.
    
    Args:
        level: Logging level (default: INFO)
        log_dir: Directory to store log files (default: tempfile.gettempdir())
    """
    # Use system temp directory if log_dir is not specified
    if log_dir is None:
        log_dir = tempfile.gettempdir()
    
    # Create logs directory if it doesn't exist
    log_path = Path(log_dir)
    log_path.mkdir(exist_ok=True)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Clear any existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Create formatters
    console_formatter = logging.Formatter(
        '%(levelname)s: %(message)s'
    )
    
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(console_formatter)
    
    # File handler for general logs
    log_file = os.path.join(log_dir, "amanuensis.log")
    file_handler = logging.handlers.RotatingFileHandler(
        log_file,
        maxBytes=10485760,  # 10 MB
        backupCount=5
    )
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(file_formatter)
    
    # Add handlers to root logger
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)
    
    # Add separate error log file if level is higher than ERROR
    if level > logging.ERROR:
        error_handler = logging.handlers.RotatingFileHandler(
            os.path.join(log_dir, "errors.log"),
            maxBytes=10485760,  # 10 MB
            backupCount=5
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(file_formatter)
        root_logger.addHandler(error_handler)
    
    return root_logger


def get_logger(module_name):
    """
    Get a logger for a specific module.
    
    Args:
        name: Name of the module
        
    Returns:
        Logger object
    """
    # Make sure logging is set up
    if not logging.getLogger().handlers:
        setup_logging()
        
    return logging.getLogger(module_name)