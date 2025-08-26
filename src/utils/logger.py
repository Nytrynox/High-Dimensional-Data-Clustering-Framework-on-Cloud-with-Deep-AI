"""
Logging utilities for the clustering framework
"""

import logging
import sys
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.logging import RichHandler


def setup_logging(
    level: str = "INFO",
    debug: bool = False,
    log_file: Optional[str] = None,
    rich_console: bool = True
) -> logging.Logger:
    """Setup logging configuration"""
    
    # Set logging level
    log_level = getattr(logging, level.upper(), logging.INFO)
    if debug:
        log_level = logging.DEBUG
    
    # Create logger
    logger = logging.getLogger("clustering_framework")
    logger.setLevel(log_level)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    simple_formatter = logging.Formatter(
        "%(levelname)s - %(message)s"
    )
    
    # Console handler
    if rich_console:
        console_handler = RichHandler(
            console=Console(stderr=True),
            show_time=True,
            show_level=True,
            show_path=False,
            rich_tracebacks=True
        )
        console_handler.setFormatter(simple_formatter)
    else:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(detailed_formatter)
    
    console_handler.setLevel(log_level)
    logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(detailed_formatter)
        file_handler.setLevel(log_level)
        logger.addHandler(file_handler)
    
    # Set third-party loggers to WARNING to reduce noise
    third_party_loggers = [
        "urllib3",
        "requests",
        "azure",
        "matplotlib",
        "PIL"
    ]
    
    for logger_name in third_party_loggers:
        third_party_logger = logging.getLogger(logger_name)
        third_party_logger.setLevel(logging.WARNING)
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance"""
    return logging.getLogger(f"clustering_framework.{name}")


class ClusteringLogger:
    """Enhanced logger for clustering operations"""
    
    def __init__(self, name: str):
        self.logger = get_logger(name)
        self.console = Console()
    
    def info(self, message: str, **kwargs):
        """Log info message with rich formatting"""
        self.logger.info(message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log warning message"""
        self.logger.warning(message, **kwargs)
    
    def error(self, message: str, **kwargs):
        """Log error message"""
        self.logger.error(message, **kwargs)
    
    def debug(self, message: str, **kwargs):
        """Log debug message"""
        self.logger.debug(message, **kwargs)
    
    def progress(self, message: str):
        """Log progress message with special formatting"""
        self.console.print(f"⏳ {message}", style="yellow")
        self.logger.info(message)
    
    def success(self, message: str):
        """Log success message with special formatting"""
        self.console.print(f"✅ {message}", style="green")
        self.logger.info(message)
    
    def failure(self, message: str):
        """Log failure message with special formatting"""
        self.console.print(f"❌ {message}", style="red")
        self.logger.error(message)
