import os
import logging
from logging.handlers import RotatingFileHandler

def setup_logger(name: str = "disaster_assessment", log_level: str = "INFO", log_file: str = "logs/disaster_assessment.log", max_bytes: int = 10 * 1024 * 1024, backup_count: int = 5) -> logging.Logger:
    """
    Sets up a logger with both console and rotating file handlers.
    
    Args:
        name (str): Name of the logger.
        log_level (str): String representation of the logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        log_file (str): Path to the log file.
        max_bytes (int): Maximum size of log file before rotating (10MB default).
        backup_count (int): Number of backup log files to retain.
        
    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    
    # Avoid duplicate handlers if already configured
    if logger.handlers:
        return logger

    # Ensure log directory exists
    log_dir = os.path.dirname(log_file)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)

    # Formatters
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(filename)s:%(lineno)d | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console Handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File Handler
    try:
        file_handler = RotatingFileHandler(
            log_file, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8"
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except Exception as e:
        logger.warning(f"Could not create file log handler: {e}. Logging to console only.")

    return logger

# Create default application logger
logger = setup_logger()
