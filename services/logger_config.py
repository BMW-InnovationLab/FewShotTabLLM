import logging
import sys
import os

def setup_logging(log_file=None, verbose=False):
    """
    Setup logging configuration.
    
    Args:
        log_file (str): Path to the log file.
        verbose (bool): If True, show INFO logs on stdout. If False, only show WARNING and above.
    """
    logger = logging.getLogger()
    # Set root logger to DEBUG to capture everything
    logger.setLevel(logging.DEBUG)

    # Clear existing handlers
    logger.handlers = []

    # Formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # File Handler (Always logs DEBUG and above - "everything")
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    if verbose:
        console_handler.setLevel(logging.INFO)
    else:
        console_handler.setLevel(logging.WARNING)  # Only show warnings/errors on stdout by default
    
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Suppress overly verbose libraries unless explicitly needed
    # Adjust these as needed based on observation
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)

    return logger
