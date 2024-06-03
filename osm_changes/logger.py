import logging

# Define a global logger instance
logger = logging.getLogger(__name__)
_LOG_CONFIGURED = False

def configure_logger(logging_level: int | str = logging.INFO):
    global _LOG_CONFIGURED  # Declare _LOG_CONFIGURED as global
    if isinstance(logging_level, str):
        logging_level = logging.getLevelName(logging_level)

    # Configure logging level
    logger.setLevel(logging_level)

    # Create a formatter and set it for the handler
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    if _LOG_CONFIGURED:
        # handle = logger.handlers[0]
        return
    else: 
        # Create a stdout handler
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        # Add the handler to the logger
        logger.addHandler(handler)
        _LOG_CONFIGURED = True
