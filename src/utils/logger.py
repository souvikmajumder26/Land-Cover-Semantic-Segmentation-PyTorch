import logging
from logging.handlers import RotatingFileHandler

def custom_logger(name, log_path, log_level="DEBUG"):

    logger = logging.getLogger(name)
    
    if log_level == "DEBUG":
        logger.setLevel(logging.DEBUG)
    elif log_level == "INFO":
        logger.setLevel(logging.INFO)
    elif log_level == "WARNING":
        logger.setLevel(logging.WARNING)
    elif log_level == "ERROR":
        logger.setLevel(logging.ERROR)
    elif log_level == "CRITICAL":
        logger.setLevel(logging.CRITICAL)
    else:
        logger.setLevel(logging.INFO)

    handler_file = RotatingFileHandler(
        log_path, mode = 'a', maxBytes = 1024 * 1024, backupCount = 5,
        encoding = None, delay = False
    )

    log_formatter = logging.Formatter(
        '[%(asctime)s] - %(name)s - {%(filename)s:%(lineno)d} - %(levelname)s - %(message)s',
        datefmt = '%Y-%m-%d %H:%M:%S'
    )

    handler_file.setFormatter(log_formatter)

    if log_level == "DEBUG":
        handler_file.setLevel(logging.DEBUG)
    elif log_level == "INFO":
        handler_file.setLevel(logging.INFO)
    elif log_level == "WARNING":
        handler_file.setLevel(logging.WARNING)
    elif log_level == "ERROR":
        handler_file.setLevel(logging.ERROR)
    elif log_level == "CRITICAL":
        handler_file.setLevel(logging.CRITICAL)
    else:
        handler_file.setLevel(logging.INFO)

    logger.addHandler(handler_file)

    return logger