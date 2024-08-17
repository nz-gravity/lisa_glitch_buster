import logging


def setup_logger(name, level=logging.INFO):
    # Create a logger
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Create a console handler
    ch = logging.StreamHandler()
    ch.setLevel(level)

    # Create a formatter and set it for the handler TIME|NAME|LEVEL|>>> MESSAGE
    formatter = logging.Formatter(
        "%(asctime)s|%(name)s|%(levelname)s|>>> %(message)s"
    )
    ch.setFormatter(formatter)

    # Add the handler to the logger
    logger.addHandler(ch)

    return logger


# Initialize the logger
logger = setup_logger("LISA-Glitch-Buster")
