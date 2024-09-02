import os
import logging

def setup_logging():
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger(__name__)
    return logger

def save_file(file_data, filename):
    with open(filename, 'wb') as f:
        f.write(file_data)
    return os.path.abspath(filename)

logger = setup_logging()