import hashlib
import numpy as np
import logging

def generate_hash_id(input_id):
    """
    Generate a SHA-256 hash for a given input ID.

    Parameters:
        input_id (str): The input identifier to be hashed.

    Returns:
        str: A SHA-256 hash of the input ID.
    """
    return hashlib.sha256(input_id.encode()).hexdigest()

def normalize_vector(vector):
    """
    Normalize a vector to unit length.

    Parameters:
        vector (np.ndarray): The vector to normalize.

    Returns:
        np.ndarray: A unit vector in the direction of the input vector.
    """
    norm = np.linalg.norm(vector)
    if norm == 0:
        logging.warning("Attempt to normalize a zero vector.")
        return vector
    return vector / norm

def setup_logger(name, log_file, level=logging.INFO):
    """
    Configure and return a logger.

    Parameters:
        name (str): Name of the logger.
        log_file (str): File path for logging.
        level (logging.Level): The logging level.

    Returns:
        logging.Logger: Configured logger.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # File handler
    fh = logging.FileHandler(log_file)
    fh.setLevel(level)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    return logger

def validate_positive_integer(value, variable_name="variable"):
    """
    Validate that a provided value is a positive integer.

    Parameters:
        value (int): The value to check.
        variable_name (str): The name of the variable for error messages.

    Raises:
        ValueError: If the value is not a positive integer.
    """
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"{variable_name} must be a positive integer, got {value}.")
