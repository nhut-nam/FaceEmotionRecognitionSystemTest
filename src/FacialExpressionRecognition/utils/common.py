import os
from box.exceptions import BoxValueError
import yaml
from FacialExpressionRecognition import logger
import joblib
import json
from ensure import ensure_annotations
from box import ConfigBox
from pathlib import Path
from typing import Any
import base64

@ensure_annotations
def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """Reads a yaml file and returns a ConfigBox object

    Args:
        path_to_yaml (Path): Path to the yaml file

    Raises:
        e: Raises an exception if the file is not found or if there is an error in reading the file

    Returns:
        ConfigBox: ConfigBox object containing the contents of the yaml file
    """
    try:
        with open(path_to_yaml, 'r') as yaml_file:
            content = yaml.safe_load(yaml_file)
            logger.info(f"yaml file: {path_to_yaml} loaded successfully")
            return ConfigBox(content)
    except BoxValueError as e:
        raise e
    except Exception as e:
        raise e
    
@ensure_annotations
def create_directories(path_to_directories: list, verbose=True):
    """Creates a list of directories

    Args:
        path_to_directories (list): List of directory paths to be created
    """
    for path in path_to_directories:
        os.makedirs(path, exist_ok=True)
        if verbose:
            logger.info(f"Directory created at: {path}")

@ensure_annotations
def save_json(path: Path, data: dict):
    """Saves a dictionary to a json file

    Args:
        path (Path): Path to the json file
        data (dict): Dictionary to be saved
    """
    with open(path, 'w') as json_file:
        json.dump(data, json_file, indent=4)
    logger.info(f"JSON file saved at: {path}")

@ensure_annotations
def load_json(path: Path) -> ConfigBox:
    """Loads a json file and returns a ConfigBox object

    Args:
        path (Path): Path to the json file

    Raises:
        e: Raises an exception if the file is not found or if there is an error in reading the file

    Returns:
        ConfigBox: ConfigBox object containing the contents of the json file
    """
    try:
        with open(path, 'r') as json_file:
            content = json.load(json_file)
            logger.info(f"JSON file: {path} loaded successfully")
            return ConfigBox(content)
    except BoxValueError as e:
        raise e
    except Exception as e:
        raise e
    
@ensure_annotations
def load_binary_file(file_path: Path) -> bytes:
    """Loads a binary file and returns its content as bytes

    Args:
        file_path (Path): Path to the binary file

    Raises:
        e: Raises an exception if the file is not found or if there is an error in reading the file

    Returns:
        bytes: Content of the binary file
    """
    try:
        with open(file_path, 'rb') as binary_file:
            content = binary_file.read()
            logger.info(f"Binary file: {file_path} loaded successfully")
            return content
    except Exception as e:
        raise e
    
@ensure_annotations
def get_size(path: Path) -> str:
    """Returns the size of a file in KB

    Args:
        path (Path): Path to the file

    Returns:
        str: Size of the file in KB
    """
    size_in_kb = round(os.path.getsize(path) / 1024, 2)
    return f"{size_in_kb} KB"


@ensure_annotations
def decode_image(image_base64: str) -> bytes:
    """Decodes a base64 encoded image string to bytes

    Args:
        image_base64 (str): Base64 encoded image string

    Returns:
        bytes: Decoded image in bytes
    """
    try:
        image_bytes = base64.b64decode(image_base64)
        logger.info("Image decoded successfully")
        return image_bytes
    except Exception as e:
        raise e
    
@ensure_annotations
def encode_image(image_bytes: bytes) -> str:
    """Encodes image bytes to a base64 string

    Args:
        image_bytes (bytes): Image in bytes

    Returns:
        str: Base64 encoded image string
    """
    try:
        image_base64 = base64.b64encode(image_bytes).decode('utf-8')
        logger.info("Image encoded successfully")
        return image_base64
    except Exception as e:
        raise e