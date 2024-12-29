import re
import time
from functools import wraps

import requests
from loguru import logger

from praice.core.config import settings


def chunked(iterable, chunk_size):
    """Yield successive chunk_size-sized chunks from iterable."""
    for i in range(0, len(iterable), chunk_size):
        yield iterable[i : i + chunk_size]


def count_words(text: str) -> int:
    """Count the number of words in a text string."""
    return len(re.findall(r"\w+", text))


def log_execution_time(func):
    """
    Decorator that logs the execution time of a function.

    Args:
        func (function): The function to be decorated.

    Returns:
        function: The decorated function.

    Example:
        @log_execution_time
        def my_function():
            # Function code here
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        """
        Wrapper function that logs the execution time of the decorated function.
        """
        func_name = func.__name__
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        duration = end_time - start_time
        logger.info(f"Function {func_name} took {duration:.2f} seconds to execute")
        return result

    return wrapper


def calculate_sentiment_score(text: str, model_name: str = "ProsusAI/finbert"):
    """
    Calculates the sentiment score of the given text using the specified model using an external API.

    Args:
        text (str): The text for which sentiment score needs to be calculated.
        model_name (str, optional): The name of the model to be used for sentiment analysis.
            Defaults to "ProsusAI/finbert".

    Returns:
        float: The sentiment score of the text in the range [-1, 1].

    Raises:
        HTTPError: If there is an error in the HTTP request to the sentiment API.
    """
    api_url = settings.SENTIMENT_API_URL
    response = requests.post(
        api_url,
        json={
            "text": text,
            "model_name": model_name,
        },
    )
    response.raise_for_status()
    return response.json()["sentiment_score"]


def calculate_similarity_score(
    text1: str, text2: str, model_name: str = "sentence-transformers/all-mpnet-base-v2"
):
    """
    Calculates the similarity score between two texts using the specified model using an external API.

    Args:
        text1 (str): The first text for comparison.
        text2 (str): The second text for comparison.
        model_name (str, optional): The name of the model to be used for sentence similarity.
            Defaults to "sentence-transformers/all-mpnet-base-v2".

    Returns:
        float: The cosine similarity score between the two texts.

    Raises:
        HTTPError: If there is an error in the HTTP request to the similarity API.
    """
    api_url = settings.SIMILARITY_API_URL
    response = requests.post(
        api_url,
        json={
            "sentence1": text1,
            "sentence2": text2,
            "model_name": model_name,
        },
    )
    response.raise_for_status()
    return response.json()["similarity_score"]
