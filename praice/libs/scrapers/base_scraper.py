import time
from abc import ABC, abstractmethod
from typing import Dict, List, Optional

import requests
from bs4 import BeautifulSoup
from loguru import logger


class NewsScraper(ABC):
    """
    Abstract base class for news scrapers.

    Attributes:
        proxy (Optional[Dict[str, str]]): A dictionary containing proxy information.
        headers (Dict[str, str]): A dictionary containing user agent information.

    Methods:
        scrape_headlines(url: str) -> List[Dict[str, str]]:
            Scrapes the headlines from the given URL and returns a list of dictionaries containing the scraped data.

        scrape_article(url: str) -> Dict[str, str]:
            Scrapes the article from the given URL and returns a dictionary containing the scraped data.

        get_soup(url: str) -> BeautifulSoup:
            Fetches the HTML content from the given URL, using the specified headers and proxy,
            if provided, and returns a BeautifulSoup object.

    """

    def __init__(self, proxy: Optional[Dict[str, str]] = None):
        self.proxy = proxy
        self.headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/103.0.0.0 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "DNT": "1",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
        }

    @abstractmethod
    def scrape_headlines(self, url: str) -> List[Dict[str, str]]:
        """
        Scrapes headlines from a given URL.

        Parameters:
        - url (str): The URL to scrape headlines from.

        Returns:
        - List[Dict[str, str]]: A list of dictionaries, where each dictionary represents a headline and contains the following keys:
            - 'title' (str): The title of the headline.
            - 'link' (str): The link to the full article.

        """

        pass

    @abstractmethod
    def scrape_article(self, url: str) -> Dict[str, str]:
        """
        Scrapes an article from the given URL and returns a dictionary containing the scraped data.

        Parameters:
            url (str): The URL of the article to scrape.

        Returns:
            Dict[str, str]: A dictionary containing the scraped data,
                where the keys represent the data fields and the values represent the corresponding values.
        """

        pass

    def get_soup(
        self, url: str, max_retries: int = 3, timeout: int = 30
    ) -> BeautifulSoup:
        """
        Retrieves the BeautifulSoup object by making a GET request to the specified URL.

        Args:
            url (str): The URL to make the GET request to.
            max_retries (int, optional): The maximum number of retries in case of request failure. Defaults to 3.
            timeout (int, optional): The timeout for the request in seconds. Defaults to 30.

        Returns:
            BeautifulSoup: The BeautifulSoup object representing the parsed HTML content.

        Raises:
            Exception: If the request fails after the maximum number of retries.
        """

        for i in range(max_retries):
            try:
                response = requests.get(
                    url, headers=self.headers, proxies=self.proxy, timeout=timeout
                )
                response.raise_for_status()
                return BeautifulSoup(response.content, "html.parser")
            except (requests.RequestException, requests.HTTPError) as e:
                logger.warning(
                    f"Request failed: {e}. Retrying ({i+1}/{max_retries})..."
                )
                time.sleep(2**i)  # Exponential backoff
        raise Exception(f"Failed to retrieve {url} after {max_retries} attempts")
