from typing import Dict, Optional

from praice.libs.scrapers.base_scraper import NewsScraper
from praice.libs.scrapers.yahoo_finance_scraper import YahooFinanceScraper


class ScraperFactory:
    """
    A factory class for creating and registering scrapers.

    Methods:
    - get_scraper(source: str, proxy: Optional[Dict[str, str]] = None) -> NewsScraper:
        Returns an instance of the scraper for the given source.

    - register_scraper(source: str, scraper_class: type):
        Registers a new scraper class for the given source.

    Attributes:
    - _scrapers: Dict[str, type]
        A dictionary mapping source names to scraper classes.
    """

    _scrapers: Dict[str, type] = {"yfinance": YahooFinanceScraper}

    @classmethod
    def get_scraper(
        cls, source: str, proxy: Optional[Dict[str, str]] = None
    ) -> NewsScraper:
        """
        Returns an instance of a NewsScraper based on the given source.

        Args:
            source (str): The source of the scraper.
            proxy (Optional[Dict[str, str]]): Optional proxy dictionary.

        Returns:
            NewsScraper: An instance of a NewsScraper.

        Raises:
            ValueError: If no scraper is available for the given source.
        """
        scraper_class = cls._scrapers.get(source)
        if scraper_class is None:
            raise ValueError(f"No scraper available for source: {source}")
        return scraper_class(proxy=proxy)

    @classmethod
    def register_scraper(cls, source: str, scraper_class: type):
        """
        Register a scraper class for a specific data source.

        Args:
            source (str): The data source to be associated with the scraper.
            scraper_class (type): The class of the scraper to be registered.
        """
        cls._scrapers[source] = scraper_class
