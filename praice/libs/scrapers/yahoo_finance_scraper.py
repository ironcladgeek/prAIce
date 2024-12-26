from datetime import datetime
from typing import Dict, List

from loguru import logger

from praice.libs.scrapers.base_scraper import NewsScraper


class YahooFinanceScraper(NewsScraper):
    """
    A scraper class for scraping headlines and articles from Yahoo Finance.
    Methods:
    - scrape_headlines(url: str) -> List[Dict[str, str]]: Scrapes headlines from the given URL and returns a list of dictionaries containing the headline and link.
    - scrape_article(url: str) -> Dict[str, str]: Scrapes an article from the given URL and returns a dictionary containing the content, published date, and symbols mentioned in the article.
    """

    def scrape_headlines(self, url: str) -> List[Dict[str, str]]:
        """
        Scrapes headlines from the given URL.

        Args:
            url (str): The URL to scrape headlines from.

        Returns:
            List[Dict[str, str]]: A list of dictionaries containing the scraped headlines and their corresponding links.
        """
        logger.info(f"Scraping headlines from: {url}")
        soup = self.get_soup(url)
        news_list = []

        for news_item in soup.find_all(
            "li", class_="stream-item story-item yf-1usaaz9"
        ):
            headline_tag = news_item.find("h3", class_="clamp yf-18q3fnf")
            if headline_tag:
                headline = headline_tag.text.strip()
                link_tag = news_item.find("a", href=True)
                if link_tag:
                    link = link_tag["href"]
                    if not link.startswith("http"):
                        link = f"https://finance.yahoo.com{link}"
                    news_list.append({"headline": headline, "link": link})

        return news_list

    def scrape_article(self, url: str) -> Dict[str, str]:
        """
        Scrapes an article from the given URL and returns a dictionary containing the scraped data.

        Parameters:
            url (str): The URL of the article to scrape.

        Returns:
            dict: A dictionary containing the scraped data with the following keys:
                - "content" (str): The content of the article.
                - "published_at" (datetime): The timestamp of when the article was published.
                - "symbols" (list): A list of symbols mentioned in the article.
        """
        logger.info(f"Scraping article from: {url}")
        soup = self.get_soup(url)

        content = ""
        content_div = soup.find("div", class_="body")
        if content_div:
            content = content_div.get_text(separator="\n", strip=True)

        timestamp = None
        time_tag = soup.find("time")
        if time_tag and "datetime" in time_tag.attrs:
            timestamp = datetime.fromisoformat(
                time_tag["datetime"].replace("Z", "+00:00")
            )

        # Extract symbols mentioned in the article
        symbols = set()
        ticker_containers = soup.find_all("a", class_="ticker")
        for container in ticker_containers:
            symbol_span = container.find("span", class_="symbol")
            if symbol_span:
                symbol = symbol_span.text.strip()
                symbols.add(symbol)

        return {"content": content, "published_at": timestamp, "symbols": list(symbols)}
