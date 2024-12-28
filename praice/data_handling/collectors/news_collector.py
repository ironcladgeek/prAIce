from datetime import UTC, datetime
from typing import Dict, List, Optional

from loguru import logger
from peewee import DoesNotExist

from praice.data_handling.db_ops.crud import (
    create_news_symbol,
    get_or_create_news,
    update_news,
)
from praice.data_handling.db_ops.news_helpers import get_news_with_null_content
from praice.data_handling.db_ops.scraping_url_helpers import (
    get_active_scraping_urls_by_source,
    get_scraping_url_by_symbol_and_source,
)
from praice.data_handling.db_ops.symbol_helpers import get_or_create_symbol
from praice.data_handling.models import db
from praice.libs.scrapers.scraper_factory import ScraperFactory


def collect_news_headlines(
    symbol: str, source: str, proxy: Optional[Dict[str, str]] = None
) -> List[Dict[str, str]]:
    """
    Collects news headlines for a given symbol and source.

    Args:
        symbol (str): The symbol for which to collect news headlines.
        source (str): The source from which to collect news headlines.
        proxy (Optional[Dict[str, str]], optional): Proxy configuration. Defaults to None.

    Returns:
        List[Dict[str, str]]: A list of dictionaries representing the collected news headlines.
    """

    logger.info(f"Starting news collection for symbol: {symbol}, source: {source}")

    with db.atomic():
        try:
            scraping_url = get_scraping_url_by_symbol_and_source(symbol, source)
        except DoesNotExist:
            logger.error(
                f"ScrapingUrl not found for symbol {symbol} and source {source}."
            )
            return []

        scraper = ScraperFactory.get_scraper(source=source, proxy=proxy)
        news_items = scraper.scrape_headlines(scraping_url.url)
        num_new_headlines = 0

        for item in news_items:
            _, created = get_or_create_news(
                title=item["headline"],
                url=item["link"],
                source=source,
                scraped_at=datetime.now(UTC),
            )
            num_new_headlines += 1 if created else 0

    logger.info(
        f"Completed news collection for symbol: {symbol}, "
        f"source: {source}. Total items: {len(news_items)}. New items: {num_new_headlines}"
    )
    return news_items


def collect_news_articles(
    proxy: Optional[Dict[str, str]] = None, limit: int = 50, randomize: bool = True
) -> None:
    """
    Collects news articles with null content and scrapes their full content.

    Args:
        proxy (Optional[Dict[str, str]]): A dictionary containing proxy information. Defaults to None.
        limit (int): The maximum number of news articles to scrape. Defaults to 50.
        randomize (bool): Whether to randomize the order of the news articles. Defaults to True.

    Returns:
        None
    """
    news_to_scrape = get_news_with_null_content(limit=limit, randomize=randomize)
    num_scraped = 0

    logger.info(
        f"Starting full article scraping for {len(news_to_scrape)} items with null content"
    )

    for news in news_to_scrape:
        try:
            scraper = ScraperFactory.get_scraper(source=news.source, proxy=proxy)
            article_data = scraper.scrape_article(news.url)

            with db.atomic():
                update_news(
                    news.id,
                    content=article_data["content"],
                    published_at=article_data["published_at"],
                    scraped_at=datetime.now(UTC),
                )
                num_scraped += 1

                # Create NewsSymbol entries for each symbol mentioned in the article
                for symbol in article_data["symbols"]:
                    try:
                        symbol_obj = get_or_create_symbol(symbol)
                        create_news_symbol(news=news, symbol=symbol_obj)
                    except ValueError as e:
                        logger.error(f"Error creating symbol {symbol}: {str(e)}")

        except Exception as e:
            logger.error(f"Error scraping article {news.url}: {str(e)}")

    logger.info(
        "Completed full article scraping. "
        f"{num_scraped} items scraped out of {len(news_to_scrape)}."
    )


def collect_news_headlines_by_source(
    source: str, proxy: Optional[Dict[str, str]] = None
) -> None:
    """
    Collects news headlines by source.

    Args:
        source (str): The source of the news headlines.
        proxy (Optional[Dict[str, str]], optional): A dictionary containing proxy information.
            Defaults to None.

    Returns:
        None
    """
    scraping_urls = get_active_scraping_urls_by_source(source=source)
    for url in scraping_urls:
        collect_news_headlines(symbol=url.symbol.symbol, source=url.source, proxy=proxy)
