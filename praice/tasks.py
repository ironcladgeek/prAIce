from datetime import datetime, timedelta

import pytz
from celery import shared_task

from praice.data_handling.collectors import (
    fundamental_collector,
    news_collector,
    price_collector,
)
from praice.data_handling.db_ops import ta_helpers
from praice.data_handling.processors import news_processor
from praice.utils.logging import get_scheduler_logger

# Get the scheduler-specific logger
logger = get_scheduler_logger()


@shared_task
def collect_headlines_by_source_job(source: str):
    """
    Collects headlines from a specific source.

    Args:
        source (str): The source from which to collect headlines.
    """
    logger.info(f"Starting headline collection job from source {source}")
    try:
        # Collects news headlines from the specified source
        news_collector.collect_news_headlines_by_source(source)
        logger.info("Headline collection job completed successfully")
    except Exception as e:
        logger.error(f"Error in headline collection job: {str(e)}")


@shared_task
def collect_articles_job():
    """
    Executes the article collection job.
    """
    logger.info("Starting article collection job")
    try:
        # Collects news articles with null content and scrapes their full content
        news_collector.collect_news_articles(limit=150, randomize=True)
        logger.info("Article collection job completed successfully")
    except Exception as e:
        logger.error(f"Error in article collection job: {str(e)}")


@shared_task
def collect_price_data_job():
    """
    Executes the price data collection job.
    """
    logger.info("Starting price data collection job")
    try:
        # Collect historical price data for all symbols that have collect_price_data
        # set to True in the SymbolConfig table for the last 5 days
        price_collector.collect_historical_prices_all(period="5d")
        logger.info("Price data collection job completed successfully")
    except Exception as e:
        logger.error(f"Error in price data collection job: {str(e)}")


@shared_task
def calculate_and_store_technical_analysis_job():
    """
    Executes the technical analysis calculation and storage job.
    """
    logger.info("Starting technical analysis calculation and storage job")
    try:
        end_date = datetime.now(
            tz=pytz.timezone("US/Eastern")
        ).date()  # get today's date
        start_date = end_date - timedelta(days=2)  # get the date 2 days ago
        # Calculate and store technical analysis for all symbols
        # that have collect_technical_indicators set to True in the SymbolConfig table
        ta_helpers.calculate_and_store_technical_analysis_for_all_symbols(
            start_date=start_date, end_date=end_date
        )
        logger.info(
            "Technical analysis calculation and storage job completed successfully"
        )
    except Exception as e:
        logger.error(
            f"Error in technical analysis calculation and storage job: {str(e)}"
        )


@shared_task
def collect_and_store_fundamental_data_job():
    """
    Executes the fundamental data collection and storage job.
    """
    logger.info("Starting fundamental data collection and storage job")
    try:
        # Collect and store fundamental data for all symbols
        # that have collect_fundamental_data set to True in the SymbolConfig table
        fundamental_collector.collect_and_store_fundamental_data_for_all_symbols()
        logger.info(
            "Fundamental data collection and storage job completed successfully"
        )
    except Exception as e:
        logger.error(f"Error in fundamental data collection and storage job: {str(e)}")


@shared_task
def populate_news_words_count_job():
    """
    Executes the news words count population job.
    """
    logger.info("Starting news words count population job")
    try:
        # Populate the words_count field for News entries with non-null content
        news_processor.populate_words_count()
        logger.info("News words count population job completed successfully")
    except Exception as e:
        logger.error(f"Error in news words count population job: {str(e)}")


@shared_task
def generate_news_summaries_job(limit: int = 5, model: str = "bart"):
    """
    Executes the news summaries generation job.
    """
    logger.info("Starting news summaries generation job")
    try:
        # Generate content summaries for News entries with words_count greater than or equal to 300
        n_generated_summary, news_ids = news_processor.populate_content_summary(
            limit=limit, model=model
        )
        logger.info(
            "News summaries generation job completed successfully for "
            f"{n_generated_summary} entries. News IDs: {news_ids}"
        )
    except Exception as e:
        logger.error(f"Error in news summaries generation job: {str(e)}")


@shared_task
def populate_sentiment_scores_job(limit: int = 5):
    """
    Executes the sentiment scores calculation job.
    """
    logger.info("Starting sentiment scores calculation job")
    try:
        # Calculate sentiment scores for News entries with content_summary but no sentiment_score
        n_calculated_scores, news_ids = news_processor.populate_sentiment_score(
            limit=limit
        )
        logger.info(
            "Sentiment scores calculation job completed successfully for "
            f"{n_calculated_scores} entries. News IDs: {news_ids}"
        )
    except Exception as e:
        logger.error(f"Error in sentiment scores calculation job: {str(e)}")


@shared_task
def populate_relevance_score_job(limit: int = 5):
    """
    Executes the relevance scores calculation job.
    """
    logger.info("Starting relevance scores calculation job")
    try:
        # Calculate relevance scores for News entries with content_summary but no relevance_score
        n_calculated_scores, ids = news_processor.populate_relevance_score(limit=limit)
        logger.info(
            "Relevance scores calculation job completed successfully for "
            f"{n_calculated_scores} entries. NewsSymbol IDs: {ids}"
        )
    except Exception as e:
        logger.error(f"Error in relevance scores calculation job: {str(e)}")
