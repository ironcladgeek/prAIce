from celery import Celery
from celery.schedules import crontab

import praice.tasks  # noqa
from praice.config import settings

app = Celery()

# Configure Celery
app.conf.broker_url = settings.CELERY_BROKER_URL
app.conf.result_backend = settings.CELERY_RESULT_BACKEND
app.conf.timezone = "UTC"


# Configure Celery Beat schedule
app.conf.beat_schedule = {
    "collect-yfinance-headlines": {
        "task": "praice.tasks.collect_headlines_by_source_job",
        "schedule": crontab(minute="*/80"),  # every 80 minutes
        "args": ("yfinance",),
    },
    "collect-articles": {
        "task": "praice.tasks.collect_articles_job",
        "schedule": crontab(minute="*/170"),  # every 170 minutes
    },
    "collect-price-data": {
        "task": "praice.tasks.collect_price_data_job",
        "schedule": crontab(minute="0", hour="22"),  # daily at 6:00 PM ET
    },
    "calculate-store-technical-analysis": {
        "task": "praice.tasks.calculate_and_store_technical_analysis_job",
        # "schedule": "cron(30 22 * * *)",  # daily at 6:30 PM ET
        "schedule": crontab(minute="30", hour="22"),  # daily at 6:00 PM ET
    },
    "collect-store-fundamental-data": {
        "task": "praice.tasks.collect_and_store_fundamental_data_job",
        "schedule": crontab(
            minute="0", hour="23", day_of_month="1"
        ),  # monthly on the 1st day at 7:00 PM ET
    },
    "populate-news-words-count": {
        "task": "praice.tasks.populate_news_words_count_job",
        "schedule": crontab(minute="0", hour="0"),  # daily at 12:00 AM
    },
    "generate-news-summaries": {
        "task": "praice.tasks.generate_news_summaries_job",
        "schedule": crontab(minute="*/6"),
        "kwargs": {"limit": 5, "model": settings.SUMMARIZATION_MODEL},
    },
    "populate-sentiment-score": {
        "task": "praice.tasks.populate_sentiment_scores_job",
        "schedule": crontab(minute="*/10"),
        "kwargs": {"limit": 5},
    },
}
