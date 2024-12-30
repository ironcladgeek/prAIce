from datetime import UTC, datetime
from enum import Enum

from peewee import (
    AutoField,
    BigIntegerField,
    BooleanField,
    CharField,
    CompositeKey,
    DateField,
    DateTimeField,
    DecimalField,
    ForeignKeyField,
    IntegerField,
    Model,
    PostgresqlDatabase,
    TextField,
)
from playhouse.postgres_ext import JSONField

from praice.core.config import settings

db = PostgresqlDatabase(
    database=settings.DB_NAME,
    user=settings.DB_USERNAME,
    password=settings.DB_PASSWORD,
    host=settings.DB_HOST,
    port=settings.DB_PORT,
)


class BaseModel(Model):
    """
    BaseModel class.

    Attributes:
        Meta: A nested class that defines the metadata for the BaseModel class.
            database (Database): The database connection for the BaseModel class.
    """

    class Meta:
        database = db


class AssetClass(Enum):
    """Enum class representing different asset classes."""

    STOCK = "stock"
    COMMODITY = "commodity"
    CURRENCY = "currency"
    CRYPTO = "crypto"
    ETF = "etf"
    INDEX = "index"
    BOND = "bond"
    MUTUALFUND = "mutualfund"
    FUTURES = "futures"
    OPTION = "option"


class Symbol(BaseModel):
    """
    Represents a symbol in the system.

    Attributes:
        id (int): The unique identifier of the symbol.
        symbol (str): The symbol string (uppercase).
        name (str): The name of the symbol.
        asset_class (str): The asset class of the symbol (lowercase).
        sector (str, optional): The sector of the symbol (title).
        industry (str, optional): The industry of the symbol (title).
        exchange (str, optional): The exchange of the symbol (uppercase).
        is_active (bool): Indicates if the symbol is active.
        created_at (datetime): The date and time when the symbol was created.
        updated_at (datetime): The date and time when the symbol was last updated.
    """

    id = AutoField(primary_key=True)
    symbol = CharField(max_length=20, unique=True, index=True)
    name = CharField(max_length=255)
    asset_class = CharField(
        max_length=50, choices=[(e.value, e.name) for e in AssetClass]
    )
    sector = CharField(max_length=100, null=True)
    industry = CharField(max_length=100, null=True)
    exchange = CharField(max_length=50, null=True)
    is_active = BooleanField(default=True)
    created_at = DateTimeField(default=lambda: datetime.now(UTC))
    updated_at = DateTimeField(default=lambda: datetime.now(UTC))
    description = TextField(null=True)

    class Meta:
        table_name = "symbols"

    def save(self, *args, **kwargs):
        """
        Save the symbol instance.

        Returns:
            The saved symbol instance.
        Raises:
            ValueError: If the asset_class is invalid.
        """

        self.updated_at = datetime.now(UTC)
        self.symbol = self.symbol.upper()
        if self.name:
            self.name = self.name.title()
        if self.exchange:
            self.exchange = self.exchange.upper()
        if self.sector:
            self.sector = self.sector.title()
        if self.industry:
            self.industry = self.industry.title()
        self.asset_class = self.asset_class.lower()

        if self.asset_class not in [e.value for e in AssetClass]:
            raise ValueError(
                f"Invalid asset_class: {self.asset_class}. "
                f"Must be one of {[e.value for e in AssetClass]}"
            )

        return super(Symbol, self).save(*args, **kwargs)

    @classmethod
    def create(cls, **query):
        """
        Create a new symbol instance.

        This method applies the necessary transformations before creating the instance.

        Args:
            **query: The attributes for creating the new symbol.

        Returns:
            The newly created Symbol instance.
        """
        if "symbol" in query:
            query["symbol"] = query["symbol"].upper()
        if "name" in query and query["name"]:
            query["name"] = query["name"].title()
        if "exchange" in query and query["exchange"]:
            query["exchange"] = query["exchange"].upper()
        if "sector" in query and query["sector"]:
            query["sector"] = query["sector"].title()
        if "industry" in query and query["industry"]:
            query["industry"] = query["industry"].title()
        if "asset_class" in query:
            query["asset_class"] = query["asset_class"].lower()
            if query["asset_class"] not in [e.value for e in AssetClass]:
                raise ValueError(
                    f"Invalid asset_class: {query['asset_class']}. "
                    f"Must be one of {[e.value for e in AssetClass]}"
                )
        return super(Symbol, cls).create(**query)


class SymbolConfig(BaseModel):
    """
    Represents the configuration for a symbol.

    Attributes:
        symbol (ForeignKeyField): The foreign key to the Symbol model.
        collect_price_data (BooleanField): Indicates whether to collect price data.
        collect_yfinance_news (BooleanField): Indicates whether to collect news from Yahoo Finance.
        collect_technical_indicators (BooleanField): Indicates whether to collect technical indicators.
        collect_fundamental_data (BooleanField): Indicates whether to collect fundamental data.
    """

    symbol = ForeignKeyField(Symbol, backref="config", unique=True)
    collect_price_data = BooleanField(default=True)
    collect_yfinance_news = BooleanField(default=True)
    collect_technical_indicators = BooleanField(default=True)
    collect_fundamental_data = BooleanField(default=True)

    class Meta:
        table_name = "symbol_configs"


class News(BaseModel):
    """
    Represents a news article.

    Attributes:
        id (int): The unique identifier of the news article.
        url (str): The URL of the news article.
        title (str): The title of the news article.
        content (str): The content of the news article.
        source (str): The source of the news article.
        published_at (datetime): The date and time when the news article was published.
        scraped_at (datetime): The date and time when the news article was scraped.
        words_count (int): The number of words in the news article.
        content_summary (str): The summary of the news article.
        sentiment_score (DecimalField): The sentiment score of the news article.
    """

    id = AutoField(primary_key=True)
    url = CharField(max_length=512, unique=True)
    title = CharField(max_length=255)
    content = TextField(null=True)
    source = CharField(max_length=100)
    published_at = DateTimeField(null=True)
    scraped_at = DateTimeField(default=lambda: datetime.now(UTC))
    words_count = IntegerField(null=True)
    content_summary = TextField(null=True)
    sentiment_score = DecimalField(max_digits=10, decimal_places=5, null=True)

    class Meta:
        table_name = "news"


class NewsSymbol(BaseModel):
    """
    Represents a mapping between a news article and a symbol.

    Attributes:
        news (ForeignKeyField): The foreign key to the News model, representing the news article.
        symbol (ForeignKeyField): The foreign key to the Symbol model, representing the symbol.
        relevance_score (DecimalField): The relevance score of the symbol in the news article.
    """

    news = ForeignKeyField(News, backref="news_symbols", on_delete="CASCADE")
    symbol = ForeignKeyField(Symbol, backref="news_symbols", on_delete="CASCADE")
    relevance_score = DecimalField(max_digits=10, decimal_places=5, null=True)

    class Meta:
        table_name = "news_symbols"
        indexes = ((("news", "symbol"), True),)


class ScrapingUrl(BaseModel):
    """
    Represents a scraping URL.

    Attributes:
        id (int): The ID of the scraping URL.
        symbol (Symbol): The symbol associated with the scraping URL.
        url (str): The URL to be scraped.
        source (str): The source of the scraping URL.
        is_active (bool): Indicates if the scraping URL is active.
        created_at (datetime): The timestamp when the scraping URL was created.
        updated_at (datetime): The timestamp when the scraping URL was last updated.
    """

    ...
    id = AutoField(primary_key=True)
    symbol = ForeignKeyField(Symbol, backref="scraping_urls")
    url = CharField(max_length=512, unique=True)
    source = CharField(max_length=100)
    is_active = BooleanField(default=True)
    created_at = DateTimeField(default=lambda: datetime.now(UTC))
    updated_at = DateTimeField(default=lambda: datetime.now(UTC))

    class Meta:
        table_name = "scraping_urls"
        indexes = ((("symbol", "source"), True),)

    def save(self, *args, **kwargs):
        """
        Save the object and update the 'updated_at' field with the current datetime in UTC.

        Returns:
            The saved object.
        """
        self.updated_at = datetime.now(UTC)
        return super(ScrapingUrl, self).save(*args, **kwargs)


class HistoricalPrice1D(BaseModel):
    symbol = ForeignKeyField(Symbol, backref="historical_prices")
    date = DateField()
    open = DecimalField(max_digits=10, decimal_places=2)
    high = DecimalField(max_digits=10, decimal_places=2)
    low = DecimalField(max_digits=10, decimal_places=2)
    close = DecimalField(max_digits=10, decimal_places=2)
    volume = BigIntegerField()
    dividends = DecimalField(max_digits=10, decimal_places=2)
    stock_splits = DecimalField(max_digits=10, decimal_places=2)

    class Meta:
        table_name = "historical_prices_1d"
        primary_key = CompositeKey("symbol", "date")


class Timeframe(Enum):
    """Enum class representing different timeframes."""

    HOURS_1 = "1H"
    HOURS_4 = "4H"
    DAYS_1 = "1D"
    WEEKS_1 = "1W"
    WEEKS_2 = "2W"
    MONTHS_1 = "1M"
    MONTHS_3 = "3M"


class TechnicalAnalysis(BaseModel):
    symbol = ForeignKeyField(Symbol, backref="technical_analysis")
    date = DateField()
    timeframe = CharField(
        max_length=5,
        choices=[(e.value, e.name) for e in Timeframe],
        default=Timeframe.DAYS_1.value,
    )
    technical_indicators = JSONField(default=dict)
    candlestick_patterns = JSONField(default=dict)

    class Meta:
        table_name = "technical_analysis"
        indexes = ((("symbol", "date", "timeframe"), True),)

    def save(self, *args, **kwargs):
        if self.timeframe not in [e.value for e in Timeframe]:
            raise ValueError(
                f"Invalid timeframe: {self.timeframe}. "
                f"Must be one of {[e.value for e in Timeframe]}"
            )
        return super(TechnicalAnalysis, self).save(*args, **kwargs)


class FundamentalData(BaseModel):
    """
    Model representing fundamental data for a specific symbol, date, and period.

    Attributes:
        symbol (ForeignKeyField): The foreign key to the Symbol model.
        date (DateField): The date of the fundamental data.
        period (CharField): The period of the fundamental data ('annual' or 'quarterly').
        data (JSONField): The fundamental data stored as a JSON object.
    """

    symbol = ForeignKeyField(Symbol, backref="fundamental_data")
    date = DateField()
    period = CharField(max_length=10)  # 'annual' or 'quarterly'
    data = JSONField(default=dict)

    class Meta:
        table_name = "fundamental_data"
        indexes = ((("symbol", "date", "period"), True),)


def create_tables():
    """
    Creates tables in the database.

    This function uses the `db` connection to create tables in the database. It takes no arguments.
    """
    with db:
        db.create_tables(
            [
                Symbol,
                SymbolConfig,
                News,
                NewsSymbol,
                ScrapingUrl,
                HistoricalPrice1D,
                TechnicalAnalysis,
                FundamentalData,
            ]
        )


if __name__ == "__main__":
    create_tables()
