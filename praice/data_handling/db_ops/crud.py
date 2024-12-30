from datetime import UTC, date, datetime
from typing import Any, Dict, List, Optional, Tuple, Union

from peewee import DoesNotExist, IntegrityError

from praice.data_handling.models import (
    FundamentalData,
    HistoricalPrice1D,
    News,
    NewsSymbol,
    ScrapingUrl,
    Symbol,
    SymbolConfig,
    TechnicalAnalysis,
    Timeframe,
    db,
)
from praice.utils import helpers

# ############################
# Symbol CRUD operations
# ############################


def get_symbol(symbol: str) -> Symbol:
    """
    Retrieves a Symbol object based on the given symbol.

    Parameters:
        symbol (str): The symbol to search for.

    Returns:
        Symbol: The Symbol object matching the given symbol.

    Raises:
        DoesNotExist: If no Symbol object is found for the given
    """
    return Symbol.get(Symbol.symbol == symbol.upper())


def _ensure_symbol(symbol: Union[Symbol, str]) -> Symbol:
    """
    Ensure that we have a Symbol object.
    If a string is provided, fetch the corresponding Symbol object.
    """
    if isinstance(symbol, str):
        return get_symbol(symbol)
    return symbol


def add_symbol(
    symbol: str,
    name: str,
    asset_class: str,
    sector: Optional[str] = None,
    industry: Optional[str] = None,
    exchange: Optional[str] = None,
    description: Optional[str] = None,
) -> Symbol:
    """Add a new symbol to the database."""
    with db.atomic():
        return Symbol.create(
            symbol=symbol,
            name=name,
            asset_class=asset_class,
            sector=sector,
            industry=industry,
            exchange=exchange,
            description=description,
        )


def list_symbols() -> List[Symbol]:
    """List all symbols in the database."""
    return list(Symbol.select())


def update_symbol(
    symbol: str,
    name: Optional[str] = None,
    asset_class: Optional[str] = None,
    sector: Optional[str] = None,
    industry: Optional[str] = None,
    exchange: Optional[str] = None,
    description: Optional[str] = None,
) -> Symbol:
    """Update an existing symbol in the database."""
    sym = Symbol.get(Symbol.symbol == symbol.upper())
    if name:
        sym.name = name
    if asset_class:
        sym.asset_class = asset_class
    if sector is not None:
        sym.sector = sector
    if industry is not None:
        sym.industry = industry
    if exchange is not None:
        sym.exchange = exchange
    if description is not None:
        sym.description = description
    sym.save()
    return sym


def delete_symbol(symbol: str) -> bool:
    """Delete a symbol from the database."""
    sym = Symbol.get(Symbol.symbol == symbol.upper())
    return bool(sym.delete_instance())


def get_asset_class(info: Dict[str, any]) -> str:
    """
    Determine the asset class based on the information from Yahoo Finance.

    Args:
        info (Dict[str, any]): The info dictionary from yfinance.

    Returns:
        str: The determined asset class.
    """
    if "quoteType" in info:
        quote_type = info["quoteType"].lower()
        if quote_type == "equity":
            return "stock"
        elif quote_type == "future":
            return "futures"
        elif quote_type in ["etf", "mutualfund", "currency", "commodity"]:
            return quote_type

    # Default to 'stock' if we can't determine the asset class
    return "stock"


# ############################
# SymbolConfig CRUD operations
# ############################


def create_symbol_config(
    symbol: Union[Symbol, str],
    collect_price_data: bool = True,
    collect_yfinance_news: bool = True,
    collect_technical_indicators: bool = True,
    collect_fundamental_data: bool = True,
) -> SymbolConfig:
    """
    Create a new symbol configuration.
    """
    symbol_obj = _ensure_symbol(symbol)
    return SymbolConfig.create(
        symbol=symbol_obj,
        collect_price_data=collect_price_data,
        collect_yfinance_news=collect_yfinance_news,
        collect_technical_indicators=collect_technical_indicators,
        collect_fundamental_data=collect_fundamental_data,
    )


def get_symbol_config(symbol: Union[Symbol, str]) -> SymbolConfig:
    """
    Retrieve the configuration for a specific symbol.
    """
    symbol_obj = _ensure_symbol(symbol)
    return SymbolConfig.get(SymbolConfig.symbol == symbol_obj)


def update_symbol_config(symbol: Union[Symbol, str], **kwargs) -> bool:
    """
    Update the configuration for a specific symbol.
    """
    symbol_obj = _ensure_symbol(symbol)
    query = SymbolConfig.update(**kwargs).where(SymbolConfig.symbol == symbol_obj)
    return query.execute() > 0


def delete_symbol_config(symbol: Union[Symbol, str]) -> bool:
    """
    Delete the configuration for a specific symbol.
    """
    symbol_obj = _ensure_symbol(symbol)
    query = SymbolConfig.delete().where(SymbolConfig.symbol == symbol_obj)
    return query.execute() > 0


def get_or_create_symbol_config(
    symbol: Union[Symbol, str],
) -> Tuple[SymbolConfig, bool]:
    """
    Get the existing symbol configuration or create a new one with default values.
    """
    symbol_obj = _ensure_symbol(symbol)
    return SymbolConfig.get_or_create(symbol=symbol_obj)


def list_symbol_configs() -> List[SymbolConfig]:
    """List all symbol configurations in the database."""
    return list(SymbolConfig.select())


# ############################
# ScrapingUrl CRUD operations
# ############################


def add_scraping_url(symbol: str, url: str, source: str) -> ScrapingUrl:
    """Add a new scraping URL for a symbol."""
    with db.atomic():
        symbol_obj = Symbol.get(Symbol.symbol == symbol.upper())
        return ScrapingUrl.create(symbol=symbol_obj, url=url, source=source)


def list_scraping_urls(symbol: Optional[str] = None) -> List[ScrapingUrl]:
    """List scraping URLs, optionally filtered by symbol."""
    query = ScrapingUrl.select(ScrapingUrl, Symbol).join(Symbol)
    if symbol:
        query = query.where(Symbol.symbol == symbol.upper())
    return list(query)


def update_scraping_url(
    id: int,
    url: Optional[str] = None,
    source: Optional[str] = None,
    is_active: Optional[bool] = None,
) -> ScrapingUrl:
    """Update an existing scraping URL."""
    scraping_url = ScrapingUrl.get_by_id(id)
    if url:
        scraping_url.url = url
    if source:
        scraping_url.source = source
    if is_active is not None:
        scraping_url.is_active = is_active
    scraping_url.save()
    return scraping_url


def delete_scraping_url(id: int) -> bool:
    """Delete a scraping URL."""
    scraping_url = ScrapingUrl.get_by_id(id)
    return bool(scraping_url.delete_instance())


# ############################
# News CRUD operations
# ############################


def create_news(
    title: str,
    url: str,
    source: str,
    content: Optional[str] = None,
    published_at: Optional[datetime] = None,
) -> News:
    """
    Create a news object with the given parameters.

    Args:
        title (str): The title of the news.
        url (str): The URL of the news.
        source (str): The source of the news.
        content (Optional[str], optional): The content of the news.
            Defaults to None.
        published_at (Optional[datetime], optional): The published date and
            time of the news. Defaults to None.

    Returns:
        News: The created news object.
    """
    with db.atomic():
        return News.create(
            title=title,
            url=url,
            source=source,
            content=content,
            published_at=published_at,
            scraped_at=datetime.now(UTC),
        )


def get_or_create_news(
    title: str,
    url: str,
    source: str,
    content: Optional[str] = None,
    published_at: Optional[datetime] = None,
    scraped_at: Optional[datetime] = None,
) -> Tuple[News, bool]:
    """
    Get or create a news object.

    Args:
        title (str): The title of the news.
        url (str): The URL of the news.
        source (str): The source of the news.
        content (Optional[str], optional): The content of the news.
            Defaults to None.
        published_at (Optional[datetime], optional): The published date and
            time of the news. Defaults to None.
        scraped_at (Optional[datetime], optional): The date and time the news was scraped.

    Returns:
        Tuple[News, bool]: A tuple containing the news object and a boolean
            indicating if the news was created.
    """
    return News.get_or_create(
        url=url,
        defaults={
            "title": title,
            "source": source,
            "content": content,
            "published_at": published_at,
            "scraped_at": scraped_at,
        },
    )


def get_news(news_id: int) -> News:
    """
    Retrieve a news item by its ID.

    Parameters:
        news_id (int): The ID of the news item to retrieve.

    Returns:
        News: The news item with the specified ID.

    Raises:
        DoesNotExist: If no news item with the specified ID is found.
    """
    return News.get_by_id(news_id)


def update_news(news_id: int, **kwargs) -> bool:
    """
    Update a news entry in the database.

    Args:
        news_id (int): The ID of the news entry to update.
        **kwargs: Keyword arguments representing the fields to update.

    Returns:
        bool: True if the news entry was successfully updated, False otherwise.
    """
    query = News.update(**kwargs).where(News.id == news_id)
    return query.execute() > 0


def delete_news(news_id: int) -> bool:
    """
    Deletes a news item from the database.

    Args:
        news_id (int): The ID of the news item to be deleted.

    Returns:
        bool: True if the news item was successfully deleted, False otherwise.
    """
    query = News.delete().where(News.id == news_id)
    return query.execute() > 0


# ############################
# NewsSymbol CRUD operations
# ############################


def create_news_symbol(news: News, symbol: Symbol) -> NewsSymbol:
    """
    Create a NewsSymbol object.

    Args:
        news (News): The news object.
        symbol (Symbol): The symbol object.

    Returns:
        NewsSymbol: The created NewsSymbol object.
    """
    return NewsSymbol.create(news=news, symbol=symbol)


def delete_news_symbol(news_symbol_id: int) -> bool:
    """
    Deletes a news symbol from the database.

    Args:
        news_symbol_id (int): The ID of the news symbol to be deleted.

    Returns:
        bool: True if the news symbol was deleted successfully, False otherwise.
    """
    query = NewsSymbol.delete().where(NewsSymbol.id == news_symbol_id)
    return query.execute() > 0


# ############################
# HistoricalPrice CRUD operations
# ############################


def create_historical_price(
    symbol: Union[Symbol, str],
    date: date,
    open_price: float,
    high: float,
    low: float,
    close: float,
    volume: int,
    dividends: float = 0.0,
    stock_splits: float = 0.0,
) -> HistoricalPrice1D:
    """
    Create a new historical price record.

    Args:
        symbol (Union[Symbol, str]): The Symbol object or symbol string.
        date (date): The date of this price record.
        open_price (float): The opening price.
        high (float): The highest price of the day.
        low (float): The lowest price of the day.
        close (float): The closing price.
        volume (int): The trading volume.
        dividends (float, optional): The dividend amount. Defaults to 0.0.
        stock_splits (float, optional): The stock split ratio. Defaults to 0.0.

    Returns:
        HistoricalPrice: The created HistoricalPrice object.
    """
    symbol_obj = _ensure_symbol(symbol)
    return HistoricalPrice1D.create(
        symbol=symbol_obj,
        date=date,
        open=open_price,
        high=high,
        low=low,
        close=close,
        volume=volume,
        dividends=dividends,
        stock_splits=stock_splits,
    )


def get_historical_price(symbol: Union[Symbol, str], date: date) -> HistoricalPrice1D:
    """
    Retrieve a specific historical price record.

    Args:
        symbol (Union[Symbol, str]): The Symbol object or symbol string.
        date (date): The date of the price record.

    Returns:
        HistoricalPrice: The retrieved HistoricalPrice object.

    Raises:
        DoesNotExist: If no matching record is found.
    """
    symbol_obj = _ensure_symbol(symbol)
    return HistoricalPrice1D.get(
        (HistoricalPrice1D.symbol == symbol_obj) & (HistoricalPrice1D.date == date)
    )


def get_historical_prices(
    symbol: Union[Symbol, str], start_date: date = None, end_date: date = None
) -> List[HistoricalPrice1D]:
    """
    Retrieve historical price records for a symbol within a date range.

    Args:
        symbol (Union[Symbol, str]): The Symbol object or symbol string.
        start_date (date, optional): The start date of the range (inclusive).
        end_date (date, optional): The end date of the range (inclusive).

    Returns:
        List[HistoricalPrice]: A list of HistoricalPrice objects.
    """
    try:
        symbol_obj = _ensure_symbol(symbol)
    except DoesNotExist as e:
        raise ValueError(f"Symbol '{symbol}' does not exist.") from e

    query = HistoricalPrice1D.select().where(HistoricalPrice1D.symbol == symbol_obj)

    if start_date:
        query = query.where(HistoricalPrice1D.date >= start_date)
    if end_date:
        query = query.where(HistoricalPrice1D.date <= end_date)

    return list(query.order_by(HistoricalPrice1D.date))


def update_historical_price(symbol: Union[Symbol, str], date: date, **kwargs) -> bool:
    """
    Update a specific historical price record.

    Args:
        symbol (Union[Symbol, str]): The Symbol object or symbol string.
        date (date): The date of the record to update.
        **kwargs: The fields to update and their new values.

    Returns:
        bool: True if the update was successful, False otherwise.
    """
    symbol_obj = _ensure_symbol(symbol)
    query = HistoricalPrice1D.update(**kwargs).where(
        (HistoricalPrice1D.symbol == symbol_obj) & (HistoricalPrice1D.date == date)
    )
    return query.execute() > 0


def delete_historical_price(symbol: Union[Symbol, str], date: date) -> bool:
    """
    Delete a specific historical price record.

    Args:
        symbol (Union[Symbol, str]): The Symbol object or symbol string.
        date (date): The date of the record to delete.

    Returns:
        bool: True if the deletion was successful, False otherwise.
    """
    symbol_obj = _ensure_symbol(symbol)
    query = HistoricalPrice1D.delete().where(
        (HistoricalPrice1D.symbol == symbol_obj) & (HistoricalPrice1D.date == date)
    )
    return query.execute() > 0


def bulk_upsert_historical_prices(
    symbol: Union[Symbol, str], price_data: List[Dict]
) -> int:
    """
    Bulk upsert (insert or update) historical price records for a symbol.

    Args:
        symbol (Union[Symbol, str]): The Symbol object or symbol string.
        price_data (List[Dict]): A list of dictionaries containing price data.
            Each dict should have keys: date, open, high, low, close, volume, dividends, stock_splits

    Returns:
        int: The number of records inserted or updated.
    """
    symbol_obj = _ensure_symbol(symbol)
    upserted_count = 0

    with db.atomic():
        for batch in helpers.chunked(price_data, 100):  # Process in batches of 100
            data_to_upsert = [{**data, "symbol": symbol_obj} for data in batch]

            for data in data_to_upsert:
                try:
                    HistoricalPrice1D.insert(data).on_conflict(
                        conflict_target=[
                            HistoricalPrice1D.symbol,
                            HistoricalPrice1D.date,
                        ],
                        update={
                            HistoricalPrice1D.open: data["open"],
                            HistoricalPrice1D.high: data["high"],
                            HistoricalPrice1D.low: data["low"],
                            HistoricalPrice1D.close: data["close"],
                            HistoricalPrice1D.volume: data["volume"],
                            HistoricalPrice1D.dividends: data["dividends"],
                            HistoricalPrice1D.stock_splits: data["stock_splits"],
                        },
                    ).execute()
                    upserted_count += 1
                except IntegrityError:
                    # Handle any integrity errors if necessary
                    pass

    return upserted_count


# ############################
# TechnicalAnalysis CRUD operations
# ############################


def create_technical_analysis(
    symbol: Union[Symbol, str],
    date: date,
    timeframe: Union[Timeframe, str] = Timeframe.DAYS_1,
    technical_indicators: Dict = None,
    candlestick_patterns: Dict = None,
) -> TechnicalAnalysis:
    """
    Create a new TechnicalAnalysis record.

    Args:
        symbol (Union[Symbol, str]): Symbol object or symbol string.
        date (date): Date of the analysis.
        timeframe (Union[Timeframe, str]): Timeframe of the analysis. Defaults to DAYS_1.
        technical_indicators (Dict, optional): Dictionary of technical indicators.
        candlestick_patterns (Dict, optional): Dictionary of candlestick patterns.

    Returns:
        TechnicalAnalysis: The created TechnicalAnalysis object.

    Raises:
        ValueError: If the symbol doesn't exist or if the timeframe is invalid.
    """
    try:
        symbol_obj = _ensure_symbol(symbol)
    except DoesNotExist as e:
        raise ValueError(f"Symbol '{symbol}' does not exist.") from e

    if isinstance(timeframe, str):
        try:
            timeframe = Timeframe(timeframe)
        except ValueError as e:
            raise ValueError(f"Invalid timeframe: {timeframe}") from e

    with db.atomic():
        return TechnicalAnalysis.create(
            symbol=symbol_obj,
            date=date,
            timeframe=timeframe.value,
            technical_indicators=technical_indicators or {},
            candlestick_patterns=candlestick_patterns or {},
        )


def update_technical_analysis(
    symbol: Union[Symbol, str],
    date: date,
    timeframe: Union[Timeframe, str] = Timeframe.DAYS_1,
    technical_indicators: Dict = None,
    candlestick_patterns: Dict = None,
) -> TechnicalAnalysis:
    """
    Update an existing TechnicalAnalysis record.

    Args:
        symbol (Union[Symbol, str]): Symbol object or symbol string.
        date (date): Date of the analysis.
        timeframe (Union[Timeframe, str]): Timeframe of the analysis. Defaults to DAYS_1.
        technical_indicators (Dict, optional): Dictionary of technical indicators to update.
        candlestick_patterns (Dict, optional): Dictionary of candlestick patterns to update.

    Returns:
        TechnicalAnalysis: The updated TechnicalAnalysis object.

    Raises:
        DoesNotExist: If the TechnicalAnalysis record doesn't exist.
        ValueError: If the symbol doesn't exist or if the timeframe is invalid.
    """
    try:
        symbol_obj = _ensure_symbol(symbol)
    except DoesNotExist as e:
        raise ValueError(f"Symbol '{symbol}' does not exist.") from e

    if isinstance(timeframe, str):
        try:
            timeframe = Timeframe(timeframe)
        except ValueError as e:
            raise ValueError(f"Invalid timeframe: {timeframe}") from e

    with db.atomic():
        analysis = TechnicalAnalysis.get(
            (TechnicalAnalysis.symbol == symbol_obj)
            & (TechnicalAnalysis.date == date)
            & (TechnicalAnalysis.timeframe == timeframe.value)
        )

        if technical_indicators:
            analysis.technical_indicators.update(technical_indicators)
        if candlestick_patterns:
            analysis.candlestick_patterns.update(candlestick_patterns)

        analysis.save()
        return analysis


def delete_technical_analysis(
    symbol: Union[Symbol, str],
    date: date,
    timeframe: Union[Timeframe, str] = Timeframe.DAYS_1,
) -> bool:
    """
    Delete a TechnicalAnalysis record.

    Args:
        symbol (Union[Symbol, str]): Symbol object or symbol string.
        date (date): Date of the analysis.
        timeframe (Union[Timeframe, str]): Timeframe of the analysis. Defaults to DAYS_1.

    Returns:
        bool: True if a record was deleted, False otherwise.

    Raises:
        ValueError: If the symbol doesn't exist or if the timeframe is invalid.
    """
    try:
        symbol_obj = _ensure_symbol(symbol)
    except DoesNotExist as e:
        raise ValueError(f"Symbol '{symbol}' does not exist.") from e

    if isinstance(timeframe, str):
        try:
            timeframe = Timeframe(timeframe)
        except ValueError as e:
            raise ValueError(f"Invalid timeframe: {timeframe}") from e

    with db.atomic():
        query = TechnicalAnalysis.delete().where(
            (TechnicalAnalysis.symbol == symbol_obj)
            & (TechnicalAnalysis.date == date)
            & (TechnicalAnalysis.timeframe == timeframe.value)
        )
        deleted_count = query.execute()
        return deleted_count > 0


def bulk_upsert_technical_analysis(
    symbol: Union[Symbol, str],
    timeframe: Union[Timeframe, str],
    data: Dict[str, Dict[str, Dict]],
) -> int:
    """
    Perform a bulk upsert operation for TechnicalAnalysis records.

    Args:
        symbol (Union[Symbol, str]): Symbol object or symbol string.
        timeframe (Union[Timeframe, str]): Timeframe of the analysis.
        data (Dict[str, Dict[str, Dict]]): Dictionary of dates with their corresponding
                                           technical indicators and candlestick patterns.
                                           Format:
                                           {
                                               'YYYY-MM-DD': {
                                                   'technical_indicators': {...},
                                                   'candlestick_patterns': {...}
                                               },
                                               ...
                                           }

    Returns:
        int: The number of records inserted or updated.

    Raises:
        ValueError: If the symbol doesn't exist, if the timeframe is invalid,
                    or if the date format is incorrect.
    """
    try:
        symbol_obj = _ensure_symbol(symbol)
    except DoesNotExist as e:
        raise ValueError(f"Symbol '{symbol}' does not exist.") from e

    if isinstance(timeframe, str):
        try:
            timeframe = Timeframe(timeframe)
        except ValueError as e:
            raise ValueError(f"Invalid timeframe: {timeframe}") from e

    upsert_count = 0

    with db.atomic():
        for date_str, analysis_data in data.items():
            try:
                date = datetime.strptime(date_str, "%Y-%m-%d").date()
            except ValueError as e:
                raise ValueError(
                    f"Invalid date format: {date_str}. Expected format: YYYY-MM-DD"
                ) from e

            technical_indicators = analysis_data.get("technical_indicators", {})
            candlestick_patterns = analysis_data.get("candlestick_patterns", {})

            analysis, created = TechnicalAnalysis.get_or_create(
                symbol=symbol_obj,
                date=date,
                timeframe=timeframe.value,
                defaults={
                    "technical_indicators": technical_indicators,
                    "candlestick_patterns": candlestick_patterns,
                },
            )

            if not created:
                analysis.technical_indicators.update(technical_indicators)
                analysis.candlestick_patterns.update(candlestick_patterns)
                analysis.save()

            upsert_count += 1

    return upsert_count


# ############################
# FundamentalData CRUD operations
# ############################


def get_fundamental_data(
    symbol: Union[Symbol, str],
    start_date: date = None,
    end_date: date = None,
    period: str = None,
) -> List[FundamentalData]:
    """
    Retrieves fundamental data for a given symbol within a specified date range and period.

    Args:
        symbol (Union[Symbol, str]): The symbol or symbol object for which to retrieve fundamental data.
        start_date (date, optional): The start date of the date range. Defaults to None.
        end_date (date, optional): The end date of the date range. Defaults to None.
        period (str, optional): The period of the fundamental data. Defaults to None.

    Returns:
        List[FundamentalData]: A list of fundamental data objects matching the specified criteria.
    """
    symbol_obj = _ensure_symbol(symbol)
    query = FundamentalData.select().where(FundamentalData.symbol == symbol_obj)

    if start_date:
        query = query.where(FundamentalData.date >= start_date)
    if end_date:
        query = query.where(FundamentalData.date <= end_date)
    if period:
        query = query.where(FundamentalData.period == period)

    return list(query.order_by(FundamentalData.date.desc()))


def create_fundamental_data(
    symbol: Union[Symbol, str], date: date, period: str, data: Dict[str, Any]
) -> FundamentalData:
    """
    Create fundamental data for a given symbol, date, period, and data.

    Parameters:
        symbol (Union[Symbol, str]): The symbol object or symbol string.
        date (date): The date of the fundamental data.
        period (str): The period of the fundamental data.
        data (Dict[str, Any]): The fundamental data.

    Returns:
        FundamentalData: The created fundamental data object.
    """
    symbol_obj = _ensure_symbol(symbol)
    return FundamentalData.create(
        symbol=symbol_obj, date=date, period=period, data=data
    )


def get_or_create_fundamental_data(
    symbol: Union[Symbol, str], date: date, period: str, data: Dict[str, Any]
) -> tuple[FundamentalData, bool]:
    """
    Get or create fundamental data for a given symbol, date, and period.

    Args:
        symbol (Union[Symbol, str]): The symbol object or symbol string.
        date (date): The date of the fundamental data.
        period (str): The period of the fundamental data.
        data (Dict[str, Any]): The data to be associated with the fundamental data.

    Returns:
        tuple[FundamentalData, bool]: A tuple containing the
            fundamental data object and a boolean indicating whether the data was created or not.
    """
    symbol_obj = _ensure_symbol(symbol)
    return FundamentalData.get_or_create(
        symbol=symbol_obj, date=date, period=period, defaults={"data": data}
    )


def update_fundamental_data(
    symbol: Union[Symbol, str], date: date, period: str, data: Dict[str, Any]
) -> bool:
    """
    Update the fundamental data for a given symbol, date, and period.

    Args:
        symbol (Union[Symbol, str]): The symbol or symbol object.
        date (date): The date of the fundamental data.
        period (str): The period of the fundamental data.
        data (Dict[str, Any]): The updated fundamental data.

    Returns:
        bool: True if the update was successful, False otherwise.
    """
    symbol_obj = _ensure_symbol(symbol)
    query = FundamentalData.update(data=data).where(
        (FundamentalData.symbol == symbol_obj)
        & (FundamentalData.date == date)
        & (FundamentalData.period == period)
    )
    return query.execute() > 0


def delete_fundamental_data(
    symbol: Union[Symbol, str], date: date, period: str
) -> bool:
    """
    Deletes the fundamental data for a given symbol, date, and period.

    Parameters:
        symbol (Union[Symbol, str]): The symbol or symbol object.
        date (date): The date of the fundamental data.
        period (str): The period of the fundamental data.

    Returns:
        bool: True if the fundamental data is deleted successfully, False otherwise.
    """
    symbol_obj = _ensure_symbol(symbol)
    query = FundamentalData.delete().where(
        (FundamentalData.symbol == symbol_obj)
        & (FundamentalData.date == date)
        & (FundamentalData.period == period)
    )
    return query.execute() > 0


def bulk_upsert_fundamental_data(
    symbol: Union[Symbol, str], data: List[Dict[str, Any]]
) -> int:
    """
    Upserts fundamental data in batches for a given symbol.

    Args:
        symbol (Union[Symbol, str]):
            The symbol or symbol object for which the fundamental data is being upserted.
        data (List[Dict[str, Any]]):
            A list of dictionaries containing the fundamental data to be upserted.
            Each dictionary should have the following keys: "date", "period", and "data".

    Returns:
        int: The number of records upserted.
    """
    symbol_obj = _ensure_symbol(symbol)
    upserted_count = 0

    with db.atomic():
        for batch in helpers.chunked(data, 100):  # Process in batches of 100
            for item in batch:
                try:
                    FundamentalData.insert(
                        symbol=symbol_obj,
                        date=item["date"],
                        period=item["period"],
                        data=item["data"],
                    ).on_conflict(
                        conflict_target=[
                            FundamentalData.symbol,
                            FundamentalData.date,
                            FundamentalData.period,
                        ],
                        update={"data": item["data"]},
                    ).execute()
                    upserted_count += 1
                except IntegrityError:
                    # Handle any integrity errors if necessary
                    pass

    return upserted_count
