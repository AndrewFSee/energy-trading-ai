"""Data ingestion layer for the Energy Trading AI system."""

from src.data.eia_client import EIAClient
from src.data.fred_client import FREDClient
from src.data.news_fetcher import NewsFetcher
from src.data.price_fetcher import PriceFetcher
from src.data.weather_client import WeatherClient

__all__ = [
    "EIAClient",
    "FREDClient",
    "NewsFetcher",
    "PriceFetcher",
    "WeatherClient",
]
