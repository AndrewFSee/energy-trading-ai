"""Data ingestion layer for the Energy Trading AI system."""

from src.data.eia_client import EIAClient
from src.data.eia_demand_client import EIADemandClient
from src.data.eia_generation_client import EIAGenerationClient
from src.data.fred_client import FREDClient
from src.data.news_fetcher import NewsFetcher
from src.data.ng_production_client import NGProductionClient
from src.data.openmeteo_client import OpenMeteoClient
from src.data.price_fetcher import PriceFetcher
from src.data.weather_client import WeatherClient

__all__ = [
    "EIAClient",
    "EIADemandClient",
    "EIAGenerationClient",
    "FREDClient",
    "NewsFetcher",
    "NGProductionClient",
    "OpenMeteoClient",
    "PriceFetcher",
    "WeatherClient",
]
