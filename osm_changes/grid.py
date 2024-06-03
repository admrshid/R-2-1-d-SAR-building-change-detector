""" Define a grid of lat/lon points """

from .config import Config
from .coordinates import Coordinate
from numpy import linspace


class Grid:
    def __init__(self, config: Config):
        self.config = config
        self.lat_points: list[float] = []
        self.lon_points: list[float] = []
        self.grid: list[Coordinate] = []
        self.generate_grid()

    def generate_grid(self):
        self.lat_points = linspace(
            self.config.min_lat, self.config.max_lat, self.config.height
        )
        self.lon_points = linspace(
            self.config.min_lon, self.config.max_lon, self.config.width
        )
        self.grid = [(lat, lon) for lat in self.lat_points for lon in self.lon_points]  # type: ignore
