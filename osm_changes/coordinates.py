"""Tools for converting between different coordinate systems.
Mainly the EPSG:3857 (WGS 84 / Pseudo-Mercator) tile system used by OSM and WGS84 (EPSG:4326) used by everything else.
"""
import math
from .types import Coordinate


def latlon_to_tile(lat: float, lon: float, zoom: int, img_size: int = 256) -> tuple[Coordinate, Coordinate]:
    """Convert lat/lon to OSM tile coordinates
    :param lat: float
    :param lon: float
    :return: tuple[Coordinate, Coordinate]
    """

    tile_x = (lon + 180) / 360 * 2 ** zoom
    tile_y = (1 - math.log(math.tan(math.radians(lat)) + 1 / math.cos(math.radians(lat))) / math.pi) / 2 * 2 ** zoom
    pixel_x = round((tile_x - int(tile_x)) * img_size)
    pixel_y = round((tile_y - int(tile_y)) * img_size)

    return (int(tile_x), int(tile_y)), (pixel_x, pixel_y)


def tile_to_latlon(tile_x: int, tile_y: int, zoom: int, img_size: int = 256) -> tuple[float, float]:
    """Convert OSM tile coordinates to lat/lon
    :param tile_x: int
    :param tile_y: int
    :return: tuple[float, float]
    """

    n = 2 ** zoom
    lon = tile_x / n * 360.0 - 180.0
    lat = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * tile_y / n))))

    return lat, lon
