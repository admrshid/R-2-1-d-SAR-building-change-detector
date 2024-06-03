import requests
from osm_changes.config import Config, TileFilepath
from datetime import datetime
from osm_changes.types import Coordinate
from osm_changes.logger import logger
import os


"""
NOTE: The older layers have a different color scheme, so the detector will not work with them until the color changes are mapped out and implemented (e.g. by changing detector properties according to which layers are selected).
"""
layer_urls: dict[str, str] = {
    "default": "https://tile.openstreetmap.org/",
    "202310": "https://os.openstreetmap.org/layer/gb_os_om_local_2023_10/",
    "202304": "https://os.openstreetmap.org/layer/gb_os_om_local_2023_04/",
    "202210": "https://os.openstreetmap.org/layer/gb_os_om_local_2022_10/",
    "202204": "https://os.openstreetmap.org/layer/gb_os_om_local_2022_04/",
    "202110": "https://os.openstreetmap.org/layer/gb_os_om_local_2021_10/",
    "202104": "https://os.openstreetmap.org/layer/gb_os_om_local_2021_04/",
    "202005": "https://os.openstreetmap.org/layer/gb_os_om_local_2020_05/",
    "202004": "https://os.openstreetmap.org/layer/gb_os_om_local_2020_04/",
    "201910": "https://os.openstreetmap.org/layer/gb_os_om_local_2019_10/",
    "201804": "https://os.openstreetmap.org/layer/gb_os_om_local_2018_04/",
    "201710": "https://os.openstreetmap.org/layer/gb_os_om_local_2017_10/",
    "201704": "https://os.openstreetmap.org/layer/gb_os_om_local_2017_04/",
    "201610": "https://os.openstreetmap.org/layer/gb_os_om_local_2016_10/",
    # "201604": "https://os.openstreetmap.org/layer/gb_os_sv_2016_04/",
    # "201511": "https://os.openstreetmap.org/layer/gb_os_sv_2015_11/",
    # "201505": "https://os.openstreetmap.org/layer/gb_os_sv_2015_05/",
    # "201411": "https://os.openstreetmap.org/layer/gb_os_sv_2014_11/",
    # "201404": "https://os.openstreetmap.org/layer/gb_os_sv_2014_04/",
    # "201311": "https://os.openstreetmap.org/layer/gb_os_sv_2013_11/",
    # "201305": "https://os.openstreetmap.org/layer/gb_os_sv_2013_05/",
    # "201211": "https://os.openstreetmap.org/layer/gb_os_sv_2012_11/",
    # "201205": "https://os.openstreetmap.org/layer/gb_os_sv_2012_05/",
    # "201111": "https://os.openstreetmap.org/layer/gb_os_sv_2011_11/",
    # "201105": "https://os.openstreetmap.org/layer/gb_os_sv_2011_05/",
    # "201011": "https://os.openstreetmap.org/layer/gb_os_sv_2010_11/",
    # "201004": "https://os.openstreetmap.org/layer/gb_os_sv_2010_04/",
}


def find_nearest_layer(given_date: str, base_urls: dict[str, str] = layer_urls) -> str:
    given_datetime = datetime.strptime(given_date, "%Y%m")
    nearest_date = min(
        base_urls.keys(),
        key=lambda x: abs(given_datetime - datetime.strptime(x, "%Y%m")),
    )
    return base_urls[nearest_date]


class Downloader:
    def __init__(self, cfg: Config):
        self.tile_url = "https://tile.openstreetmap.org/"
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:125.0) Gecko/20100101 Firefox/125.0"
        }
        self.zoom: int = cfg.zoom
        self.layer: str | None = None

    def get_tile_png(self, zoom: int, x: int, y: int) -> bytes:
        # https://tile.openstreetmap.org/17/65521/43969.png
        if self.layer is None:
            raise Exception("Layer not set, use this.set_layer(layer_name) to set the layer (where layer_name is a string YYYYMM, e.g. '202310' for October 2023)")
        url = f"{self.tile_url}{zoom}/{x}/{y}.png"
        logger.info(f"Downloading tile {zoom}/{x}/{y} from {url}")
        response = requests.get(url, headers=self.headers)
        # check if the request was successful and bytes were returned
        response.raise_for_status()
        return response.content

    def set_layer(self, layer: str):
        # check for the layer in the dictionary
        if layer not in layer_urls:
            nearest_layer = find_nearest_layer(layer)
            # get the key of the nearest layer
            nearest_layer_key = list(layer_urls.keys())[
                list(layer_urls.values()).index(nearest_layer)
            ]
            raise Exception(
                f"Layer {layer} not found, suggested layer: {nearest_layer_key} - {nearest_layer}"
            )
        self.tile_url = layer_urls[layer]
        self.layer = layer

    def save_tile(self, filepath: str, x: int, y: int, zoom: int | None = None):
        if zoom is None:
            zoom = self.zoom

        # skip if the file already exists
        if os.path.exists(filepath):
            logger.debug(f"File {filepath} already exists, skipping")
            return
        bytes = self.get_tile_png(zoom, x, y)
        if bytes:
            # create the directory if it does not exist
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            with open(filepath, "wb") as f:
                f.write(bytes)
        else:
            raise Exception(f"Failed to download tile {zoom}/{x}/{y}")

    def download_tiles(self, tiles: set[Coordinate]):
        if self.layer is None:
            raise Exception("Layer not set, use this.set_layer(layer_name) to set the layer (where layer_name is a string YYYYMM, e.g. '202310' for October 2023)")
        for tile in tiles:
            filepath = TileFilepath(self.layer, tile[0], tile[1], self.zoom)()
            self.save_tile(filepath, *tile)  # type: ignore


# def test_save_tile():
#     adapter = OSMAdapter()
#     bytes = adapter.get_tile_png(17, 65521, 43969)
#     # save the image bytes to a file
#     if bytes:
#         current_save_dir = os.path.dirname(os.path.realpath(__file__))
#         with open("tile.png", "wb") as f:
#             print(f"Saving file in {current_save_dir}")
#             f.write(bytes)
