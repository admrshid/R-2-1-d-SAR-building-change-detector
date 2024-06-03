"""Load configuration file"""

import os

from osm_changes.logger import logger, configure_logger

SUPPORTED_OUTPUTS = ["png", "tiff"]


class TileFilepath:
    """
    Used to automatically generate filepaths for tiles. 
    
    The output_dir and config_cwd must be set by initialising a Config() object before using this class.
    """
    output_dir: str | None = None
    config_cwd: str | None = None

    def __init__(
        self, layer: str, x: int | float, y: int | float, zoom: int, output: str = "png"
    ):
        if self.output_dir is None or self.config_cwd is None:
            raise Exception("Filepath not initialized, use config.init_filepaths()")

        if output not in SUPPORTED_OUTPUTS:
            raise RuntimeError(f"Unsupported output type {output}")

        self.output = output

        # convert relative path to absolute path
        if not os.path.isabs(self.output_dir):
            self.output_dir = os.path.normpath(
                os.path.join(self.config_cwd, self.output_dir)
            )
        self.layer = layer
        # Have to convert x and y to int, as they are sometimes floats which makes the path weird
        self.x = int(x)
        self.y = int(y)
        self.zoom = zoom

    def __str__(self):
        if self.output == "png":
            return f"{self.output_dir}/{self.layer}/{self.zoom}/{self.x}/{self.y}.png"
        elif self.output == "tiff":
            return f"{self.output_dir}/{self.layer}/{self.x}_{self.y}_{self.zoom}.tiff"
        else:
            raise RuntimeError(f"Unknown output type {self.output}")

    def __repr__(self):
        return self.__str__()

    def __call__(self):
        return self.__str__()



class Config:
    def __init__(self):
        self._cwd = os.getcwd()
        self.filepath = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "config", "default.json"
        )
        # configure logger
        configure_logger()

        self.load()
        self.init_filepaths()

    def set_output_dir(self, new_output_dir: str):
        self.output_dir = new_output_dir
        self.init_filepaths()

    def set_cwd(self, new_cwd: str):
        self._cwd = new_cwd
        self.init_filepaths()

    def load(self):
        import json

        with open(self.filepath, "r") as file:
            self.data = json.load(file)
            keys = self.data.keys()

            if "log_level" in keys:
                lvl = self.data["log_level"]
                configure_logger(lvl)
                logger.info(f"Log level set to {lvl}")

            self.config = self.data["config"]
            self.zoom = self.data["zoom"]

            self.layer1 = self.data["layer1"]
            self.layer2 = self.data["layer2"]

            self.initial_label = self.data["initial_label"]
            self.final_label = self.data["final_label"]

            self.height: int = self.data["height"]
            self.width: int = self.data["width"]

            self.min_lat = self.data["min_latitude"]
            self.max_lat = self.data["max_latitude"]
            self.min_lon = self.data["min_longitude"]
            self.max_lon = self.data["max_longitude"]

            # calculate step sizes
            self.lat_step = (self.max_lat - self.min_lat) / self.height
            self.lon_step = (self.max_lon - self.min_lon) / self.width

            self.output = self.data["output"]

            if self.output not in SUPPORTED_OUTPUTS:
                raise RuntimeError(f"Unsupported output type {self.output}, supported types: {SUPPORTED_OUTPUTS}")

            self._output_dir = self.data["output_dir"]

            # check the output directory exists
            if not os.path.exists(self._output_dir):
                os.makedirs(self._output_dir)
                logger.info(f"Created output directory {self._output_dir}")

    def init_filepaths(self):
        TileFilepath.output_dir = self._output_dir
        TileFilepath.config_cwd = self._cwd


