from osm_changes.logger import logger
from osm_changes.config import Config
from osm_changes.grid import Grid
from osm_changes.coordinates import Coordinate, latlon_to_tile
from osm_changes.downloader import Downloader
from osm_changes.detector import Detector

def main():

    # load configuration
    cfg = Config()
    
    # create grid
    grid = Grid(cfg)

    # create a set of tiles
    tiles: set[Coordinate] = set()

    # for each point in the grid, find the tile
    for point in grid.grid:
        tile = latlon_to_tile(point[0], point[1], cfg.zoom)[0]
        tiles.add(tile)

    logger.info(f"Number of tiles: {len(tiles)}")

    for layer in [cfg.layer1, cfg.layer2]:
        logger.info(f"Downloading tiles for layer {layer}")
        downloader = Downloader(cfg)
        downloader.set_layer(layer)
        downloader.download_tiles(tiles)
        logger.info("Layer download complete")

    logger.info("Detecting changes in tiles")
    detector = Detector(cfg)
    detector.detect_changes_in_tiles(tiles)
    logger.info("Detection complete")


if __name__ == "__main__":
    main()
