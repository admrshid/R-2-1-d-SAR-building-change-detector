""" Takes two images and detects a given change between them. """

from osm_changes.types import Color, Image, Coordinate
from osm_changes.config import Config, TileFilepath, SUPPORTED_OUTPUTS
from osm_changes.images import bytes_to_image
import osm_changes.display
from osm_changes.logger import logger
import numpy as np
import os


def normalize_difference(x: float | Image) -> float | Image:
    return x / 2 + 0.5


class Detector:
    # change_colors: dict[str, Color] = {
    #     "new_build": (0.502, 0.565, 0.624),
    #     "new_build_temp": (0.00392, 0.129, 0.247),
    # }

    class_colors: dict[str, Color] = {
        "Building": (0.973, 0.847, 0.722),
        "Nothing": (0.976, 0.976, 0.969),
        "Text": (0.0, 0.0, 0.0),
    }

    def __init__(
        self,
        config: Config,
        overwrite: bool = False,
    ) -> None:
        self.zoom = config.zoom
        self.overwrite = overwrite
        self.output = config.output

        if self.output not in SUPPORTED_OUTPUTS:
            raise RuntimeError(f"Unsupported output type {self.output}")

        self.initial_label = ""
        self.final_label = ""
        self.set_layers(config.layer1, config.layer2)
        self.set_target(config.initial_label, config.final_label)

    def update_layer_name(self) -> None:
        self.layerName = f"{self.layer1}To{self.layer2}Detected{self.initial_label}To{self.final_label}"

    def set_layers(self, layer1: str, layer2: str) -> None:
        self.layer1 = layer1
        self.layer2 = layer2
        self.update_layer_name()

    def set_target(self, initial_label: str, final_label: str) -> None:
        self.initial_label = initial_label
        self.final_label = final_label
        self.update_layer_name()

    def compare_image_files(self, tile1_filepath: str, tile2_filepath: str):
        with open(tile1_filepath, "rb") as f1, open(tile2_filepath, "rb") as f2:
            tile1 = bytes_to_image(f1.read(), format="png")
            tile2 = bytes_to_image(f2.read(), format="png")
            self.compare_difference_image(tile1, tile2)

    def subtract_color(self, color1: Color, color2: Color) -> Color:
        """Compute the color difference between two colors, color1 - color2."""
        x = tuple(float(normalize_difference(a - b)) for a, b in zip(color1, color2))[
            0:3
        ]
        assert len(x) == 3  # for type checker
        return x

    def get_difference(self, tile1: Image, tile2: Image) -> Image:
        # Compare two images
        diff_img = np.abs(tile1 - tile2)
        # divide by 2 and add 0.5 to normalize the values
        diff_img = normalize_difference(diff_img)
        # Ensure values are within [0, 1]
        diff_img = np.clip(diff_img, 0, 1)
        return diff_img

    def compare_difference_image(
        self, tile1_or_difference: Image, tile2: Image | None = None
    ) -> Image:
        if tile2 is None:
            # If only one image is given, assume it is the difference image
            diff_img = tile1_or_difference
        else:
            # If two images are given, compute the difference
            diff_img = self.get_difference(tile1_or_difference, tile2)

        color1 = self.class_colors[self.initial_label]
        color2 = self.class_colors[self.final_label]
        new_build_color = self.subtract_color(color1, color2)
        new_build_img = self.detect_difference(diff_img, new_build_color)

        # osm_changes.display.show_image(new_build_img)
        return new_build_img

    def detect_difference(self, diff_img: Image, rgb_value: Color) -> Image:
        """
        Highlight pixels in diff_img that match the specified RGB value.
        """
        # Tolerance for pixel matching
        tolerance = 1 / 256

        # Ignore alpha channel
        if diff_img.shape[2] == 4:
            diff_img = diff_img[:, :, :3]

        # Create a boolean mask for pixels matching the specified RGB value within a tolerance
        mask = np.all(np.abs(diff_img - rgb_value) < tolerance, axis=-1)
        return mask

    def color_mask(
        self, mask: Image, color: Color | str, tolerance: float = 0.1
    ) -> Image:
        if isinstance(color, str):
            if color in self.class_colors:
                color = self.class_colors[color]
            # elif color in self.change_colors:
            #     color = self.change_colors[color]
            else:
                raise ValueError(
                    f"Color {color} not found in class_colors or change_colors"
                )

        # Ignore alpha channel
        if mask.shape[2] == 4:
            mask = mask[:, :, :3]

        # Create a boolean mask for pixels matching the specified RGB value within a tolerance
        mask = np.all(np.abs(mask - color) < tolerance, axis=-1)
        return mask

    def detect_change(self, img1: Image, img2: Image):
        """A more robust methods if to find where the pixel is type A in the first image and type B in the second image"""
        mask1 = self.color_mask(img1, self.initial_label)
        mask2 = self.color_mask(img2, self.final_label)
        mask = mask1 & mask2
        return mask

    def detect_changes_in_tiles(self, tiles: set[Coordinate]):
        for tile in tiles:
            new_filepath = TileFilepath(
                self.layerName, tile[0], tile[1], self.zoom, output=self.output
            )()
            if not self.overwrite and os.path.exists(new_filepath):
                logger.debug(f"File {new_filepath} already exists, skipping")
                continue
            # make output directory if it does not exist
            os.makedirs(os.path.dirname(new_filepath), exist_ok=True)
            detection_mask = self.detect_changes_in_tile(tile)
            logger.info(f"Saving detection mask to {new_filepath}")

            # save the detection mask
            if self.output == "png":
                from matplotlib.pyplot import imsave  # type: ignore

                imsave(new_filepath, detection_mask, cmap="gray")
            elif self.output == "tiff":
                from osm_changes.images import tile_to_geotiff

                tile_to_geotiff(
                    detection_mask, int(tile[0]), int(tile[1]), self.zoom, new_filepath
                )
            else:
                raise RuntimeError(f"Unknown output type {self.output}")

    def detect_changes_in_tile(self, tile: Coordinate):
        fp1 = TileFilepath(self.layer1, tile[0], tile[1], self.zoom)()
        fp2 = TileFilepath(self.layer2, tile[0], tile[1], self.zoom)()

        # check if the files exist
        if not os.path.exists(fp1) or not os.path.exists(fp2):
            raise FileNotFoundError(
                f"Files not found, please download the tiles first using e.g. the downloader.py or main script:\n{fp1}\n{fp2}"
            )

        with open(fp1, "rb") as f1, open(fp2, "rb") as f2:
            img1 = bytes_to_image(f1.read(), format="png")
            img2 = bytes_to_image(f2.read(), format="png")
            detection_mask = self.detect_change(img1, img2)

            return detection_mask


def example():
    # f1p = "./output/201610_32449_21776_16.png"
    # f2p = "./output/202310_32449_21776_16.png"
    f1p = TileFilepath("201610", 32449, 21776, 16)()
    f2p = TileFilepath("202310", 32449, 21776, 16)()

    # check if the files exist
    if not os.path.exists(f1p) or not os.path.exists(f2p):
        raise FileNotFoundError(
            f"Files not found, please download the tiles first using e.g. the downloader.py or main script:\n{f1p}\n{f2p}"
        )

    # open each file
    img1 = bytes_to_image(open(f1p, "rb").read(), format="png")
    img2 = bytes_to_image(open(f2p, "rb").read(), format="png")

    # create detector and compare images
    detector = Detector(Config())
    diff_img = detector.get_difference(img1, img2)
    detection_mask1 = detector.compare_difference_image(img1, img2)
    detection_mask2 = detector.detect_change(img1, img2)

    # display all show_images_in_row
    osm_changes.display.show_images_in_grid(
        [img1, img2, diff_img, detection_mask1 & detection_mask2]
    )


if __name__ == "__main__":
    example()
