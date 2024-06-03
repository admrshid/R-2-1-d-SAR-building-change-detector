import matplotlib.pyplot as plt
import numpy as np
import io
from typing import Any
from osm_changes.types import Image
import os


def show_image(image: np.ndarray[Any, Any]) -> None:
    # Display the image using matplotlib.pyplot.imshow()
    plt.imshow(image)
    plt.axis("off")
    plt.show()


def show_images_in_row(images: list[Image]) -> None:
    # Create a 1 x n plot axis
    fig, ax = plt.subplots(1, len(images), figsize=(15, 5))

    for i, image in enumerate(images):
        ax[i].imshow(image)
        ax[i].axis("off")

    plt.show()


def show_images_in_grid(images: list[Image]):
    # sqrt of the number of images
    n = int(np.sqrt(len(images)))
    # Create a n x n plot axis
    fig, ax = plt.subplots(n, n, figsize=(15, 15))

    for i, image in enumerate(images):
        ax[i // n, i % n].imshow(image)
        ax[i // n, i % n].axis("off")

    plt.show()


def show_png_from_bytes(png_bytes: bytes) -> None:
    # Convert bytes to numpy array

    img: np.ndarray[Any, Any] = plt.imread(io.BytesIO(png_bytes), format="png")

    # Display the image using matplotlib.pyplot.imshow()
    plt.imshow(img)  # type: ignore
    plt.axis("off")  # type: ignore
    plt.show()  # type: ignore


def show_png_difference(png_bytes1: bytes, png_bytes2: bytes) -> None:
    # Convert bytes to numpy array. We have to be careful here to not remove the alpha channel!
    img1 = plt.imread(io.BytesIO(png_bytes1), format="png")[
        :, :, :3
    ]  # Select RGB channels only
    img2 = plt.imread(io.BytesIO(png_bytes2), format="png")[
        :, :, :3
    ]  # Select RGB channels only

    diff_img = np.abs(img1 - img2)
    diff_img = np.clip(diff_img, 0, 1)  # Ensure values are within [0, 1]

    new_build_color = (0.00392, 0.129, 0.247)  # RGB color for new buildings
    new_build_img = highlight_pixels(diff_img, new_build_color)
    plt.imshow(new_build_img)
    plt.axis("off")
    plt.title(f"New build image")
    plt.show()
    print("!")


def highlight_pixels(diff_img: np.ndarray, rgb_value: tuple) -> np.ndarray:
    """
    Highlight pixels in diff_img that match the specified RGB value.

    Parameters:
        diff_img (np.ndarray): Difference image array.
        rgb_value (tuple): RGB value to highlight, e.g., (r, g, b).

    Returns:
        np.ndarray: Copy of diff_img with highlighted pixels.
    """
    tolerance = 1 / 256  # Tolerance for pixel matching

    # TODO MASK TEXT HERE, AND OTHER HIGH FREQUENCY THINGS

    # Create a boolean mask for pixels matching the specified RGB value within a tolerance
    mask = np.all(np.abs(diff_img - rgb_value) < tolerance, axis=-1)
    return mask


def one_image_example() -> None:
    # Example usage
    with open("./output/32449_21776_16.png", "rb") as f:
        png_bytes = f.read()
        show_png_from_bytes(png_bytes)


def two_image_example() -> None:
    # Example usage
    with open("./output/202310_32449_21776_16.png", "rb") as f1, open(
        "./output/201505_32450_21776_16.png", "rb"
    ) as f2:
        png_bytes1 = f1.read()
        png_bytes2 = f2.read()
        show_png_difference(png_bytes1, png_bytes2)


if __name__ == "__main__":
    # get the tiles from the output directory
    outdir_files = os.listdir("./output")
    # filter to find those from 202310 (prefix of filename)
    layer1_files = [f for f in outdir_files if f.startswith("202310")]

    # to test, lets just use 202310_32449_21776_16.png
    # layer1_files = ["202310_32449_21776_16.png"]
    # loop through the files
    for filename in layer1_files:
        print(f"Showing {filename}")
        # find the corresponding file from 201505
        filename2 = filename.replace("202310", "201610")
        # check if the file exists
        if filename2 in outdir_files:
            # open the files
            with open(f"./output/{filename}", "rb") as f1, open(
                f"./output/{filename2}", "rb"
            ) as f2:
                png_bytes1 = f1.read()
                png_bytes2 = f2.read()
                show_png_difference(png_bytes1, png_bytes2)
                plt.draw()
        else:
            print(f"File {filename2} not found")
