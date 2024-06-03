from osm_changes.coordinates import tile_to_latlon
from osm_changes.types import Image
from io import BytesIO
import numpy as np
import rasterio  # type: ignore
from rasterio.warp import reproject, Resampling  # type: ignore


def bytes_to_image(binary_image_data: bytes, format: str) -> Image:
    from matplotlib.pyplot import imread

    # Convert binary image data to an image
    image: Image = imread(BytesIO(binary_image_data), format=format)
    return image


def tile_to_geotiff(image: Image, x: int, y: int, z: int, filename: str):
    with rasterio.Env():
        # check if we're 2D:
        if len(image.shape) != 3:
            # reshape to (1xHxW)
            image = np.expand_dims(image, axis=0)
        else:
            if image.shape[0] == 4:  # trim the alpha channel
                image = image[:3]
            if image.shape[2] == 3 or image.shape[2] == 1:  # rasterio expects 3x256x256, not 256x256x3
                image = np.moveaxis(image, -1, 0)
        band_count = image.shape[0]

        # convert any bool images to uint8
        if image.dtype == bool:
            image = image.astype(np.uint8) * 255

        dst_shape = image.shape

        destination = image.copy()

        src_crs = {"init": "EPSG:3857"}
        dst_crs = {"init": "EPSG:4326"}

        # Get the lat/lon bounds of the tile.
        lat1, lon1 = tile_to_latlon(x, y, z)
        lat2, lon2 = tile_to_latlon(x + 1, y + 1, z)

        # Get the transform for the source image.
        src_transform: rasterio.Affine = rasterio.transform.from_bounds(  # type: ignore
            lon1, lat2, lon2, lat1, 256, 256
        )
        dst_transform: rasterio.Affine = rasterio.transform.from_bounds(  # type: ignore
            lon1, lat2, lon2, lat1, 256, 256
        )

        reproject(
            image,
            destination,
            src_transform=src_transform,
            src_crs=src_crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            resampling=Resampling.cubic,
        )

        # Write it out to a file.
        with rasterio.open(  # type: ignore
            filename,
            "w",
            driver="GTiff",
            width=dst_shape[2],
            height=dst_shape[1],
            count=band_count,
            dtype=image.dtype,
            transform=dst_transform,
            crs=dst_crs,
        ) as dst:  # type: ignore
            for i in range(0, band_count):
                dst.write(image[i], i + 1)  # type: ignore


def tile_to_geotiff_example():
    # get out of our tiles:
    fp = "./output/201610/16/32449/21779.png"  # a 256x256x3 tile, z/x/y.png
    x = 32449
    y = 21779
    z = 16

    ofp = "exampletiff.tif"
    with open(fp, "rb") as f1:
        img1 = bytes_to_image(f1.read(), format="png")
        img1 = img1[:, :, :3]  # trim the alpha channel
        tile_to_geotiff(img1, x, y, z, ofp)
