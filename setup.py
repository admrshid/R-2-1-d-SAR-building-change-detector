from setuptools import setup, find_packages

setup(
    name="building_change_detector",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "pytest",
        "coverage",
        "numpy",
        "requests",
        "requests_mock",
        "matplotlib",
        "rasterio",
        "tensorflow",
        "scikit-image",
        "keras",
        "seaborn",
        "fnmatch",
        "einops",
        "collections"
    ],
    include_package_data=True,
    package_data={"osm_changes": ["config/*", "default.json"], 
                  "run": ["get_label.bat","get_video.bat","model_run.bat","model_train_synthetic.bat"]},
    test_suite="tests",
)
