from distutils.core import setup

setup(
    name="house_price",
    version="1.0",
    packages=["house_price"],
    install_requires=[
        "bs4",
        "geopy",
        "googlemaps",
        "matplotlib",
        "pandas",
        "Pillow",
        "pymongo",
        "PySide6",
        "python-dotenv",
        "scikit-learn",
        "tensorflow",
        "xgboost",
        "numpy",
        "pre-commit",
    ],
)
