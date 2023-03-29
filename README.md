
## House Price Prediction
The primary objective of the application is to estimate the prices of apartments based on user input and data obtained through web scraping. \
To achieve optimal accuracy, the program generates six regression models and selects the one with the smallest error.

## Models used in the program:
1. Linear Regression
2. Decision Tree Regressor
3. Random Forest Regressor
4. SVR
5. XGBoost
6. Neural network

## Prerequisites
1. Start a MongoDB database instance
2. Retrieve and store the training data in the directory called `csv_files`. Contact the project's author for further details.
3. Migrate training data to the database:
```bash
python db_utils/csv_to_db.py
```

## Setup
```bash
pip install -e .
```

## Usage
Start application
```bash
python3 -m house_price gui
```

### Scraping tab
Choose search parameters in order to parse, download and store the data in the database and csv files.

![window3](https://user-images.githubusercontent.com/67312266/152689372-e6620ec0-0353-42c8-87f4-3171d3255ff5.PNG)
Sections "domy" and "działki" are unavailable yet.

### Model tab
Use your fetched data to train and compare your chosen ML models.
![window5](https://user-images.githubusercontent.com/67312266/152689376-28c8af35-d456-4027-aa0b-3ef89f70ae02.PNG)

![window7](https://user-images.githubusercontent.com/67312266/152689379-67f45555-e320-40d2-b5ea-d98c6392e392.PNG)

### Inference tab
Estimate the cost of a flat for the chosen place.
![3](https://user-images.githubusercontent.com/67312266/152689385-61fd1da6-735c-46f8-bcdd-7c6f703709d3.PNG)