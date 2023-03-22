
## House Price Prediction

The main goal of the application is to predict the cost of flats according to entered data by users and data scraped from the website www.OtoDom.pl.
In order to get the best results program creates a 6 regression model which compares and extracts the model with the smallest error.

## Models used in the program:
1. Linear Regression
2. Decision Tree Regressor
3. Random Forest Regressor
4. SVR
5. XGBoost
6. Neural network

## MongoDB
* To run the application, MongoDB is required

1. Download and install from: 
<https://www.mongodb.com/products/compass>

2. Use Docker to use a container
<https://hub.docker.com/_/mongo>

## Setup
* Install all packages
```bash
pip install -e .
```

## Before the first usage
* Module to load all .csv files from the folder to Mongodb. Required only once before the first usage
```bash 
python house_price/houses_prices_before_the_first_useage.py
```
## Usage
```bash 
python house_price/__main__.py
```

* Run the main GUI
```bash 
python house_price/__main__.py  gui
```

* Print all recordes in MongoDB
```bash 
python house_price/__main__.py  models -db mongo
```

* Scraping data from <https://www.otodom.pl/>
```bash 
python house_price/__main__.py oto_webscraping -w1 mieszkanie -w2 szczecin
```


## Tabs in GUI:

1. TAB_1
The first tab is for scraping data from a website. Can choose the place, price, and area.
After finished downloading, data are cleaned and saved to MongoDB and .csv files.
Section "domy" , and "działaki" are unavailable yet.

![window3](https://user-images.githubusercontent.com/67312266/152689372-e6620ec0-0353-42c8-87f4-3171d3255ff5.PNG)


2. TAB_2
We can use the second tab to visualization our data, saved before. There are 6 models to compare: 
Each model is split into 2 section Train and Test set, each of them consist of MSE error.

![window5](https://user-images.githubusercontent.com/67312266/152689376-28c8af35-d456-4027-aa0b-3ef89f70ae02.PNG)
![window7](https://user-images.githubusercontent.com/67312266/152689379-67f45555-e320-40d2-b5ea-d98c6392e392.PNG)



3. TAB_3
The last tab allows us to estimate the cost of a flat for the chosen place.
In spite of all models working fine and data are up to date there are some points to improve. 
For example, data don't include the price of parking place which raise the total cost.

![3](https://user-images.githubusercontent.com/67312266/152689385-61fd1da6-735c-46f8-bcdd-7c6f703709d3.PNG)
 
##



## Note!

In connection with a website otodom.pl is still changing ( name of class , span etc).Program requires updating all time.

• The last update , January 2022

