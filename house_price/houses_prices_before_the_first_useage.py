import os
from pathlib import Path

import pymongo
import pandas as pd


mongo_db = pymongo.MongoClient("mongodb://localhost:27017/")


def from_csv_to_mongo():

    my_path = Path(os.getcwd()) / 'csv_files'
    list_of_city = os.listdir(my_path)
    list_of_city.remove('miasta.csv')
    list_of_city = [i.replace('.csv','') for i in list_of_city]
    for city in list_of_city:
        df_clear = pd.read_csv(my_path/f'{city}.csv', sep=';')
        city_short = city.split('_')[0]
        # name our base
        db_Mongo_name =  city
        # deleting base if exist
        for db_n in mongo_db.list_database_names():
            if (db_n == db_Mongo_name):
                mongo_db.drop_database(db_Mongo_name)

        # create base
        flats_db_date = mongo_db[db_Mongo_name]
        # split frame to X and Y
        x_features = df_clear.drop(columns=['Cena'])
        y_price = df_clear['Cena']
        flats_db_x = city_short + '_' + 'flats_x'
        flats_db_y = city_short + '_' + 'flats_y'
        flats_db_date[flats_db_x].insert_many(x_features.to_dict('records'))
        flats_db_date[flats_db_y].insert_many(y_price.to_frame().to_dict('records'))


from_csv_to_mongo()
