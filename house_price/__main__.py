import argparse

# from pathlib import Path
import sys

import pymongo

from house_price.gui import run as run_gui
from house_price.houses_prices_GUI_models import Models
from house_price.houses_prices_GUI_scraping import OtoDomWebScraping


def oto_webscraping(args):
    p = OtoDomWebScraping(
        args.rodzaj,
        args.city,
        args.radius,
        args.min_price,
        args.max_price,
        args.min_area,
        args.max_area,
    )
    p.WebScraping()
    p.Convert_and_Clean_Date()
    p.Save_to_Mongo()


def models(args):
    if args.db == "mongo":
        mongo_db = pymongo.MongoClient("mongodb://localhost:27017/")
        list_of_db = mongo_db.list_database_names()
        list_of_db.remove("admin")
        list_of_db.remove("config")
        list_of_db.remove("local")
        for db_name in list_of_db:
            print(db_name)
    else:
        p = Models(args.db)
        p.load_from_Mongo()
        p.Split_Date_for_Test_and_Train()
        if args.model == 1:
            p.LinearRegression()
        elif args.model == 2:
            p.RandomForest()
        elif args.model == 3:
            p.xgboost()
        elif args.model == 4:
            p.DecisionTreeRegressor()
        elif args.model == 55:
            p.SVR()
        elif args.model == 6:
            p.Neural_Network()


def gui(args):
    run_gui()


if __name__ == "__main__":
    argparse = argparse.ArgumentParser(description="Process some integers.")
    subparsers = argparse.add_subparsers()

    parser_gui = subparsers.add_parser("gui")
    parser_gui.add_argument("-x", type=str, default="")
    parser_gui.set_defaults(func=gui)

    parser_models = subparsers.add_parser("models")
    parser_models.add_argument(
        "-db", default="mongo", help="show all db in mongo", metavar=""
    )
    parser_models.add_argument("-model", choices=range(1, 7), type=int, metavar="")
    parser_models.set_defaults(func=models)

    parser_web = subparsers.add_parser("oto_webscraping")
    parser_web.add_argument(
        "-w1",
        "--rodzaj",
        type=str,
        metavar="",
        help="apartamentowiec, blok",
        required=True,
    )
    parser_web.add_argument("-w2", "--city", type=str, metavar="", required=True)
    parser_web.add_argument("-w3", "--radius", type=int, metavar="", default=0)
    parser_web.add_argument("-w4", "--min_price", type=int, metavar="", default=0)
    parser_web.add_argument(
        "-w5", "--max_price", type=int, metavar="", default=10000000
    )
    parser_web.add_argument("-w6", "--min_area", type=int, metavar="", default=0)
    parser_web.add_argument("-w7", "--max_area", type=int, metavar="", default=10000000)
    parser_web.set_defaults(func=oto_webscraping)

    args = argparse.parse_args()
    args.func(args)
