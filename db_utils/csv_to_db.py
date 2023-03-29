from pathlib import Path

import pandas as pd
import pymongo


def import_csv_to_mongo(client: pymongo.MongoClient, csv_dir: Path = Path("csv_files")):
    existing_dbs = set(client.list_database_names())

    for csv_fname in [c for c in csv_dir.glob("*.csv") if c.name != "miasta.csv"]:
        df = pd.read_csv(csv_fname, sep=";")
        city = csv_fname.stem
        city_short_name = city.split("_")[0]

        if city in existing_dbs:
            client.drop_database(city)

        db = client[city]

        features_col_name = f"{city_short_name}_features"
        target_col_name = f"{city_short_name}_target"

        db.create_collection(features_col_name)
        db.create_collection(target_col_name)

        db[features_col_name].insert_many(df.drop(columns=["Cena"]).to_dict("records"))
        db[target_col_name].insert_many(df["Cena"].to_frame().to_dict("records"))


if __name__ == "__main__":
    client = pymongo.MongoClient("mongodb://localhost:27017/")
    import_csv_to_mongo(client)
