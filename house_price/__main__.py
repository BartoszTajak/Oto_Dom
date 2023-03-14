import argparse
from pathlib import Path
from houses_prices_GUI import main_gui
from houses_prices_GUI_models import Models
from houses_prices_GUI_scraping import OtoDomWebScraping




if __name__ == "__main__":
    argparse = argparse.ArgumentParser('Script to searching, compering and predict price of houses')
    argparse.add_argument("choices", choices=['gui', 'models', 'scraping'], help='Options include: gui , models , scraping.')
    args = argparse.parse_args()
    if args.choices == "gui":
        main_gui()
