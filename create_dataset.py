#TO CREATE A SMALLER DATASET
# #Cut out a small csv from the HandInfo.csv file manually
#Provide path to that csv, path to the images folder and path to output folder

import pandas as pd
import argparse
import shutil

def prepare_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--csv-path', type=str)
    parser.add_argument('-d', '--data-path', type=str)
    parser.add_argument('-o', '--output-path', type=str)
    return parser

def populate_small_dataset(args):
    data = pd.read_csv(args.csv_path)
    data = data.to_dict(orient="records")
    for dt in data:
        shutil.copy(args.data_path + dt['imageName'], args.output_path)

if __name__ == "__main__":
    parser = prepare_parser()
    args = parser.parse_args()
    populate_small_dataset(args)

