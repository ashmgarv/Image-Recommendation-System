from feature import moment, sift
from dynaconf import settings
from pathlib import Path

import argparse

def describe(img_path, model):
    if model == "moment":
        data = moment.process_img(img_path, settings.WINDOW.WIN_HEIGHT, settings.WINDOW.WIN_WIDTH)
    elif model == "sift":
        data = sift.process_img(img_path, bool(settings.SIFT.USE_OPENCV))

    print(data)

def visualize(img_path, model):
    if model == "moment":
        moment.visualize_yuv(img_path, Path(settings.OUTPUT_PATH))
        moment.visualize_moments(img_path, Path(settings.OUTPUT_PATH), settings.WINDOW.WIN_HEIGHT, settings.WINDOW.WIN_WIDTH)
    elif model == "sift":
        sift.visualize_sift(img_path, Path(settings.OUTPUT_PATH))

def prepare_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', type=str, required=True)
    parser.add_argument('-i', '--image-path', type=str, required=True)
    parser.add_argument('-v', '--visualize', type=bool)
    return parser

if __name__ == "__main__":
    parser = prepare_parser()
    args = parser.parse_args()

    img_path = Path(args.image_path)
    if not img_path.exists() or not img_path.is_file():
        raise Exception("Invalid Image file path.")

    if args.visualize:
        visualize(img_path, args.model)
    else:
        describe(img_path, args.model)
