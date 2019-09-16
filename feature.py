from feature import moment, sift
from dynaconf import settings
from pathlib import Path

import argparse


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

    model = None
    if args.model == "moment":
        model = moment.Moment(settings.WINDOW.WIN_HEIGHT, settings.WINDOW.WIN_WIDTH)
    elif args.model == "sift":
        model = sift.Sift(bool(settings.SIFT.USE_OPENCV))

    if args.visualize:
        model.visualize(img_path, Path(settings.OUTPUT_PATH))
    else:
        model.process_img(img_path)
