import sys
from pathlib import Path

sys.path.append('../')
from output import write_to_file
from feedback import ppr, svm
from dynaconf import settings
from utils import get_metadata
from task6_probab import feedback_probab
from classification import decision_tree

from pymongo import MongoClient

feedback_systems = {
    'ppr': ppr.ppr_feedback,
    'svm': svm.svm_feedback,
    'dt': decision_tree.decision_tree_feedback,
    'probab': feedback_probab
}


def take_feedback_system_input():
    while True:
        system = input("Select feedback system: ")
        if not system.strip():
            print("Exiting...")
            sys.exit(0)
        if system.strip() not in feedback_systems:
            print("Inavlid feeback system selected. Choices are: {}".format(
                list(feedback_systems.keys())))
            continue
        return system.strip()


def take_images_input(image_type, meta):
    image_path = Path(settings.path_for(settings.MASTER_DATA_PATH))
    while True:
        image_ids = input(
            "Enter image ids (space seperated) that you think are {}: ".format(
                image_type)).split(' ')
        invalid_input = False
        for image in image_ids:
            if not image:
                return []
            if image not in meta:
                print(
                    "Image '{}' not found. Please check the spelling and enter again."
                    .format(image))
                invalid_input = True
        if not invalid_input:
            return [str((image_path / i).resolve()) for i in image_ids]


def fetch_image_meta(paths=None):
    if paths:
        meta = get_metadata(f={'path': {'$in': paths}}, master_db=True)
    else:
        meta = get_metadata(master_db=True)

    meta = {m['imageName']: 1 for m in meta}
    return meta


def get_task5_results():
    client = MongoClient(host=settings.HOST,
                         port=settings.PORT,
                         username=settings.USERNAME,
                         password=settings.PASSWORD)
    collection = client[settings.DATABASE][settings.TASK_FIVE_OUTPUT]
    res = collection.find_one()
    if res is None:
        return None, None
    return res['query'], res['results']


def main():
    relevant_images = set()
    irrelevant_images = set()
    prev_results = None

    task5_query, prev_results = get_task5_results()
    if not task5_query or not prev_results:
        print("Please run Task 5 first. Exiting.")
        sys.exit(0)

    meta = fetch_image_meta(prev_results)
    images_to_display = len(prev_results)

    while True:
        feedback_system = take_feedback_system_input()
        relevant_images = relevant_images.union(take_images_input("relevant", meta))
        irrelevant_images = irrelevant_images.union(take_images_input("irrelevant", meta))
        if not relevant_images and not irrelevant_images:
            print(
                "No relevant images or irrelevant images provided! Doing nothing."
            )
            continue

        new_relevant_images = feedback_systems[feedback_system](
            list(relevant_images), list(irrelevant_images), images_to_display, task5_query,
            prev_results)

        write_to_file("task6.html",
                      "task6-{}.html".format(feedback_system),
                      relevant=relevant_images,
                      irrelevant=irrelevant_images,
                      result=new_relevant_images,
                      title="TEST")

        prev_results = new_relevant_images


if __name__ == '__main__':
    main()
