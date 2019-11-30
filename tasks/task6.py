import sys
from pathlib import Path

sys.path.append('../')
from output import write_to_file
from feedback import ppr
from dynaconf import settings
from utils import get_metadata
from task6_probab import feedback_probab

feedback_systems = {'ppr': ppr.ppr_feedback, 'probab':feedback_probab}


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
                    "Image '{}' not found in database. Please check the spelling and enter again."
                    .format(image))
                invalid_input = True
        if not invalid_input:
            return [str((image_path / i).resolve()) for i in image_ids]


def fetch_image_meta():
    meta = get_metadata(master_db=True)
    meta = {m['imageName']: 1 for m in meta}
    return meta


def main():
    relevant_images = []
    irrelevant_images = []

    meta = fetch_image_meta()
    images_to_display = 20

    while True:
        feedback_system = take_feedback_system_input()
        relevant_images.extend(take_images_input("relevant", meta))
        irrelevant_images.extend(take_images_input("irrelevant", meta))
        print(relevant_images)
        if not relevant_images and not irrelevant_images:
            print(
                "No relevant images or irrelevant images provided! Doing nothing."
            )
            continue

        new_relevant_images = feedback_systems[feedback_system](
            relevant_images, irrelevant_images, images_to_display)

        print(new_relevant_images)
        write_to_file("task6.html",
                      "task6-{}.html".format(feedback_system),
                      relevant=relevant_images,
                      irrelevant=irrelevant_images,
                      result=new_relevant_images,
                      title="TEST")
        # Output


if __name__ == '__main__':
    main()
