import sys
from pathlib import Path

sys.path.append('../')
from output import write_to_file
from feedback import ppr

feedback_systems = {'ppr': ppr.ppr_feedback}


def take_feedback_system_input():
    while True:
        system = input("Select feedback system: ")
        if system.strip() not in feedback_systems:
            print("Inavlid feeback system selected. Choices are: {}".format(
                feedback_systems.keys()))
            continue
        return system.strip()


def take_images_input(image_type):
    while True:
        image_ids = input(
            "Enter image ids (space seperated) that you think are {}: ".format(
                image_type))
        return image_ids.split(' ')


def main():
    relevant_images = []
    irrelevant_images = []
    while True:
        feedback_system = take_feedback_system_input()
        relevant_images.extend(take_images_input("relevant"))
        irrelevant_images.extend(take_images_input("irrelevant"))

        new_relevant_images = feedback_systems[feedback_system](
            relevant_images, irrelevant_images)

        print(new_relevant_images)
        # Output


if __name__ == '__main__':
    main()
