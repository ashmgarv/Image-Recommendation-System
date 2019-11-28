from pathlib import Path
import csv
from dynaconf import settings

from jinja2 import Environment
from jinja2.loaders import FileSystemLoader

env = Environment(loader=FileSystemLoader(settings.path_for(settings.TEMPLATE_PATH)))


def write_to_file(tmp, file_name, **kwargs):
    """
    Writes the top k image comparisions to a html file using Jinja templates.

    Args:
        tmp: The template file
        file_name: The filename of the output
        **kwargs: The data to render
    """
    tmpl = env.get_template(tmp)
    op_path = Path(settings.path_for(settings.OUTPUT_PATH)) / file_name

    f = open(op_path, "w")
    f.write(tmpl.render(**kwargs))
    print("Fin. Check {} in Outputs folder".format(file_name))

def print_term_weight_pairs(term_weight_pairs, file_name):
    file_name = Path(settings.path_for(settings.OUTPUT_PATH)) / file_name
    with open(file_name, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(term_weight_pairs)

