import timeit
from pathlib import Path

from dynaconf import settings

from jinja2 import Environment
from jinja2.loaders import FileSystemLoader

env = Environment(loader=FileSystemLoader('templates'))


def write_to_file(tmp, file_name, **kwargs):
    """
    Writes the top k image comparisions to a html file using Jinja templates.

    Args:
        tmp: The template file
        file_name: The filename of the output
        **kwargs: The data to render
    """
    tmpl = env.get_template(tmp)
    op_path = Path(settings.OUTPUT_PATH) / file_name
    s = timeit.default_timer()

    f = open(op_path, "w")
    f.write(tmpl.render(**kwargs))

    e = timeit.default_timer()
    print("Took {} to write".format(e - s))
