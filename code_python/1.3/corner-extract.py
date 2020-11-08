import click

from all_3 import harris_detector
from all_3 import harris_detector_bsize
from all_3 import harris_detector_ksize
from all_3 import harris_detector_k
from all_3 import harris_detector_animate


@click.group()
def entry_point():
    pass


entry_point.add_command(harris_detector)
entry_point.add_command(harris_detector_bsize)
entry_point.add_command(harris_detector_ksize)
entry_point.add_command(harris_detector_k)
entry_point.add_command(harris_detector_animate)

if __name__ == "__main__":
    entry_point()
