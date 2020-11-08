import click

from all_2 import sobel_filter
from all_2 import sobel_filter_ddepth
from all_2 import scharr_filter
from all_2 import scharr_filter_threshold
from all_2 import prewitt_filter
from all_2 import roberts_filter
from all_2 import canny_filter
from all_2 import canny_filter_animate
from all_2 import laplacian_filter


@click.group()
def entry_point():
    pass


entry_point.add_command(sobel_filter)
entry_point.add_command(sobel_filter_ddepth)
entry_point.add_command(scharr_filter)
entry_point.add_command(scharr_filter_threshold)
entry_point.add_command(prewitt_filter)
entry_point.add_command(roberts_filter)
entry_point.add_command(canny_filter)
entry_point.add_command(canny_filter_animate)
entry_point.add_command(laplacian_filter)

if __name__ == "__main__":
    entry_point()
