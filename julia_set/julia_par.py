#! /usr/bin/env python

from re import U
import numpy as np
import argparse
import time
from multiprocessing import Pool, TimeoutError
from julia_curve import c_from_group

# Update according to your group size and number (see TUWEL)
GROUP_SIZE = 3
GROUP_NUMBER = 2

# do not modify BENCHMARK_C
BENCHMARK_C = complex(-0.2, -0.65)


def compute_julia_set_sequential(xmin, xmax, ymin, ymax, im_width, im_height, c):

    zabs_max = 10
    nit_max = 300

    xwidth = xmax - xmin
    yheight = ymax - ymin

    julia = np.zeros((im_width, im_height))
    for ix in range(im_width):
        for iy in range(im_height):
            nit = 0
            # Map pixel position to a point in the complex plane
            z = complex(ix / im_width * xwidth + xmin,
                        iy / im_height * yheight + ymin)
            # Do the iterations
            while abs(z) <= zabs_max and nit < nit_max:
                z = z**2 + c
                nit += 1
            ratio = nit / nit_max
            julia[ix, iy] = ratio

    return julia


def compute_single_patch(args):
    x_start, y_start, patch_width, patch_height, patch_xmin, patch_xmax, patch_ymin, patch_ymax, c = args
    patch = compute_julia_set_sequential(
        patch_xmin, patch_xmax, patch_ymin, patch_ymax,
        patch_width, patch_height, c
    )
    return (x_start, y_start, patch)


def compute_julia_in_parallel(size, x_min, x_max, y_min, y_max, patch, n_procs, c):
    julia_img = np.zeros((size, size))
    tasks = []
    for x_start in range(0, size, patch):
        for y_start in range(0, size, patch):
            # in case of size % pach != 0
            x_end = min(x_start + patch, size)
            y_end = min(y_start + patch, size)

            patch_width = x_end - x_start
            patch_height = y_end - y_start

            x_width = x_max - x_min
            y_height = y_max - y_min

            # rectangle in complex plain
            patch_x_min = x_min + (x_start / size) * x_width
            patch_x_max = x_min + (x_end / size) * x_width
            patch_y_min = y_min + (y_start / size) * y_height
            patch_y_max = y_min + (y_end / size) * y_height

            tasks.append((x_start, y_start, patch_width, patch_height,
                          patch_x_min, patch_x_max, patch_y_min, patch_y_max, c))

    with Pool(processes=n_procs) as pool:
        results = pool.map(compute_single_patch, tasks)

        for x_start, y_start, patch_data in results:
            patch_height, patch_width = patch_data.shape
            julia_img[x_start:x_start+patch_height,
                      y_start:y_start+patch_width] = patch_data

    return julia_img


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--size", help="image size in pixels (square images)", type=int, default=500)
    parser.add_argument("--xmin", help="", type=float, default=-1.5)
    parser.add_argument("--xmax", help="", type=float, default=1.5)
    parser.add_argument("--ymin", help="", type=float, default=-1.5)
    parser.add_argument("--ymax", help="", type=float, default=1.5)
    parser.add_argument("--group-size", help="", type=int, default=3)
    parser.add_argument("--group-number", help="", type=int, default=2)
    parser.add_argument(
        "--patch", help="patch size in pixels (square images)", type=int, default=20)
    parser.add_argument(
        "--nprocs", help="number of workers", type=int, default=1)
    parser.add_argument(
        "--draw-axes", help="Whether to draw axes", action="store_true")
    parser.add_argument("-o", help="output file")
    parser.add_argument(
        "--benchmark", help="Whether to execute the script with the benchmark Julia set", action="store_true")
    args = parser.parse_args()

    # print(args)
    if args.group_size is not None:
        GROUP_SIZE = args.group_size
    if args.group_number is not None:
        GROUP_NUMBER = args.group_number

    # assign c based on mode
    c = None
    if args.benchmark:
        c = BENCHMARK_C
    else:
        c = c_from_group(GROUP_SIZE, GROUP_NUMBER)

    stime = time.perf_counter()
    julia_img = compute_julia_in_parallel(
        args.size,
        args.xmin, args.xmax,
        args.ymin, args.ymax,
        args.patch,
        args.nprocs,
        c)
    rtime = time.perf_counter() - stime

    print(f"{args.size};{args.patch};{args.nprocs};{rtime}")

    if not args.o is None:
        import matplotlib
        matplotlib.use('agg')
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
        fig, ax = plt.subplots()
        ax.imshow(julia_img, interpolation='nearest', cmap=plt.get_cmap("hot"))

        if args.draw_axes:
            # set labels correctly
            im_width = args.size
            im_height = args.size
            xmin = args.xmin
            xmax = args.xmax
            xwidth = args.xmax - args.xmin
            ymin = args.ymin
            ymax = args.ymax
            yheight = args.ymax - args.ymin

            xtick_labels = np.linspace(xmin, xmax, 7)
            ax.set_xticks([(x-xmin) / xwidth * im_width for x in xtick_labels])
            ax.set_xticklabels(['{:.1f}'.format(xtick)
                               for xtick in xtick_labels])
            ytick_labels = np.linspace(ymin, ymax, 7)
            ax.set_yticks(
                [(y-ymin) / yheight * im_height for y in ytick_labels])
            ax.set_yticklabels(['{:.1f}'.format(-ytick)
                               for ytick in ytick_labels])
            ax.set_xlabel("Imag")
            ax.set_ylabel("Real")
        else:
            # disable axes
            ax.axis("off")

        plt.tight_layout()
        plt.savefig(args.o, bbox_inches='tight')
        # plt.show()
