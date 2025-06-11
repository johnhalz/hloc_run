import argparse
from pathlib import Path

from hloc.utils import viz_3d
from pycolmap import Reconstruction


def parse_arguments():
    """Parse input arguments to app"""
    parser = argparse.ArgumentParser(
        prog="View HLOC Results",
        description="Script to view HLOC results from a pickled model file",
    )
    parser.add_argument(
        "-i", "--input", dest="input", type=Path, required=True, help="Path to model binaries"
    )

    return parser.parse_args()


def evaluate_args():
    """Evaluate input arguments"""
    args = parse_arguments()

    assert args.input.exists(), f"Input path {args.input} does not exist"

    return args.input


def main(input_folder: Path):
    loaded_model = Reconstruction(path=input_folder)

    assert isinstance(loaded_model, Reconstruction), (
        "Loaded model is not a valid Reconstruction object"
    )

    fig = viz_3d.init_figure()
    viz_3d.plot_reconstruction(
        fig, loaded_model, color="rgba(255,0,0,0.5)", name="mapping", points_rgb=True
    )
    fig.show()


if __name__ == "__main__":
    input_folder = evaluate_args()
    main(input_folder)
