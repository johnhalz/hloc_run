import argparse
from datetime import datetime
from pathlib import Path

from hloc import (
    extract_features,
    match_features,
    pairs_from_exhaustive,
    reconstruction,
)
from loguru import logger
from pycolmap import Reconstruction

from hloc_run.localized_image import get_image_files


def parse_arguments():
    """Parse input arguments to app"""
    parser = argparse.ArgumentParser(prog="Run HLOC", description="Run HLOC on a folder of images")
    parser.add_argument(
        "-i",
        "--input",
        dest="input",
        type=Path,
        required=True,
        help="Path to folder containing images to process",
    )

    return parser.parse_args()


def evaluate_args():
    """Evaluate input arguments"""
    args = parse_arguments()

    assert args.input.is_dir(), f"Input path {args.input} is not a directory"
    assert args.input.exists(), f"Input path {args.input} does not exist"

    return args.input


def setup_hloc_env(input_folder: Path):
    output_folder = input_folder / f"hloc_output_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_folder.mkdir(exist_ok=True, parents=True)
    sfm_pairs = output_folder / "pairs-sfm.txt"
    loc_pairs = output_folder / "pairs-loc.txt"
    sfm_dir = output_folder / "sfm"
    features = output_folder / "features.h5"
    matches = output_folder / "matches.h5"

    feature_conf = extract_features.confs["disk"]
    matcher_conf = match_features.confs["disk+lightglue"]

    return (
        output_folder,
        sfm_pairs,
        loc_pairs,
        sfm_dir,
        features,
        matches,
        feature_conf,
        matcher_conf,
    )


def run_mapping(
    input_folder: Path, feature_conf, matcher_conf, sfm_pairs, features, matches, sfm_dir
) -> Reconstruction:
    # Getting image files from the input folder
    references = get_image_files(input_folder)
    references = [p.relative_to(input_folder).as_posix() for p in references]
    logger.info(f"Found {len(references)} images in {input_folder}")

    # Extracting features and matching across image pairs
    extract_features.main(feature_conf, input_folder, image_list=references, feature_path=features)
    pairs_from_exhaustive.main(sfm_pairs, image_list=references)
    match_features.main(matcher_conf, sfm_pairs, features=features, matches=matches)

    # Run incremental SfM
    model = reconstruction.main(
        sfm_dir, input_folder, sfm_pairs, features, matches, image_list=references
    )

    return model


def main(input_folder: Path):
    output_folder, sfm_pairs, _, sfm_dir, features, matches, feature_conf, matcher_conf = (
        setup_hloc_env(input_folder)
    )

    model = run_mapping(
        input_folder,
        feature_conf,
        matcher_conf,
        sfm_pairs,
        features,
        matches,
        sfm_dir,
    )

    # Save the model to a file
    model.write_binary(output_folder)

    logger.success(f"Mapping completed. Model saved in {output_folder}")


if __name__ == "__main__":
    input_path = evaluate_args()
    main(input_path)
