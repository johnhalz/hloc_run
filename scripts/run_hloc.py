import argparse
from datetime import datetime
from pathlib import Path

from hloc import extract_features, match_features, pairs_from_exhaustive, reconstruction
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


def trigger_hloc(image_files: list[Path], input_folder: Path) -> Reconstruction:
    feature_conf = extract_features.confs["disk"]
    matcher_conf = match_features.confs["disk+lightglue"]

    # Output paths
    output_folder = input_folder / f"hloc_output_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_folder.mkdir(parents=True, exist_ok=True)

    sfm_pairs = output_folder / "pairs-sfm.txt"
    sfm_dir = output_folder / "sfm"
    features = output_folder / "features.h5"
    matches = output_folder / "matches.h5"

    # Extract and match features
    logger.info(f"Extracting and matching features from {len(image_files)} images")
    if not features.exists():
        extract_features.main(
            conf=feature_conf,
            image_dir=input_folder,
            export_dir=output_folder,
            image_list=image_files,
            feature_path=features,
        )
    else:
        logger.info(f"Features already exist at {features.as_posix()}")

    logger.info(f"Creating pairs from {len(image_files)} images")
    if not sfm_pairs.exists():
        pairs_from_exhaustive.main(sfm_pairs, image_list=image_files)
    else:
        logger.info(f"Pairs already exist at {sfm_pairs.as_posix()}")

    if not matches.exists():
        match_features.main(
            conf=matcher_conf,
            pairs=sfm_pairs,
            features=features,
            matches=matches,
        )
    else:
        logger.info(f"Matches already exist at {matches.as_posix()}")

    logger.success("Extracted and matched features")

    # Run SFM
    logger.info("Running SFM")
    model = reconstruction.main(
        sfm_dir=sfm_dir,
        image_dir=input_folder,
        pairs=sfm_pairs,
        features=features,
        matches=matches,
        image_list=image_files,
    )
    logger.success("SFM completed")

    return model


def main(input_folder: Path):
    image_files = get_image_files(input_folder)
    logger.info(f"Found {len(image_files)} images in {input_folder}")

    hloc_model = trigger_hloc(image_files, input_folder)
    pass


if __name__ == "__main__":
    input_path = evaluate_args()
    main(input_path)
