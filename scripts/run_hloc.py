import argparse
from pathlib import Path

def parse_arguments():
    '''Parse input arguments to app'''
    parser = argparse.ArgumentParser(
        prog='Run HLOC',
        description='Run HLOC on a folder of images'
    )
    parser.add_argument('-i', '--input',
                        dest='input',
                        type=Path,
                        required=True,
                        help='Path to folder containing images to process')

    return parser.parse_args()

def evaluate_args():
    '''Evaluate input arguments'''
    args = parse_arguments()

    assert args.input.is_dir(), f"Input path {args.input} is not a directory"
    assert args.input.exists(), f"Input path {args.input} does not exist"

    return args.input, args.results_folder_regex