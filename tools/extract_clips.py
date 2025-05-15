import argparse
from parsers import DatasetAParser

parser = argparse.ArgumentParser(
    description="Extract action clips from the raw video dataset"
)
parser.add_argument("--data-path", type=str, default="data/raw")
parser.add_argument("--output-path", type=str, default="data/clips")
parser.add_argument("--dataset", type=str, default="dataset_a", choices=["dataset_a"])

args = parser.parse_args()


def main():
    # Select the appropriate parser based on the dataset name
    if args.dataset == "dataset_a":
        parser = DatasetAParser(args.data_path)
    else:
        raise ValueError(f"지원되지 않는 데이터셋: {args.dataset}")
    
    # Run the parsing pipeline which extracts clips
    parser.parse()


if __name__ == "__main__":
    main()
