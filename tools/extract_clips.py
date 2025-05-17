import argparse
import time
from parsers import DatasetAParser

parser = argparse.ArgumentParser(
    description="Extract action clips from the raw video dataset"
)
parser.add_argument("--data-path", type=str, default="data/raw", help="원본 데이터 경로")
parser.add_argument("--output-path", type=str, default="data/clips", help="출력 클립 저장 경로")
parser.add_argument("--dataset", type=str, default="dataset_a", choices=["dataset_a"], help="처리할 데이터셋 종류")
parser.add_argument("--workers", type=int, default=None, help="동시 worker 수")

args = parser.parse_args()


def main():
    print(f"데이터셋 '{args.dataset}' 클립 추출 시작...")
    start_time = time.time()
    
    # Select the appropriate parser based on the dataset name
    if args.dataset == "dataset_a":
        parser = DatasetAParser(args.data_path, max_workers=args.workers)
    else:
        raise ValueError(f"지원되지 않는 데이터셋: {args.dataset}")
    
    # Run the parsing pipeline which extracts clips
    parser.parse()
    
    # 완료 시간 및 소요 시간 출력
    elapsed_time = time.time() - start_time
    minutes, seconds = divmod(elapsed_time, 60)
    hours, minutes = divmod(minutes, 60)
    
    print(f"\n클립 추출 완료!")
    print(f"소요 시간: {int(hours)}시간 {int(minutes)}분 {seconds:.2f}초")


if __name__ == "__main__":
    main()
