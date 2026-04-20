import argparse
import os

# ===== 不同数据集的预处理 =====
from src.data.ml_preprocess import preprocess_and_save as ml_preprocess
from src.data.preprocess import preprocess_and_save as base_preprocess
from src.data.ubf_preprocess import preprocess_and_save as ubf_preprocess


def main(args):
    print(" 开始数据预处理...")
    print(f" 数据集: {args.dataset}")

    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../datasets"))
    dataset_dir = os.path.join(base_dir, args.dataset)

    os.makedirs(dataset_dir, exist_ok=True)

    # ===== 数据集路径定义 =====
    if args.dataset == "multi-category":
        input_path = os.path.join(dataset_dir, "mul-cate.txt")

        ml_preprocess(
            input_path=input_path,
            output_dir=dataset_dir
        )

    elif args.dataset == "2019-oct":
        input_path = os.path.join(dataset_dir, "processed_data.txt")

        base_preprocess(
            input_path=input_path,
            output_dir=dataset_dir
        )

    elif args.dataset == "ubf":
        input_path = os.path.join(dataset_dir, "ubf_process.csv")

        ubf_preprocess(
            input_path=input_path,
            output_dir=dataset_dir,
            hash_ratio=1.5,
            min_item_freq=4,
            days_limit=1500,
            sample_ratio=0.1,
            random_state=42
        )

    else:
        raise ValueError(f" 未支持的数据集: {args.dataset}")

    print(" 数据预处理完成！")
    print(f" 输出路径: {dataset_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        type=str,
        default="2019-oct",
        choices=["multi-category", "2019-oct", "ubf"]
    )

    args = parser.parse_args()

    main(args)
