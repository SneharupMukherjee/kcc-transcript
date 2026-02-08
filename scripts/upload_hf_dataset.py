import argparse
import os
from pathlib import Path

from huggingface_hub import HfApi


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Upload a dataset file to Hugging Face Datasets.")
    parser.add_argument(
        "--repo-id",
        default="D3m1-g0d/kcc-24-25",
        help="HF dataset repo id, e.g. username/dataset-name",
    )
    parser.add_argument(
        "--file",
        default="data/kcc_merged_2024_2025.csv",
        help="Path to dataset file",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create repo as private",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("HF_TOKEN is required in the environment.")

    file_path = Path(args.file)
    if not file_path.exists():
        raise SystemExit(f"File not found: {file_path}")

    api = HfApi(token=token)
    api.create_repo(
        repo_id=args.repo_id,
        repo_type="dataset",
        exist_ok=True,
        private=args.private,
    )

    api.upload_file(
        path_or_fileobj=str(file_path),
        path_in_repo=file_path.name,
        repo_id=args.repo_id,
        repo_type="dataset",
        commit_message=f"Upload {file_path.name}",
    )


if __name__ == "__main__":
    main()
