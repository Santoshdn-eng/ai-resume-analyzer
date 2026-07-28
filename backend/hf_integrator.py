"""
hf_integrator.py
Hugging Face Integration Service for AI Resume Analyser.

Functions:
  1. push_to_hf()     -> Uploads your Kaggle dataset or trained model to Hugging Face Hub.
  2. load_from_hf()   -> Downloads trained dataset CSV or model weights from Hugging Face Hub into backend.
"""

import os
from huggingface_hub import HfApi, hf_hub_download

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


def push_to_huggingface(file_path: str, repo_id: str, repo_type: str = "dataset", token: str = None) -> str:
    """
    Uploads a local file (e.g. Kaggle dataset CSV or model .joblib) to Hugging Face Hub.
    
    Parameters:
      file_path : Absolute path to the file to upload (e.g. 'backend/data/AI_Resume_Screening.csv')
      repo_id   : Your HF Repo ID in format 'username/repo-name' (e.g. 'santoshdebnath/ai-resume-dataset')
      repo_type : 'dataset' or 'model'
      token     : Your Hugging Face User Access Token (read/write access)
    """
    hf_token = token or os.getenv("HF_TOKEN")
    if not hf_token:
        raise ValueError("Hugging Face API token is required. Set HF_TOKEN environment variable or pass token parameter.")

    api = HfApi(token=hf_token)

    # Create repository if it doesn't exist
    print(f"Ensuring Hugging Face {repo_type} repository '{repo_id}' exists...")
    api.create_repo(repo_id=repo_id, repo_type=repo_type, exist_ok=True)

    filename = os.path.basename(file_path)
    print(f"Uploading '{filename}' to Hugging Face ({repo_type}: {repo_id})...")

    uploaded_url = api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=filename,
        repo_id=repo_id,
        repo_type=repo_type,
    )
    print(f"Success! Uploaded to: {uploaded_url}")
    return uploaded_url


def load_from_huggingface(repo_id: str, filename: str, repo_type: str = "dataset", token: str = None) -> str:
    """
    Downloads a trained dataset or model file from Hugging Face Hub into the backend automatically.
    
    Parameters:
      repo_id   : Hugging Face Repo ID (e.g. 'santoshdebnath/ai-resume-dataset')
      filename  : Name of the file in the repo (e.g. 'AI_Resume_Screening.csv' or 'recruiter_model.joblib')
      repo_type : 'dataset' or 'model'
      token     : Hugging Face token (optional for public repositories)
    """
    hf_token = token or os.getenv("HF_TOKEN")
    target_dir = DATA_DIR if repo_type == "dataset" else MODEL_DIR

    print(f"Downloading '{filename}' from Hugging Face ({repo_type}: {repo_id})...")
    local_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        repo_type=repo_type,
        local_dir=target_dir,
        token=hf_token
    )
    print(f"Downloaded successfully to: {local_path}")
    return local_path


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Hugging Face Push & Pull CLI Utility")
    parser.add_argument("--action", choices=["push", "pull"], required=True, help="Action to perform: push or pull")
    parser.add_argument("--repo_id", required=True, help="Hugging Face repo ID e.g. username/resume-dataset")
    parser.add_argument("--file_path", help="Local file path for pushing (e.g. backend/data/AI_Resume_Screening.csv)")
    parser.add_argument("--filename", help="Filename in HF repo for pulling (e.g. AI_Resume_Screening.csv)")
    parser.add_argument("--repo_type", choices=["dataset", "model"], default="dataset")
    parser.add_argument("--token", help="Hugging Face token")

    args = parser.parse_args()

    if args.action == "push":
        if not args.file_path:
            raise ValueError("--file_path is required for push action")
        push_to_huggingface(args.file_path, args.repo_id, args.repo_type, args.token)
    elif args.action == "pull":
        if not args.filename:
            raise ValueError("--filename is required for pull action")
        load_from_huggingface(args.repo_id, args.filename, args.repo_type, args.token)
