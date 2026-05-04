import os
from huggingface_hub import hf_hub_download
from huggingface_hub import HfApi

def get_latest_checkpoint(repo_id: str, local_dir: str = "src/model_ckpts") -> str | None:
    """Downloads the latest checkpoint from Hub, returns local path."""
    api = HfApi()
    
    try:
        files = list(api.list_repo_files(repo_id, repo_type="model"))
    except Exception:
        return None
    
    ckpt_files = [f for f in files if f.startswith("checkpoints/") and f.endswith(".ckpt")]
    if not ckpt_files:
        return None

    # Sort by step number
    def extract_step(path):
        name = os.path.basename(path)          # "step_50000.ckpt"
        return int(name.replace("step_", "").replace(".ckpt", ""))

    latest = max(ckpt_files, key=extract_step)
    print(f"Resuming from {latest}")

    local_path = hf_hub_download(
        repo_id=repo_id,
        filename=latest,
        repo_type="model",
        local_dir=local_dir,
    )
    return local_path