import os
from pytorch_lightning import Callback
from huggingface_hub import HfApi

class HubCheckpointSync(Callback):
    def __init__(self, repo_id: str, token: str):
        self.repo_id = repo_id
        self.api = HfApi(token=token)

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        # Lightning calls this right after saving a .ckpt file
        ckpt_path = trainer.checkpoint_callback.last_model_path
        if not ckpt_path or not os.path.exists(ckpt_path):
            return
        
        step = trainer.global_step
        print(f"Syncing checkpoint step {step} to Hub...")
        
        self.api.upload_file(
            path_or_fileobj=ckpt_path,
            path_in_repo=f"checkpoints/step_{step}.ckpt",
            repo_id=self.repo_id,
            repo_type="model",
        )
        print("synced")