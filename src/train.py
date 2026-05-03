from datetime import datetime
import os

import pytorch_lightning as L
from pytorch_lightning.loggers import WandbLogger

from torch.utils.data import DataLoader

from datasets import load_dataset

from src.data_module.dataset import ImgDataset
from src.nets.DiT import DIT

from src.hf_save import HubCheckpointSync
from.resume import get_latest_checkpoint

batch_size = 128
embed_dims = 384
head_size = 64
num_heads = 6
block_num = 16

patch_size = 2
out_channels = 4
in_dims = 4

latent_h = 64
latent_w = 64

lr = 1e-4
iterations = 25000
acc_grad = 1

log_steps = 50
run_name = "anime_captions_v1"

HF_TOKEN = os.environ["HF_TOKEN"]   # never hardcode tokens
HF_REPO  = "Jumpr/anime-dit-checkpoints"
     

def main():
     ckpt_path = get_latest_checkpoint(HF_REPO, local_dir="src/model_ckpts")
     
     ds = load_dataset(
          "none-yet/anime-captions",
          split="train"
     )

     img_ds = ImgDataset(ds) #. hw patches hardcoded
     train_dataloader = DataLoader(
          img_ds, batch_size=batch_size, 
          shuffle=True, 
          num_workers=20,
          drop_last=True, 
          pin_memory=True,
          ) 
     
     model = DIT(
          batch_size,
          patch_size,
          out_channels,
          in_dims,
          embed_dims,
          head_size,
          num_heads,
          block_num,
          lr,
          iterations,
          latent_h,
          latent_w,
          "REPA-E/e2e-sdvae-hf",
          0.18215
     )

     run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
     
     wandb_logger = WandbLogger(
          log_model=False,
          resume="allow",
          name=run_name,
          id = run_id
    )

     trainer = L.Trainer(
          logger=wandb_logger,
          max_epochs=1,
          limit_train_batches=iterations,
          precision="bf16-mixed",
          gradient_clip_val=1.0,
          accumulate_grad_batches=acc_grad,
          log_every_n_steps=log_steps,
          enable_checkpointing=True,
          devices=1,
          strategy="auto",
          callbacks=[
               L.callbacks.ModelCheckpoint(
                    dirpath='.src/model_ckpts', every_n_train_steps=10, save_top_k=-1
               ),
               HubCheckpointSync(repo_id=HF_REPO, token=HF_TOKEN)
          ],
     )
     
     trainer.fit(
          model, 
          train_dataloaders=train_dataloader,
          ckpt_path=ckpt_path
     ) 

if __name__ == "__main__":
     main()