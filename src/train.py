from datetime import datetime

import pytorch_lightning as L
from pytorch_lightning.loggers import WandbLogger

from torch.utils.data import DataLoader

from datasets import load_dataset

from src.data_module.dataset import ImgDataset
from src.nets.DiT import DIT

def main():
     batch_size = 64
     embed_dims = 512
     head_size = 64
     num_heads = 8
     block_num = 16
     
     patch_size = 2
     out_channels = 4
     in_dims = 4
     
     latent_h = 64
     latent_w = 64

     lr = 1e-4
     iterations = 1000
     acc_grad = 1
     
     
     ds = load_dataset(
          "none-yet/anime-captions",
          split="train"
     )

     img_ds = ImgDataset(ds) #. hw patches hardcoded
     train_dataloader = DataLoader(img_ds, batch_size=batch_size, shuffle=True, num_workers=20) 
     
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
          name="artbench_train",
          id = run_id
    )

     trainer = L.Trainer(
          logger=wandb_logger,
          max_epochs=1,
          limit_train_batches=iterations,
          precision="bf16-mixed",
          gradient_clip_val=1.0,
          accumulate_grad_batches=acc_grad,
          log_every_n_steps=10,
          enable_checkpointing=True,
          devices=1,
          strategy="auto",
          callbacks=[
               L.callbacks.ModelCheckpoint(
                    dirpath='.src/model_ckpts', every_n_train_steps=50, save_top_k=-1
               )
          ],
     )
     
     trainer.fit(model, train_dataloaders=train_dataloader) # fails on last batch if batch_size != set batch_size

if __name__ == "__main__":
     main()