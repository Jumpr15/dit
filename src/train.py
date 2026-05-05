from datetime import datetime

import pytorch_lightning as L
from pytorch_lightning.loggers import WandbLogger

from torch.utils.data import DataLoader

from datasets import load_dataset

import yaml

from data_module.dataset import ImgDataset
from nets.DiT import DIT

from resume import get_latest_checkpoint

def main():
     with open('config.yaml', 'r') as f:
          config = yaml.safe_load(f)
          
          run_name = config['run_name']
          run_id = config['run_id']
          hf_repo = config['hf_repo']
          
          ckpt_dir = config['ckpt_dir']
          
          dataset = config['dataset']
          num_workers = config['num_workers']
          pixel_h = config['pixel_h']
          pixel_w = config['pixel_w']
          
          vae = config['vae']
          vae_downsample_factor = config['vae_downsample_factor']
          vae_scale_factor = config['vae_scale_factor']
          
          latent_h = pixel_h // vae_downsample_factor
          latent_w = pixel_w // vae_downsample_factor
          
          batch_size = config['batch_size']
          iterations = config['iterations']
          log_steps = config['log_steps']
          checkpoint_steps = config['checkpoint_steps']
          
          num_devices = config['num_devices']
          
          embed_dims = config['embed_dims']
          head_size = config['head_size']
          num_heads = embed_dims // head_size
          block_num = config['block_num']
          
          patch_size = config['patch_size']
          in_dims = config['in_dims'] # no. latent space input dimensions
          out_channels = config['out_channels'] # no. latent space output dimensions
          
          acc_grad = config['acc_grad']
          lr = config['lr']
     
     ckpt_path = get_latest_checkpoint(hf_repo, local_dir=ckpt_dir)
     
     ds = load_dataset(
          dataset,
          split="train"
     )

     img_ds = ImgDataset(ds) #. hw patches hardcoded
     train_dataloader = DataLoader(
          img_ds, batch_size=batch_size, 
          shuffle=True, 
          num_workers=num_workers,
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
          vae,
          vae_scale_factor
     )

     if run_id is None:
          run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
     
     wandb_logger = WandbLogger(
          log_model=False,
          resume="allow",
          name=run_name,
          id = run_id
    )

     trainer = L.Trainer(
          logger=wandb_logger,
          max_epochs=-1,
          max_steps=iterations,
          precision="bf16-mixed",
          gradient_clip_val=1.0,
          accumulate_grad_batches=acc_grad,
          log_every_n_steps=log_steps,
          enable_checkpointing=True,
          devices=num_devices,
          strategy="auto",
          callbacks=[
               L.callbacks.ModelCheckpoint(
                    dirpath=ckpt_dir, every_n_train_steps=checkpoint_steps, save_top_k=-1
               ),
          ],
     )
     
     trainer.fit(
          model, 
          train_dataloaders=train_dataloader,
          ckpt_path=ckpt_path
     ) 

if __name__ == "__main__":
     main()