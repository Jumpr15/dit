import lightning as L
from lightning.pytorch.loggers import WandbLogger

from torch.utils.data import DataLoader

from datasets import load_dataset

from data_module.dataset import DigitsDataset
from nets.DiT import DIT

def main():
     batch_size = 8
     embed_dims = 64
     head_size = 16
     num_heads = 4
     block_num = 8
     
     patch_size = 1
     out_channels = 4
     in_dims = 4

     lr = 2e-4
     iterations = 10000
     acc_grad = 4
     
     
     ds = load_dataset(
          "nguyenminh4099/handwritten-digits",
          split="train"
     )

     digits_ds = DigitsDataset(ds) #. hw patches hardcoded
     train_dataloader = DataLoader(digits_ds, batch_size=batch_size, shuffle=True)
     
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
          iterations
     )
     
     wandb_logger = WandbLogger(
          log_model=False,
          resume="allow",
          id="digitGeneratorTraining"
    )

     trainer = L.Trainer(
          logger=wandb_logger,
          max_epochs=1,
          limit_train_batches=iterations,
          precision="bf16-mixed",
          gradient_clip_val=1.0,
          accumulate_grad_batches=acc_grad,
          log_every_n_steps=1000,
          enable_checkpointing=True,
          devices=1,
          strategy="auto",
          callbacks=[
               L.pytorch.callbacks.ModelCheckpoint(
                    dirpath='/model_ckpts', every_n_train_steps=500, save_top_k=-1
               )
          ],
     )
     
     trainer.fit(model, train_dataloaders=train_dataloader)

if __name__ == "__main__":
     main()