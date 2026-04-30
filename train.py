import lightning as L
from lightning.pytorch.loggers import WandbLogger

from nets.Dit import DIT

def main():
     wandb_logger = WandbLogger(
          log_model=False,
          resume="allow",
          id="" ###
    )

     trainer = L.Trainer(
          logger=wandb_logger
     )

if __name__ == "__main__":
     main()