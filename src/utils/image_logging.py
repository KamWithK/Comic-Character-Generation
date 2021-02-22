import torch
import wandb

from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities.cloud_io import load as pl_load
from torch.utils.data import Subset, DataLoader

class ImageLoggingCallback(Callback):
    def __init__(self, num_log=10):
        super().__init__()
        self.num_log = num_log

    def on_epoch_end(self, trainer, pl_module):
        noise = torch.randn(self.num_log, *trainer.datamodule.full_dataset[0][0].shape).to(pl_module.device)

        pl_module.logger.log_metrics({"Generations": wandb.Image(pl_module(noise))})
    
    def on_train_end(self, trainer, pl_module):
        checkpoint = pl_load(trainer.checkpoint_callback.best_model_path)
        pl_module.load_state_dict(checkpoint["state_dict"])

        noise = torch.randn(self.num_log, *trainer.datamodule.full_dataset[0][0].shape).to(pl_module.device)
        
        pl_module.logger.log_metrics({"Final Generations": wandb.Image(pl_module(noise))})
