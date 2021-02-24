import torch

from pytorch_lightning import LightningModule
from torch.optim import Adam
from torch.nn import MSELoss, BCEWithLogitsLoss
from torch.optim.lr_scheduler import ExponentialLR
from utils.feature_loss import FeatureLoss

class NoGANModule(LightningModule):
    def __init__(self, generator=None, discriminator=None, discriminator_criterion=BCEWithLogitsLoss(), hparams={"loss_threshold": 0.2, "pretrain_generator": 0, "pretrain_discriminator": 0}):
        super().__init__()

        hparams = dict(hparams)
        self.generator, self.discriminator, = generator, discriminator

        self.feature_loss = FeatureLoss()
        self.pixel_loss = MSELoss()
        self.discriminator_criterion = discriminator_criterion

        # Avoid crash when starting by training the discriminator
        self.generator_loss = None

        # Presetting previous batch to generator, forces first batch to train discriminator
        # Only when pretraining isn't used
        self.pretrain_generator, self.pretrain_discriminator = hparams["pretrain_generator"], hparams["pretrain_discriminator"]

        hparams.update({
            "generator_criterion": FeatureLoss.__class__.__name__,
            "discriminator_criterion": discriminator_criterion.__class__.__name__
        })

        self.hparams = hparams

    @property
    def automatic_optimization(self) -> bool:
        return False

    def forward(self, input):
        return self.generator(input)

    def configure_optimizers(self):
        generator_optimizer, discriminator_optimizer = Adam(self.generator.parameters()), Adam(self.discriminator.parameters())

        return [generator_optimizer, discriminator_optimizer]

    # Switch between training generator and discriminator after pretraining
    def on_train_batch_start(self, batch, batch_idx, dataloader_idx):
        if self.pretrain == False:
            # Update discriminator loss if training the generator
            if self.train_stage == "generator" or "discriminator_loss" not in self.__dict__:
                self.discriminator_loss = self.hparams["loss_threshold"]
            
            # Switch modes to generator if discriminator is trained enough
            self.train_stage = "generator" if self.discriminator_loss < self.hparams["loss_threshold"] else "discriminator"

    # Handle pretraining
    def on_epoch_start(self) -> None:
        self.pretrain = True if self.pretrain_generator > 0 or self.pretrain_discriminator > 0 else False

        if self.pretrain_generator > 0:
            self.pretrain_generator -= 1
            self.train_stage = "generator"
        elif self.pretrain_discriminator > 0:
            self.pretrain_discriminator -= 1
            self.train_stage = "discriminator"
            
    def training_step(self, batch, batch_idx, optimizer_idx):
        # Get input images and generator output
        imgs, _ = batch
        noise = torch.randn_like(imgs, device=self.device)

        real = torch.FloatTensor([[1, 0] for _ in range(imgs.size(0))]).to(imgs.device)
        fake = torch.FloatTensor([[0, 1] for _ in range(imgs.size(0))]).to(imgs.device)

        gen_outputs = self.forward(noise)

        # Account for accumulated gradients
        accumulated_grad_batches = batch_idx % self.trainer.accumulate_grad_batches == 0

        # Use the right optimizer
        optimizer = self.optimizers()[0] if self.train_stage == "generator" else self.optimizers()[1]

        # Train generator
        def generator_closure():
            losses = {
                "generator_loss": self.feature_loss(gen_outputs, imgs),
                "pixel_loss": self.pixel_loss(gen_outputs, imgs),
                "feature_loss": self.feature_loss(gen_outputs, imgs)
            }

            if self.pretrain_generator == 0:
                losses["adversarial_loss"] = self.discriminator_criterion(self.discriminator(gen_outputs), real)
                losses["generator_loss"] += losses["adversarial_loss"]

            self.generator_loss = losses["generator_loss"]

            # Return the total and individual losses
            self.log_dict(losses, prog_bar=True, logger=True)

            self.manual_backward(self.generator_loss, optimizer)

            if accumulated_grad_batches:
                optimizer.zero_grad()
        
        # Train discriminator
        def discriminator_closure():
            losses = {
                "real_loss": self.discriminator_criterion(self.discriminator(imgs), real),
                "fake_loss": self.discriminator_criterion(self.discriminator(gen_outputs.detach()), fake)
            }

            losses["discriminator_loss"] = losses["real_loss"] + losses["fake_loss"]
            self.discriminator_loss = losses["discriminator_loss"]

            # Return the total and individual losses
            self.log_dict(losses, prog_bar=True, logger=True)

            self.manual_backward(self.discriminator_loss, optimizer)

            if accumulated_grad_batches:
                optimizer.zero_grad()

        with optimizer.toggle_model(sync_grad=accumulated_grad_batches):
            optimizer.step(closure=generator_closure) if self.train_stage == "generator" else optimizer.step(closure=discriminator_closure)

        return self.generator_loss

    def validation_step(self, batch, batch_idx):
        # Get input images and generator output
        imgs, _ = batch
        noise = torch.randn_like(imgs, device=self.device)
        gen_outputs = self.forward(noise)
        
        return self.feature_loss(gen_outputs, imgs)
