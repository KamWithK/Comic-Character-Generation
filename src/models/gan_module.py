import torch

from pytorch_lightning import LightningModule
from torch import optim
from torch.optim import Adam
from torch.nn import MSELoss, BCEWithLogitsLoss
from utils.feature_loss import FeatureLoss
from torch.optim.lr_scheduler import ExponentialLR

class GANModule(LightningModule):
    def __init__(self, generator=None, discriminator=None, discriminator_criterion=BCEWithLogitsLoss(), hparams={"pixel_scale": 100, "feature_scale": 100}):
        super().__init__()

        hparams = dict(hparams)
        self.generator, self.discriminator, = generator, discriminator

        self.feature_loss = FeatureLoss()
        self.pixel_loss = MSELoss()
        self.discriminator_criterion = discriminator_criterion

        hparams.update({
            "generator_criterion": FeatureLoss().__class__.__name__,
            "discriminator_criterion": discriminator_criterion.__class__.__name__
        })

        self.hparams = hparams

    @property
    def automatic_optimization(self) -> bool:
        return False

    def forward(self, input):
        return self.generator(input)

    def configure_optimizers(self):
        discriminator_optimizer = Adam(self.discriminator.parameters(), lr=0.0003, betas=(0.9, 0.999))
        
        generator_optimizer = Adam(self.generator.parameters(), lr=0.0002, betas=(0, 0.5))
        generator_scheduler, discriminator_scheduler = ExponentialLR(generator_optimizer, 0.9), ExponentialLR(discriminator_optimizer, 0.94)

        return [generator_optimizer, discriminator_optimizer], [generator_scheduler, discriminator_scheduler]
            
    def training_step(self, batch, batch_idx, optimizer_idx):
        # Ensure that every batch is run through the network once per epoch
        if optimizer_idx != 0: return

        generator_optimizer, discriminator_optimizer = self.optimizers()

        # Get input images and generator output
        imgs, _ = batch
        noise = torch.randn_like(imgs, device=self.device)

        real = torch.FloatTensor([[1, 0] for _ in range(imgs.size(0))]).to(imgs.device)
        fake = torch.FloatTensor([[0, 1] for _ in range(imgs.size(0))]).to(imgs.device)

        gen_outputs = self.forward(noise)

        # Account for accumulated gradients
        accumulated_grad_batches = batch_idx % self.trainer.accumulate_grad_batches == 0
        
        # Train generator
        def generator_closure():
            adversarial_loss = self.discriminator_criterion(self.discriminator(gen_outputs), real)
            pixel_loss = self.pixel_loss(gen_outputs, imgs)
            feature_loss = self.feature_loss(gen_outputs, imgs)

            self.generator_loss = adversarial_loss + self.hparams["feature_scale"] * feature_loss

            # Return the total and individual losses
            self.log_dict({
                "generator_loss": self.generator_loss,
                "adversarial_loss": adversarial_loss,
                "pixel_loss": pixel_loss,
                "feature_loss": feature_loss,
                "scaled_pixel_loss": self.hparams["pixel_scale"] * pixel_loss,
                "scaled_feature_loss": self.hparams["feature_scale"] * feature_loss
            }, on_step=False, on_epoch=True, prog_bar=True, logger=True)

            self.manual_backward(self.generator_loss, generator_optimizer)

            if accumulated_grad_batches:
                generator_optimizer.zero_grad()
        
        # Train discriminator
        def discriminator_closure():
            real_loss = self.discriminator_criterion(self.discriminator(imgs), real)
            fake_loss = self.discriminator_criterion(self.discriminator(gen_outputs.detach()), fake)

            self.discriminator_loss = real_loss + fake_loss

            # Return the total and individual losses
            self.log_dict({
                "discriminator_loss": self.discriminator_loss,
                "real_loss": real_loss,
                "fake_loss": fake_loss
            }, on_step=False, on_epoch=True, prog_bar=True, logger=True)

            self.manual_backward(self.discriminator_loss, discriminator_optimizer)

            if accumulated_grad_batches:
                discriminator_optimizer.zero_grad()

        # Train generator and then discriminator
        with generator_optimizer.toggle_model(sync_grad=accumulated_grad_batches):
            generator_optimizer.step(closure=generator_closure)
        with discriminator_optimizer.toggle_model(sync_grad=accumulated_grad_batches):
            discriminator_optimizer.step(closure=discriminator_closure)

        return self.generator_loss

    def validation_step(self, batch, batch_idx):
        # Get input images and generator output
        imgs, _ = batch
        noise = torch.randn_like(imgs, device=self.device)
        gen_outputs = self.forward(noise)
        pixel_loss = self.feature_loss(gen_outputs, imgs)
        
        return pixel_loss
