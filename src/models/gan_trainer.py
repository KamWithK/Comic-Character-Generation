import torch

import torch.nn as nn

class GANTrainer():
    def __init__(self, generator, discriminator, generator_optimiser, discriminator_optimiser, relavistic="average"):
        self.generator, self.discriminator = generator, discriminator
        self.generator_optimiser, self.discriminator_optimiser = generator_optimiser, discriminator_optimiser

        self.criterion = nn.BCEWithLogitsLoss()

        self.relavistic = relavistic

    def train_generator(self, noise, imgs, real_label, fake_label):
        self.generator.train()
        self.generator_optimiser.zero_grad()

        real_preds, fake_preds = self.discriminator(imgs).view(-1), self.discriminator(self.generator(noise)).view(-1)

        if self.relavistic == False:
            generator_loss = self.criterion(fake_preds, real_label)
        elif self.relavistic == True:
            generator_loss = self.criterion(fake_preds - real_preds, real_label)
        elif self.relavistic == "average":
            generator_loss = self.criterion(real_preds - torch.mean(fake_preds), fake_label) + self.criterion(fake_preds - torch.mean(real_preds), real_label)

        generator_loss.backward()
        self.generator_optimiser.step()

        return generator_loss

    def train_discriminator(self, noise, imgs, real_label, fake_label):
        self.discriminator.train()
        self.discriminator_optimiser.zero_grad()

        real_preds, fake_preds = self.discriminator(imgs).view(-1), self.discriminator(self.generator(noise).detach()).view(-1)

        if self.relavistic == False:
            real_loss = self.criterion(real_preds, real_label)
            fake_loss = self.criterion(fake_preds, fake_label)
        elif self.relavistic == True:
            # Real/fake duplicates for concistancy with official recommended loss function and other loss functions here
            real_loss = self.criterion(real_preds - fake_preds, real_label)
            fake_loss = self.criterion(real_preds - fake_preds, real_label)
        elif self.relavistic == "average":
            real_loss = self.criterion(real_preds - torch.mean(fake_preds), real_label)
            fake_loss = self.criterion(fake_preds - torch.mean(real_preds), fake_label)

        (real_loss + fake_loss).backward()
        self.discriminator_optimiser.step()

        return real_loss, fake_loss