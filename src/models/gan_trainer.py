import torch.nn as nn

class GANTrainer():
    def __init__(self, generator, discriminator, generator_optimiser, discriminator_optimiser):
        self.generator, self.discriminator = generator, discriminator
        self.generator_optimiser, self.discriminator_optimiser = generator_optimiser, discriminator_optimiser

        self.criterion = nn.BCELoss()

    def train_generator(self, noise, real_label):
        self.generator.train()
        self.generator_optimiser.zero_grad()

        generator_loss = self.criterion(self.discriminator(self.generator(noise)).view(-1), real_label)

        generator_loss.backward()
        self.generator_optimiser.step()

        return generator_loss

    def train_discriminator(self, noise, imgs, real_label, fake_label):
        self.discriminator.train()
        self.discriminator_optimiser.zero_grad()

        real_loss = self.criterion(self.discriminator(imgs).view(-1), real_label)
        fake_loss = self.criterion(self.discriminator(self.generator(noise).detach()).view(-1), fake_label)

        (real_loss + fake_loss).backward()
        self.discriminator_optimiser.step()

        return real_loss, fake_loss