import math
import os
import random
import natsort
from PIL import Image

import torch
from torch import optim
from models import BaseVAE
from models.types_ import *
from utils import data_loader
import pytorch_lightning as pl
from torchvision import transforms
import torchvision.utils as vutils
from torchvision.datasets import CelebA
from torch.utils.data import DataLoader, Dataset


class VAEXperiment(pl.LightningModule):

    def __init__(self,
                 vae_model: BaseVAE,
                 params: dict) -> None:
        super(VAEXperiment, self).__init__()

        self.model = vae_model
        self.params = params
        self.curr_device = None
        self.hold_graph = False
        try:
            self.hold_graph = self.params['retain_first_backpass']
        except:
            pass

    def forward(self, input: Tensor, **kwargs) -> Tensor:
        return self.model(input, **kwargs)

    def training_step(self, batch, batch_idx, optimizer_idx=0):
        # real_img, labels = batch
        self.curr_device = batch.device

        results = self.forward(batch)
        # SD: changed to *SD_*loss_function
        train_loss = self.model.SD_loss_function(*results,
                                              M_N=self.params['batch_size'] / self.num_train_imgs,
                                              optimizer_idx=optimizer_idx,
                                              batch_idx=batch_idx)
        self.log("train_loss", train_loss, on_step=True, on_epoch=False, prog_bar=False, logger=True)

        #self.logger.experiment.add_scalars('train_loss', {key: val.item() for key, val in train_loss.items()}, batch_idx)

        return train_loss

    def validation_step(self, batch, batch_idx, optimizer_idx=0):
        # real_img, labels = batch
        self.curr_device = batch.device

        results = self.forward(batch)
        val_loss = self.model.loss_function(*results,
                                            M_N=self.params['batch_size'] / self.num_val_imgs,
                                            optimizer_idx=optimizer_idx,
                                            batch_idx=batch_idx)
        self.log("val_loss", val_loss, on_step=True, on_epoch=False, prog_bar=False, logger=True)

        return val_loss

    def on_train_start(self):
        samples_orig = next(iter(self.sample_dataloader))
        self.logger.experiment.add_images('samples_orig', samples_orig[:64])

    def on_validation_end(self):
        # avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        # tensorboard_logs = {'avg_val_loss': avg_loss}
        self.sample_images()
        # return {'val_loss': avg_loss, 'log': tensorboard_logs}

    def sample_images(self):
        # Get sample reconstruction image
        test_input = next(iter(self.sample_dataloader))
        test_input = test_input.to(self.curr_device)
        # test_label = test_label.to(self.curr_device)
        recons = self.model.generate(test_input)
        # vutils.save_image(recons.data,
        #                   f"{self.logger.save_dir}{self.logger.name}/version_{self.logger.version}/"
        #                   f"recons_{self.logger.name}_{self.current_epoch}.png",
        #                   normalize=True,
        #                   nrow=12)

        self.logger.experiment.add_images('samples_rec', recons[:64], self.current_epoch)

        # vutils.save_image(test_input.data,
        #                   f"{self.logger.save_dir}{self.logger.name}/version_{self.logger.version}/"
        #                   f"real_img_{self.logger.name}_{self.current_epoch}.png",
        #                   normalize=True,
        #                   nrow=12)

        try:
            samples = self.model.sample(144,
                                        self.curr_device)
            # vutils.save_image(samples.cpu().data,
            #                   f"{self.logger.save_dir}{self.logger.name}/version_{self.logger.version}/"
            #                   f"{self.logger.name}_{self.current_epoch}.png",
            #                   normalize=True,
            #                   nrow=12)
            self.logger.experiment.add_images('samples_gen', samples[:64], self.current_epoch)
        except:
            pass

        del test_input, recons  # , samples

    def configure_optimizers(self):

        optims = []
        scheds = []

        optimizer = optim.Adam(self.model.parameters(),
                               lr=self.params['LR'],
                               weight_decay=self.params['weight_decay'])
        optims.append(optimizer)
        # Check if more than 1 optimizer is required (Used for adversarial training)
        try:
            if self.params['LR_2'] is not None:
                optimizer2 = optim.Adam(getattr(self.model, self.params['submodel']).parameters(),
                                        lr=self.params['LR_2'])
                optims.append(optimizer2)
        except:
            pass

        try:
            if self.params['scheduler_gamma'] is not None:
                scheduler = optim.lr_scheduler.ExponentialLR(optims[0],
                                                             gamma=self.params['scheduler_gamma'])
                scheds.append(scheduler)

                # Check if another scheduler is required for the second optimizer
                try:
                    if self.params['scheduler_gamma_2'] is not None:
                        scheduler2 = optim.lr_scheduler.ExponentialLR(optims[1],
                                                                      gamma=self.params['scheduler_gamma_2'])
                        scheds.append(scheduler2)
                except:
                    pass
                return optims, scheds
        except:
            return optims

    @data_loader
    def train_dataloader(self):
        transform = self.data_transforms()

        if self.params['dataset'] == 'celeba':
            dataset = CustomDataSet(n=100000, transform=transform)

        else:
            raise ValueError('Undefined dataset type')

        self.num_train_imgs = len(dataset)
        return DataLoader(dataset,
                          batch_size=self.params['batch_size'],
                          num_workers=12,
                          shuffle=True,
                          drop_last=True)

    @data_loader
    def val_dataloader(self):
        transform = self.data_transforms()

        if self.params['dataset'] == 'celeba':
            self.sample_dataloader = DataLoader(CustomDataSet(n=10000, transform=transform),
                                                num_workers=12,
                                                batch_size=144,
                                                shuffle=False,
                                                drop_last=True)
            self.num_val_imgs = len(self.sample_dataloader)
        else:
            raise ValueError('Undefined dataset type')

        return self.sample_dataloader

    def data_transforms(self):

        # SetRange = transforms.Lambda(lambda X: 2 * X - 1.)
        # SetScale = transforms.Lambda(lambda X: X / X.sum(0).expand_as(X))

        if self.params['dataset'] == 'celeba':
            transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                            transforms.CenterCrop(148),
                                            transforms.Resize(self.params['img_size']),
                                            transforms.ToTensor()])
                                            # transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                            #                      std=[0.229, 0.224, 0.225])])
                                            # SetRange])
                                            # SetScale])
        else:
            raise ValueError('Undefined dataset type')
        return transform


class CustomDataSet(Dataset):
    def __init__(self, n, transform):
        self.main_dir = "/home/simon/Pictures/img_align_celeba"
        self.transform = transform
        all_imgs = random.sample(os.listdir(self.main_dir), k=n)
        self.total_imgs = natsort.natsorted(all_imgs)

    def __len__(self):
        return len(self.total_imgs)

    def __getitem__(self, idx):
        img_loc = os.path.join(self.main_dir, self.total_imgs[idx])
        image = Image.open(img_loc).convert("RGB")
        tensor_image = self.transform(image)
        return tensor_image
