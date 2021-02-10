from argparse import ArgumentParser
from typing import Optional, Union, List, Dict
import yaml

import torch
import torchvision
import pytorch_lightning as pl
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import timm 

from mlc import datasets
from mlc.utils import load_pretrained_weight, initialize_dataset
from mlc.loss import aggregate_loss, generate_loss

class MultiLabelClassifier(pl.LightningModule):
    def __init__(self, 
                 model: str, 
                 num_class: int,
                 in_chans: int,
                 learning_rate: float, 
                 loss_functions: Union[List[Dict], Dict],
                 checkpoint_path: str='',
                 pretrained:Optional[bool] = True):
        super().__init__()
        self.save_hyperparameters()  
        self.model = timm.create_model(model_name=self.hparams.model, 
                                       pretrained=pretrained, 
                                       checkpoint_path=checkpoint_path, 
                                       num_classes=self.hparams.num_class,
                                       in_chans=self.hparams.in_chans)

        # load pre-traned weight
        if self.hparams.checkpoint_path and self.hparams.pretrained:
            load_pretrained_weight(model=self.model, weight_path=self.hparams.checkpoint_path, device=self.device)
        
        # loss
        self.loss = generate_loss(loss_cfg=loss_functions)

    def forward(self, x):
        # use forward for inference/predictions
        logits = self.model(x)
        sig = torch.nn.Sigmoid()
        pred = sig(logits)
        return pred
    
    def training_step(self, dataloader, batch_idx):
        x, y = dataloader
        y_hat = self.forward(x)

        # calculate loss
        prefix = 'train'
        loss = aggregate_loss(loss_fs=self.loss, preds=y_hat, targets=y, prefix=prefix)

        # log 
        for log in loss:
            self.log(f'{log}', loss.get(log), on_epoch=True)
        
        return loss[f'{prefix}_total_loss']

    def validation_step(self, dataloader, batch_idx):
        x, y = dataloader
        y_hat = self.forward(x)

        # calculate loss
        prefix = 'val'
        loss = aggregate_loss(loss_fs=self.loss, preds=y_hat, targets=y, prefix=prefix)

        # log 
        for log in loss:
            self.log(f'{log}', loss.get(log), on_epoch=True)

    def test_step(self, dataloader, batch_idx):
        x, y = dataloader
        y_hat = self.forward(x)

        # calculate loss
        prefix = 'test'
        loss = aggregate_loss(loss_fs=self.loss, preds=y_hat, targets=y, prefix=prefix)

        # log 
        for log in loss:
            self.log(f'{log}', loss.get(log), on_epoch=True)

    def configure_optimizers(self):
        # self.hparams available because we called self.save_hyperparameters()
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)


def cli_main():
    pl.seed_everything(1234)

    # ------------
    # args
    # ------------
    parser = ArgumentParser()
    parser.add_argument('-c', '--cfg', default='', type=str, help='path to trainer config')
    parser.add_argument('-g', '--gpus', default=1, type=int, help='number of gpu for training')
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()
    config = yaml.safe_load(args.cfg)

    # ------------
    # data
    # ------------
    # trainsform
    train_transform = transforms.Compose([
                       transforms.ToPILImage(mode='RGBA'),
                       transforms.Resize((224, 224)), 
                       transforms.RandomHorizontalFlip(),
                       transforms.RandomRotation(10),
                       transforms.ToTensor(),
                       transforms.Normalize((0.5, 0.5, 0.5, 0.5), (0.5, 0.5, 0.5, 0.5))])
    val_transform = transforms.Compose([
                     transforms.ToPILImage(mode='RGBA'),
                     transforms.Resize((224, 224)), 
                     transforms.ToTensor(),
                     transforms.Normalize((0.5, 0.5, 0.5, 0.5), (0.5, 0.5, 0.5, 0.5))])
    test_transform = transforms.Compose([
                      transforms.ToPILImage(mode='RGBA'),
                      transforms.Resize((224, 224)), 
                      transforms.ToTensor(),
                      transforms.Normalize((0.5, 0.5, 0.5, 0.5), (0.5, 0.5, 0.5, 0.5))])

    # dataset
    dataset_cfg = str(config.dataset)
    dataset_name = str(dataset_cfg.name)
    root = str(dataset_cfg.root)
    metadata = str(dataset_cfg.metada)
    num_class = int(dataset.model.num_class)

    dataset_train = initialize_dataset(datasets, dataset_name, root=root, metadata=metadata, num_class=num_class, transform=train_transform)
    dataset_test =  initialize_dataset(datasets, dataset_name, root=root, metadata=metadata, num_class=num_class, transform=test_transform)
    dataset_train, dataset_val = random_split(dataset_train, [55000, 5000])

    # dataloader
    train_loader = DataLoader(dataset_train, batch_size=args.batch_size)
    val_loader = DataLoader(dataset_val, batch_size=args.batch_size)
    test_loader = DataLoader(dataset_test, batch_size=args.batch_size)

    # ------------
    # model
    # ------------
    model_name = str(config.model.model_name)
    num_class = int(config.model.num_class)
    lr = float(config.trainer.learning_rate)
    loss_f = config.loss
    chkpt_path = str(config.model.checkpoint)
    pretrained = bool(config.model.pretrained)
    model = MultiLabelClassifier(model=model_name,
                                 num_class=num_class,
                                 learning_rate=lr, 
                                 loss_functions=loss_f,
                                 checkpoint_path=chkpt_path,
                                 pretrained=pretrained)

    # ------------
    # training
    # ------------
    trainer = pl.Trainer.from_argparse_args(args)
    trainer.fit(model, train_loader, val_loader)

    # ------------
    # testing
    # ------------
    result = trainer.test(test_dataloaders=test_loader)
    print(result)


if __name__ == '__main__':
    cli_main()
