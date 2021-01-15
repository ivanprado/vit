"""
This example is largely adapted from pl_examples/domain_templates/imagenet.py
to being able to train timm Vit models
"""
import os
from argparse import ArgumentParser, Namespace

from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import NeptuneLogger

from timm.data.transforms import _pil_interp

import timm
import torch
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data
import torch.utils.data.distributed
import torchvision.datasets as datasets
import torchvision.transforms as transforms

import pytorch_lightning as pl
from pytorch_lightning.core import LightningModule
from timm.models import vision_transformer
from timm.models.vision_transformer import VisionTransformer, _cfg


class ImageNetLightningModelForVit(LightningModule):
    # pull out vit models
    MODEL_NAMES = sorted(list(vision_transformer.default_cfgs.keys() | {"tiny"}))

    def __init__(
            self,
            arch: str,
            pretrained: bool,
            lr: float,
            weight_decay: int,
            data_path: str,
            batch_size: int,
            workers: int,
            **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.arch = arch
        self.pretrained = pretrained
        self.lr = lr
        self.weight_decay = weight_decay
        self.data_path = data_path
        self.batch_size = batch_size
        self.workers = workers
        if self.arch == "tiny":
            # Tiny empty model for development purposes.
            img_size = [32, 32]
            self.model_cfg = _cfg(input_size=[3] + img_size)
            self.model = VisionTransformer(
                img_size=img_size, patch_size=4, in_chans=3, num_classes=1000, embed_dim=16, depth=2,
                num_heads=1)
        else:
            self.model: VisionTransformer = timm.create_model(self.arch, pretrained=self.pretrained)
            self.model_cfg = vision_transformer.default_cfgs[self.arch]

    def setup(self, stage: str):
        # Configuring the head of the model to the number of classes
        train_dir = os.path.join(self.data_path, 'train')
        train_dataset = datasets.ImageFolder(
            train_dir)
        self.model.reset_classifier(len(train_dataset.classes))

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, target = batch
        output = self(images)
        loss_train = F.cross_entropy(output, target)
        acc1, acc5 = self.__accuracy(output, target, topk=(1, 5))
        self.log('train_loss', loss_train, on_step=True, on_epoch=True, logger=True)
        self.log('train_acc1', acc1, on_step=True, prog_bar=True, on_epoch=True, logger=True)
        self.log('train_acc5', acc5, on_step=True, on_epoch=True, logger=True)
        return loss_train

    def validation_step(self, batch, batch_idx):
        images, target = batch
        output = self(images)
        loss_val = F.cross_entropy(output, target)
        acc1, acc5 = self.__accuracy(output, target, topk=(1, 5))
        self.log('val_loss', loss_val, on_step=True, on_epoch=True)
        self.log('val_acc1', acc1, on_step=True, prog_bar=True, on_epoch=True)
        self.log('val_acc5', acc5, on_step=True, on_epoch=True)

    @staticmethod
    def __accuracy(output, target, topk=(1,)):
        """Computes the accuracy over the k top predictions for the specified values of k"""
        with torch.no_grad():
            maxk = max(topk)
            batch_size = target.size(0)

            _, pred = output.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))

            res = []
            for k in topk:
                correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
                res.append(correct_k.mul_(100.0 / batch_size))
            return res

    def configure_optimizers(self):
        # optimizer = optim.SGD(
        #     self.parameters(),
        #     lr=self.lr,
        #     momentum=self.momentum,
        #     weight_decay=self.weight_decay
        # )
        optimizer = optim.Adam(self.parameters(),
                               lr=0.3,
                               betas=[0.9, 0.999],
                               weight_decay=self.weight_decay)
        scheduler = lr_scheduler.LambdaLR(
            optimizer,
            lambda epoch: 0.1 ** (epoch // 30)
        )
        return [optimizer], [scheduler]

    def train_dataloader(self):
        cfg = self.model_cfg
        normalize = transforms.Normalize(
            mean=cfg['mean'],
            std=cfg['std'],
        )

        train_dir = os.path.join(self.data_path, 'train')
        train_dataset = datasets.ImageFolder(
            train_dir,
            transforms.Compose([
                transforms.RandomResizedCrop(cfg['input_size'][1:]),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]))

        train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.workers,
        )
        return train_loader

    def val_dataloader(self):
        cfg = self.model_cfg
        crop_pct = cfg['crop_pct']
        img_size = cfg['input_size'][1:]
        scale_size = tuple([int(x / crop_pct) for x in img_size])

        normalize = transforms.Normalize(
            mean=cfg['mean'],
            std=cfg['std'],
        )
        val_dir = os.path.join(self.data_path, 'val')
        val_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(val_dir, transforms.Compose([
                transforms.Resize(scale_size, _pil_interp(cfg['interpolation'])),
                transforms.CenterCrop(img_size),
                transforms.ToTensor(),
                normalize,
            ])),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.workers,
        )
        return val_loader

    def test_dataloader(self):
        return self.val_dataloader()

    def test_step(self, *args, **kwargs):
        return self.validation_step(*args, **kwargs)

    def test_epoch_end(self, *args, **kwargs):
        outputs = self.validation_epoch_end(*args, **kwargs)

        def substitute_val_keys(out):
            return {k.replace('val', 'test'): v for k, v in out.items()}

        outputs = {
            'test_loss': outputs['val_loss'],
            'progress_bar': substitute_val_keys(outputs['progress_bar']),
            'log': substitute_val_keys(outputs['log']),
        }
        return outputs

    @staticmethod
    def add_model_specific_args(parent_parser):  # pragma: no-cover
        model_names = ImageNetLightningModelForVit.MODEL_NAMES
        parser = ArgumentParser(parents=[parent_parser])
        parser.add_argument('-a', '--arch', metavar='ARCH', default=model_names[0],
                            choices=model_names,
                            help=('model architecture: ' + ' | '.join(model_names)
                                  + ' (default: resnet18)'))
        parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                            help='number of data loading workers (default: 4)')
        parser.add_argument('-b', '--batch-size', default=256, type=int,
                            metavar='N',
                            help='mini-batch size (default: 256), this is the total '
                                 'batch size of all GPUs on the current node when '
                                 'using Data Parallel or Distributed Data Parallel')
        parser.add_argument('--lr', '--learning-rate', default=0.3, type=float,
                            metavar='LR', help='initial learning rate', dest='lr')
        parser.add_argument('--wd', '--weight-decay', default=1e-1, type=float,
                            metavar='W', help='weight decay (default: 1e-4)',
                            dest='weight_decay')
        parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                            help='use pre-trained model')
        return parser


def main(args: Namespace) -> None:
    if args.seed is not None:
        pl.seed_everything(args.seed)

    if args.accelerator == 'ddp':
        # When using a single GPU per process and per
        # DistributedDataParallel, we need to divide the batch size
        # ourselves based on the total number of GPUs we have
        args.batch_size = int(args.batch_size / max(1, args.gpus))
        args.workers = int(args.workers / max(1, args.gpus))

    model = ImageNetLightningModelForVit(**vars(args))

    kwargs = {}
    if args.neptune:
        kwargs = dict(logger=NeptuneLogger(
            project_name="ivan.prado/vit-sandbox",
            params={k: v for k, v in vars(args).items()
                    if isinstance(v, (type(None), int, float, str))}))

    lr_monitor = LearningRateMonitor(logging_interval='step')
    trainer = pl.Trainer.from_argparse_args(args, callbacks=[lr_monitor], **kwargs)

    if args.evaluate:
        trainer.test(model)
    else:
        trainer.fit(model)


def run_cli():
    parent_parser = ArgumentParser(add_help=False)
    parent_parser = pl.Trainer.add_argparse_args(parent_parser)
    parent_parser.add_argument('--data-path', metavar='DIR', type=str,
                               help='path to dataset')
    parent_parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                               help='evaluate model on validation set')
    parent_parser.add_argument('--seed', type=int, default=42,
                               help='seed for initializing training.')
    parent_parser.add_argument('--neptune', action='store_true',
                               help='enable logging to Neptune')
    parser = ImageNetLightningModelForVit.add_model_specific_args(parent_parser)
    parser.set_defaults(
        profiler=False,
        deterministic=True,
        max_epochs=90,
    )
    args = parser.parse_args()
    main(args)


if __name__ == '__main__':
    run_cli()
