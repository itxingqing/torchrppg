import torch
import argparse
import numpy as np
import collections
import losses.loss as module_loss
import models.model as module_arch
from parse_config import ConfigParser
import data_loaders.data_loaders as module_data
import metrices.metric as module_metric
from trainer import Trainer

# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True
np.random.seed(SEED)


def main(config):
    logger = config.get_logger('train')

    # Set DataLoader
    train_dataloader = config.init_obj('train_dataloader', module_data)
    val_dataloader = config.init_obj('val_dataloader', module_data)

    # Set loss and metrics
    criterion = config.init_obj('loss', module_loss)
    metrics = [getattr(module_metric, met) for met in config['metrics']]

    # Set model
    model = config.init_obj('arch', module_arch)
    logger.info(model)

    # Set optimizer
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = config.init_obj('optimizer', torch.optim, trainable_params)
    lr_scheduler = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizer)

    # Set Trainer
    trainer = Trainer(model, criterion, metrics, optimizer,
                      config=config,
                      data_loader=train_dataloader,
                      valid_data_loader=val_dataloader,
                      lr_scheduler=lr_scheduler)
    trainer.train()


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default='config.json', type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default='0,1', type=str,
                      help='indices of GPUs to enable (default: all)')

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'], type=float, target='optimizer;args;lr'),
        CustomArgs(['--bs', '--batch_size'], type=int, target='train_dataloader;args;batch_size')
    ]
    train_config = ConfigParser.from_args(args, options)
    main(train_config)