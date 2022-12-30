import numpy as np
import torch
from torchvision.utils import make_grid
from base import BaseTrainer
from utils import inf_loop
from utils.ppg_process_common_function import postprocess
from metrices import MetricTracker


class Trainer(BaseTrainer):
    """
    Trainer class
    """
    def __init__(self, model, criterion, metric_ftns, optimizer, config, data_loader,
                 valid_data_loader=None, lr_scheduler=None, len_epoch=None):
        super().__init__(model, criterion, metric_ftns, optimizer, config)
        self.config = config
        self.data_loader = data_loader
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.data_loader)
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(data_loader.batch_size))

        self.train_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)
        self.valid_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()
        pred_value = torch.zeros(1, 1).cuda()
        gt_value = torch.zeros(1, 1).cuda()
        pred_wave = torch.zeros(1, 160).cuda()
        gt_wave = torch.zeros(1, 160).cuda()
        for batch_idx, (data, target, value, subject, fps) in enumerate(self.data_loader):
            data, target, value, subject= data.to(self.device), target.to(self.device), value.to(self.device), subject.to(self.device)
            for op in self.optimizer:
                op.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target, value, subject, fps)
            if len(output) > 1:
                pred_wave_temp = output[0]
                pred_value_temp = postprocess(output[0], fps=fps, length=160)
                pred_value_temp = pred_value_temp.cuda()
            else:
                pred_wave_temp = output
                pred_value_temp = postprocess(output, fps=fps, length=160)
                pred_value_temp = pred_value_temp.cuda()
            gt_val_temp = torch.mean(value, dim=1).view(1, -1).cuda()
            pred_value = torch.cat((pred_value, pred_value_temp), dim=0).cuda()
            gt_value = torch.cat((gt_value, gt_val_temp), dim=0).cuda()

            gt_wave_temp = target
            pred_wave = torch.cat((pred_wave, pred_wave_temp), dim=0).cuda()
            gt_wave = torch.cat((gt_wave, gt_wave_temp), dim=0).cuda()
            loss.backward()
            for op in self.optimizer:
                op.step()

            # self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.train_metrics.update('loss', loss.item())
            if batch_idx % self.log_step == 0:
                self.logger.debug('Train Epoch: {} {} Loss: {:.6f}'.format(
                    epoch,
                    self._progress(batch_idx),
                    loss.item()))
                # self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

            if batch_idx == self.len_epoch:
                break
        for met in self.metric_ftns:
            if met.__name__ == 'r':
                self.train_metrics.update(met.__name__, met(pred_wave[1:, :], gt_wave[1:, :]))
            else:
                self.train_metrics.update(met.__name__, met(pred_value[1:, :], gt_value[1:, :]))
        log = self.train_metrics.result()

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(**{'val_'+k : v for k, v in val_log.items()})

        if self.lr_scheduler is not None:
            for lr in self.lr_scheduler:
                lr.step()
        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.valid_metrics.reset()
        pred_value = torch.zeros(1, 1).cuda()
        gt_value = torch.zeros(1, 1).cuda()
        pred_wave = torch.zeros(1, 160).cuda()
        gt_wave = torch.zeros(1, 160).cuda()
        with torch.no_grad():
            for batch_idx, (data, target, value, subject, fps) in enumerate(self.data_loader):
                data, target, value, subject = data.to(self.device), target.to(self.device), value.to(self.device), subject.to(self.device)

                output = self.model(data)
                loss = self.criterion(output, target, value, subject, fps)
                if len(output) > 1:
                    pred_wave_temp = output[0]
                    pred_value_temp = postprocess(output[0], fps=fps, length=160)
                    pred_value_temp = pred_value_temp.cuda()
                else:
                    pred_wave_temp = output
                    pred_value_temp = postprocess(output, fps=fps, length=160)
                    pred_value_temp = pred_value_temp.cuda()
                gt_val_temp = torch.mean(value, dim=1).view(1, -1).cuda()
                pred_value = torch.cat((pred_value, pred_value_temp), dim=0).cuda()
                gt_value = torch.cat((gt_value, gt_val_temp), dim=0).cuda()

                gt_wave_temp = target
                pred_wave = torch.cat((pred_wave, pred_wave_temp), dim=0).cuda()
                gt_wave = torch.cat((gt_wave, gt_wave_temp), dim=0).cuda()

                # self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                self.valid_metrics.update('loss', loss.item())
                # self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))
        # add histogram of model parameters to the tensorboard
        for met in self.metric_ftns:
            if met.__name__ == 'r':
                self.valid_metrics.update(met.__name__, met(pred_wave[1:, :], gt_wave[1:, :]))
            else:
                self.valid_metrics.update(met.__name__, met(pred_value[1:, :], gt_value[1:, :]))
        for name, p in self.model.named_parameters():
            self.writer.add_histogram(name, p, bins='auto')
        return self.valid_metrics.result()

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)
