import torch
import pandas as pd


def pearson(x: torch.Tensor, y: torch.Tensor):
    n = x.size(1)
    # simple sums
    sum1 = torch.sum(x)
    sum2 = torch.sum(y)
    # sum up the squares
    sum1_pow = torch.sum(torch.pow(x, 2))
    sum2_pow = torch.sum(torch.pow(y, 2))
    # sum up the products
    p_sum = torch.sum(x * y)
    # 分子num，分母den
    num = p_sum - (sum1*sum2/n)
    den = torch.sqrt((sum1_pow-torch.pow(sum1, 2)/n)*(sum2_pow-torch.pow(sum2, 2)/n))
    if den.data == 0:
        return 0.0
    return num/den


def mae(pred_value: torch.Tensor, gt_value: torch.Tensor):
    with torch.no_grad():
        return torch.mean(torch.abs(pred_value - gt_value))


def rmse(pred_value: torch.Tensor, gt_value: torch.Tensor):
    with torch.no_grad():
        return torch.mean(torch.sqrt(torch.square(pred_value - gt_value)))


def std(pred_value: torch.Tensor, gt_value: torch.Tensor):
    with torch.no_grad():
        return torch.std(pred_value)


def r(pred_value: torch.Tensor, gt_value: torch.Tensor):
    with torch.no_grad():
        return pearson(pred_value, gt_value)


class MetricTracker:
    def __init__(self, *keys, writer=None):
        self.writer = writer
        self._data = pd.DataFrame(index=keys, columns=['total', 'counts', 'average'])
        self.reset()

    def reset(self):
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, key, value, n=1):
        if self.writer is not None:
            self.writer.add_scalar(key, value)
        self._data.total[key] += value * n
        self._data.counts[key] += n
        self._data.average[key] = self._data.total[key] / self._data.counts[key]

    def avg(self, key):
        return self._data.average[key]

    def result(self):
        return dict(self._data.average)
