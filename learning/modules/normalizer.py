# MIT License
#
# Copyright (c) 2020 Preferred Networks, Inc.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
import numpy as np

import torch
from torch import nn
from typing import Tuple


class EmpiricalNormalization(nn.Module):
    """Normalize mean and variance of values based on empirical values.
    Args:
        shape (int or tuple of int): Shape of input values except batch axis.
        batch_axis (int): Batch axis.
        eps (float): Small value for stability.
        dtype (dtype): Dtype of input values.
        until (int or None): If this arg is specified, the link learns input values until the sum of batch sizes
        exceeds it.
        update_obs_norm (bool): If true, learns updates mean and variance
    """

    def __init__(
        self,
        shape,
        batch_axis=0,
        eps=1e-2,
        dtype=np.float32,
        until=None,
        clip_threshold=None,
        update_obs_norm=True,
    ):
        super(EmpiricalNormalization, self).__init__()
        dtype = np.dtype(dtype)
        self.batch_axis = batch_axis
        self.eps = eps
        self.until = until
        self.clip_threshold = clip_threshold
        self.register_buffer(
            "_mean",
            torch.tensor(np.expand_dims(np.zeros(shape, dtype=dtype), batch_axis)),
        )
        self.register_buffer(
            "_var",
            torch.tensor(np.expand_dims(np.ones(shape, dtype=dtype), batch_axis)),
        )
        self.register_buffer("count", torch.tensor(0))
        self.in_features = shape[0]

        # cache
        self._cached_std_inverse = torch.tensor(np.expand_dims(np.ones(shape, dtype=dtype), batch_axis))
        self._is_std_cached = False
        self._is_training = update_obs_norm

    @property
    def mean(self):
        return torch.squeeze(self._mean, self.batch_axis).clone()

    @property
    def std(self):
        return torch.sqrt(torch.squeeze(self._var, self.batch_axis)).clone()

    @property
    def _std_inverse(self):
        if self._is_std_cached is False:
            self._cached_std_inverse = (self._var + self.eps) ** -0.5

        return self._cached_std_inverse

    @torch.jit.unused
    @torch.no_grad()
    def experience(self, x):
        """Learn input values without computing the output values of them"""

        if self.until is not None:
            if self.count >= self.until:
                return

        count_x = x.shape[self.batch_axis]
        if count_x == 0:
            return

        self.count += count_x
        rate = count_x / self.count.float()
        assert rate > 0
        assert rate <= 1

        var_x = torch.var(x, dim=self.batch_axis, unbiased=False, keepdim=True)
        mean_x = torch.mean(x, dim=self.batch_axis, keepdim=True)
        delta_mean = mean_x - self._mean
        self._mean += rate * delta_mean
        self._var += rate * (var_x - self._var + delta_mean * (mean_x - self._mean))

        # clear cache
        self._is_std_cached = False

    def forward(self, x):
        """Normalize mean and variance of values based on emprical values.
        Args:
            x (ndarray or Variable): Input values
        Returns:
            ndarray or Variable: Normalized output values
        """

        if self._is_training:
            self.experience(x)

        if not x.is_cuda:
            self._is_std_cached = False
        normalized = (x - self._mean) * self._std_inverse
        if self.clip_threshold is not None:
            normalized = torch.clamp(normalized, -self.clip_threshold, self.clip_threshold)
        if not x.is_cuda:
            self._is_std_cached = False
        return normalized

    @torch.jit.unused
    def inverse(self, y):
        std = torch.sqrt(self._var + self.eps)
        return y * std + self._mean

    def load_numpy(self, mean, var, count, device="cpu"):
        self._mean = torch.from_numpy(np.expand_dims(mean, self.batch_axis)).to(device)
        self._var = torch.from_numpy(np.expand_dims(var, self.batch_axis)).to(device)
        self.count = torch.tensor(count).to(device)

class Normalizer:
    def __init__(self, input_dim, device, epsilon=1e-2, clip=10.0):
        self.device = device
        self.mean = torch.zeros(input_dim, device=self.device)
        self.var = torch.ones(input_dim, device=self.device)
        self.count = epsilon
        self.epsilon = epsilon
        self.clip = clip

    def normalize(self, data):
        mean_ = self.mean
        std_ = torch.sqrt(self.var + self.epsilon)
        return torch.clamp((data - mean_) / std_, -self.clip, self.clip)

    def update(self, data):
        batch_mean = torch.mean(data, dim=0)
        batch_var = torch.var(data, dim=0)
        batch_count = data.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        new_var = (self.var * self.count +
                   batch_var * batch_count +
                   torch.square(delta) * self.count * batch_count / tot_count) / tot_count
        self.mean = new_mean
        self.var = new_var
        self.count = tot_count
