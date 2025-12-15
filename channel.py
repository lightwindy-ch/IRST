import torch.nn as nn
import numpy as np
import scipy.special as sc
from scipy.stats import rv_continuous
import os
import torch
import time
# torch.cuda.set_device('')

class ShadowedRiceDistribution(rv_continuous):
    def _pdf(self, s):
        b = [0.158, 0.063, 0.251, 0.126]
        m = [19.4, 0.739, 5.21, 10.1]
        omega = [1.29, 8.97 * 10 ** (-4), 0.278, 0.835]
        # a = [1/0.16124, 1/0.06060, 1/0.94092, 1/0.07653]
        a = [1,1,1,1]

        F1 = sc.hyp1f1(m[i], 1, omega[i] * s / (2 * b[i] * (2 * b[i] * m[i] + omega[i])))
        Ps = a[i]*(2 * b[i] * m[i] / (2 * b[i] * m[i] + omega[i])) ** m[i] * (1 / (2 * b[i])) ** (-s / (2 * b[i])) * F1
        return Ps

    def shadowed_rice_channel(m):
        global i
        i = m
        custom_dist = ShadowedRiceDistribution(a=0, b=5, name='custom_dist')
        samples = custom_dist.rvs(size=1)
        # print(samples[:10])
        return samples

class Channel(nn.Module):
    """
    Currently the channel model is either error free, erasure channel,
    rayleigh channel or the AWGN channel.
    """

    def __init__(self, args, config):
        super(Channel, self).__init__()
        self.config = config
        self.chan_type = args.channel_type
        self.device = config.device
        # self.h = torch.sqrt(torch.randn(1) ** 2
        #                     + torch.randn(1) ** 2) / 1.414
        self.h = ShadowedRiceDistribution.shadowed_rice_channel(0)
        if config.logger:
            config.logger.info('【Channel】: Built {} channel, SNR {} dB.'.format(
                args.channel_type, args.multiple_snr))

    def shadowed_rice_noise_layer(self, input_layer, std, name=None):
        # device = 2
        device = input_layer.get_device()
        noise_real = torch.normal(mean=0.0, std=std, size=np.shape(input_layer), device=device)
        noise_imag = torch.normal(mean=0.0, std=std, size=np.shape(input_layer), device=device)
        noise = noise_real + 1j * noise_imag
        x = torch.from_numpy(ShadowedRiceDistribution.shadowed_rice_channel(0))
        x = torch.tensor(x, dtype=torch.float32).to(device)                     
        h = torch.zeros(size=np.shape(input_layer), device=device)
        h = x+h
        if self.config.CUDA:
            # noise = noise.to(input_layer.get_device())
            # h = h.to(input_layer.get_device())
            noise = noise.to(device)
            h = h.to(device)
        return input_layer * h + noise


    def gaussian_noise_layer(self, input_layer, std, name=None):
        device = input_layer.get_device()
        noise_real = torch.normal(mean=0.0, std=std, size=np.shape(input_layer), device=device)
        noise_imag = torch.normal(mean=0.0, std=std, size=np.shape(input_layer), device=device)
        noise = noise_real + 1j * noise_imag
        return input_layer + noise

    def rayleigh_noise_layer(self, input_layer, std, name=None):
        noise_real = torch.normal(mean=0.0, std=std, size=np.shape(input_layer))
        noise_imag = torch.normal(mean=0.0, std=std, size=np.shape(input_layer))
        noise = noise_real + 1j * noise_imag
        h = torch.sqrt(torch.normal(mean=0.0, std=1, size=np.shape(input_layer)) ** 2
                       + torch.normal(mean=0.0, std=1, size=np.shape(input_layer)) ** 2) / np.sqrt(2)
        if self.config.CUDA:
            noise = noise.to(input_layer.get_device())
            h = h.to(input_layer.get_device())
        return input_layer * h + noise


    def complex_normalize(self, x, power):
        pwr = torch.mean(x ** 2) * 2
        out = np.sqrt(power) * x / torch.sqrt(pwr)
        return out, pwr


    def forward(self, input, chan_param, avg_pwr=False):
        if avg_pwr:
            power = 1
            channel_tx = np.sqrt(power) * input / torch.sqrt(avg_pwr * 2)
        else:
            channel_tx, pwr = self.complex_normalize(input, power=1)
        input_shape = channel_tx.shape
        channel_in = channel_tx.reshape(-1)
        L = channel_in.shape[0]
        channel_in = channel_in[:L // 2] + channel_in[L // 2:] * 1j
        channel_output = self.complex_forward(channel_in, chan_param)
        channel_output = torch.cat([torch.real(channel_output), torch.imag(channel_output)])
        channel_output = channel_output.reshape(input_shape)
        if self.chan_type == 1 or self.chan_type == 'awgn':
            noise = (channel_output - channel_tx).detach()
            noise.requires_grad = False
            channel_tx = channel_tx + noise
            if avg_pwr:
                return channel_tx * torch.sqrt(avg_pwr * 2)
            else:
                return channel_tx * torch.sqrt(pwr)
        elif self.chan_type == 2 or self.chan_type == 'rayleigh':
            if avg_pwr:
                return channel_output * torch.sqrt(avg_pwr * 2)
            else:
                return channel_output * torch.sqrt(pwr)
        elif self.chan_type == 3 or self.chan_type == 'shadowed_rice':
            if avg_pwr:
                return channel_output * torch.sqrt(avg_pwr * 2)
            else:
                return channel_output * torch.sqrt(pwr)

    def complex_forward(self, channel_in, chan_param):
        if self.chan_type == 0 or self.chan_type == 'none':
            return channel_in

        elif self.chan_type == 1 or self.chan_type == 'awgn':
            channel_tx = channel_in
            sigma = np.sqrt(1.0 / (2 * 10 ** (chan_param / 10)))
            chan_output = self.gaussian_noise_layer(channel_tx,
                                                    std=sigma,
                                                    name="awgn_chan_noise")
            return chan_output

        elif self.chan_type == 2 or self.chan_type == 'rayleigh':
            channel_tx = channel_in
            sigma = np.sqrt(1.0 / (2 * 10 ** (chan_param / 10)))
            chan_output = self.rayleigh_noise_layer(channel_tx,
                                                    std=sigma,
                                                    name="rayleigh_chan_noise")
            return chan_output

        elif self.chan_type == 3 or self.chan_type == 'shadowed_rice':
            channel_tx = channel_in
            sigma = np.sqrt(1.0 / (2 * 10 ** (chan_param / 10)))
            chan_output = self.shadowed_rice_noise_layer(channel_tx,
                                                    std=sigma,
                                                    name="shadowed_rice_chan_noise")
            return chan_output


    def noiseless_forward(self, channel_in):
        channel_tx = self.normalize(channel_in, power=1)
        return channel_tx