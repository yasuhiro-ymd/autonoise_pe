import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import init
import numpy as np
import pytorch_lightning as pl
import sys
from tqdm import tqdm

class GMM(pl.LightningModule):
    """Gaussian mixture model.

    Contains functions for calculating Gaussian mixture model
    loglikelihood and for autoregressive image sampling.

    Attributes:
        n_gaussians: An integer for the number of components in the Gaussian
        mixture model.
        noise_mean: Float for the mean of the noise samples, used to normalise
        data.
        noise_std: Float for the standard deviation of the noise samples, also
        used to normalise the data

    """
    
    def __init__(self, n_gaussians, noise_mean, noise_std, lr):
        super().__init__()  
        self.n_gaussians = n_gaussians
        self.noise_mean = noise_mean
        self.noise_std = noise_std
        self.lr = lr
        
    def get_gaussian_params(self, pred):
        blockSize = self.n_gaussians
        means = pred[:,:blockSize,...]
        stds = torch.sqrt(torch.exp(pred[:,blockSize:2*blockSize,...]))
        weights = torch.exp(pred[:,2*blockSize:3*blockSize,...])
        weights = weights / torch.sum(weights,dim = 1, keepdim = True)
        return means, stds, weights   
    
    def loglikelihood(self, x, s=None):
        """Calculates loglikelihood of noise input.

        Passes noise image through network to obtain Gaussian mixture model
        parameters. Uses those parameters to calculate loglikelihood of noise
        image.


        Parameters
        ----------
        x : torch.FloatTensor
            This is a noise sample when training the noise model, but, when using
            the noise model to evaluate the denoiser, this is a noisy image.
        s : torch.FloatTensor
            When training the noise model, this should be left as none. When training
            the denoiser this is the estimated signal and will be subtracted from the
            noisy image to obtain a noise image.

        Returns
        -------
        loglikelihoods : torch.FloatTensor
            The elementwise loglikelihood of the input.

        """
        
        if s is None:
            s = torch.zeros_like(x)

        n = x - s

        if n.shape[1] == 1:
            n_norm = (n - self.noise_mean) / self.noise_std
            model_in = n_norm
            n_like = n_norm
        else:
            # Channel 0: noise (normalised). Remaining channels: e.g. positional
            # encoding, passed through without noise_mean / noise_std.
            n0 = (n[:, :1] - self.noise_mean) / self.noise_std
            model_in = torch.cat([n0, n[:, 1:]], dim=1)
            n_like = n0

        if self.training:
            pred = self.forward(model_in)
        else:
            pred = self.forward(model_in).detach()

        means, stds, weights = self.get_gaussian_params(pred)
        likelihoods = (
            -0.5 * ((means - n_like) / stds) ** 2
            - torch.log(stds)
            - np.log(2.0 * np.pi) * 0.5
        )
        temp = torch.max(likelihoods, dim = 1, keepdim = True)[0].detach()
        likelihoods=torch.exp( likelihoods -temp) * weights
        loglikelihoods = torch.log(torch.sum(likelihoods, dim = 1, keepdim = True))
        loglikelihoods = loglikelihoods + temp 
        return loglikelihoods
    
    def sampleFromMix(self, means, stds, weights):
        num_components = means.shape[1]
        shape = means[:,0,...].shape
        selector = torch.rand(shape, device = means.device)
        gauss = torch.normal(means[:,0,...]*0, means[:,0,...]*0 + 1)
        out = means[:,0,...]*0

        for i in range(num_components):
            mask = torch.zeros(shape)
            mask = (selector<weights[:,i,...]) & (selector>0)
            out += mask* (means[:,i,...] + gauss*stds[:,i,...])
            selector -= weights[:,i,...]
        
        del gauss
        del selector
        del shape
        return out    
    
    @torch.no_grad()
    def sample(self, img_shape, positional_encoding=None, noise_channels=None):
        """Samples images from the trained autoregressive model.

        Parameters
        ----------
        img_shape : List or tuple
            Shape ``[N, C, H, W]`` (batch, channels, height, width). For data
            with noise plus positional encodings, set ``C = 1 + d_model`` and pass
            ``positional_encoding`` with the fixed PE channels.
        positional_encoding : torch.FloatTensor, optional
            Shape ``[N, C - noise_channels, H, W]``. Copied into ``img`` after
            the noise channels (e.g. channels ``1:`` when ``noise_channels=1``).
        noise_channels : int, optional
            How many leading channels are modelled autoregressively. Default is
            ``img_shape[1]`` (all channels), or ``1`` when ``positional_encoding``
            is given.

        Returns
        -------
        torch.FloatTensor
            The generated tensor in the same layout as training inputs (noise
            channels are denormalised; conditioning channels are unchanged).

        """
        n_batch, c_tot, height, width = img_shape
        if noise_channels is None:
            noise_channels = 1 if positional_encoding is not None else c_tot
        if positional_encoding is not None:
            if positional_encoding.shape != (
                n_batch,
                c_tot - noise_channels,
                height,
                width,
            ):
                raise ValueError(
                    "positional_encoding shape must be [N, C - noise_channels, H, W]"
                )

        img = torch.zeros(img_shape, dtype=torch.float, device=self.device)
        if positional_encoding is not None:
            img[:, noise_channels:] = positional_encoding.to(
                device=self.device, dtype=torch.float
            )

        for h in tqdm(range(height), leave=False):
            for w in range(width):
                for c in range(noise_channels):
                    pred = self.forward(img[:, :, : h + 1, :])
                    means, stds, weights = self.get_gaussian_params(pred)
                    means = means[:, :, h, w][..., np.newaxis, np.newaxis]
                    stds = stds[:, :, h, w][..., np.newaxis, np.newaxis]
                    weights = weights[:, :, h, w][..., np.newaxis, np.newaxis]
                    samp = self.sampleFromMix(means, stds, weights).detach()
                    img[:, c, h, w] = samp[:, 0, 0]

        if noise_channels == c_tot:
            return img * self.noise_std + self.noise_mean
        out = img.clone()
        out[:, :noise_channels] = (
            out[:, :noise_channels] * self.noise_std + self.noise_mean
        )
        return out
    
    def training_step(self, batch, batch_idx):
        loss = -torch.mean(self.loglikelihood(batch))
        self.log("train/nll", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = -torch.mean(self.loglikelihood(batch))
        self.log("val/nll", loss, prog_bar=True)
        
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.99)
        return [optimizer], [scheduler]
