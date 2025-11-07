import torch  
import numpy as np
from pathlib import Path
from scipy.ndimage import convolve1d
from scipy.stats import norm
from scipy.signal import filtfilt, lfilter


class GaussianSmoother():
    def __init__(self, std, Fs=1, axis=0, kernel_multiplier=None, causal=False):
        """_summary_
        Args:
            std (float): standard deviation of the gaussian kernel to be used. In seconds. If Fs not provided, std 
                can also be though of as being in the unit of signal samples.
            Fs (int, optional): sampling rate of signal. Defaults to 1.
        """        
        self.std = std # In seconds
        self.Fs = Fs
        self.kernel_sigma = std * Fs # effective std for the gaussian kernel

        self.axis = axis
        if kernel_multiplier is None:
            conf = 0.99
            kernel_multiplier = norm.ppf( 1 - 0.5 * (1-conf) )
        self.kernel_multiplier = kernel_multiplier
        self.causal = causal

        if self.causal:
            sigma = self.kernel_sigma
        else:
            sigma = self.kernel_sigma / np.sqrt(2) # Because applying the smoothing filter twice with filtfilt multiplies its effective std by sqrt(2)

        m = self.kernel_multiplier
        filterLen = int(sigma*m + 0.5)
        x0 = 0
        xVals = np.arange(-filterLen, filterLen+1, 1)
        weights = np.exp( -0.5 * (xVals-x0)**2 / sigma**2 )
        if self.causal:
            w_norm = np.sum(weights)
        else:
            w_padded = np.concatenate((np.zeros_like(weights), weights, np.zeros_like(weights)))
            w_norm = np.sqrt( np.sum(convolve1d(w_padded, w_padded, mode='constant', cval=0, origin=0)) )

        weights = weights / w_norm

        self.weights = weights

    def apply(self, data, axis=None):
        if axis is None:
            axis = self.axis
        if self.causal:
            filtered_data = lfilter(self.weights, 1, data, axis=axis)
        else:
            filtered_data = filtfilt(self.weights, 1, data, axis=axis, method='pad', padtype='constant')
        return filtered_data



def convert_to_tensor(x, cat_dim=0):
    if isinstance(x, np.ndarray): 
        return torch.tensor(x, dtype=torch.float32) # use np.ndarray as middle step so that function works with tf tensors as well
    elif isinstance(x, list):
        return torch.cat(x, dim=cat_dim, dtype=torch.float32) 
    elif isinstance(x, torch.Tensor) and x.dtype == torch.float64: # change dtype to float32
        return x.float()
    else:
        return x

def smooth_spikes(spikes, sigma, delta_ms, switch_causal, ignore_nans=True): # delta in ms, sigma in ms
    if len(spikes.shape) == 3:
        num_trials, num_steps, num_feat = spikes.shape 
        spikes_resh = np.reshape(spikes, (-1, num_feat))
    else:
        spikes_resh = np.array(spikes)

    filter = GaussianSmoother(std=sigma, Fs=1/delta_ms, causal=switch_causal)
    smoothed_spikes_resh = np.empty_like(spikes_resh)
    for feat in range(spikes_resh.shape[1]):
        if ignore_nans:
            try:
                nan_inds = np.isnan(spikes_resh[:, feat]).bool()
            except:
                nan_inds = np.isnan(spikes_resh[:, feat]).astype(bool)

            nonnan_inds = ~nan_inds
            smoothed_spikes_resh[nonnan_inds, feat] = filter.apply(spikes_resh[nonnan_inds, feat])
            smoothed_spikes_resh[nan_inds, feat] = spikes_resh[nan_inds, feat]
        else:
            smoothed_spikes_resh[:, feat] = filter.apply(spikes_resh[:, feat])
    smoothed_spikes_resh = convert_to_tensor(smoothed_spikes_resh.copy())
    if len(spikes.shape) == 3:
        smoothed_spikes = smoothed_spikes_resh.reshape(num_trials, num_steps, num_feat)
    else:
        smoothed_spikes = smoothed_spikes_resh
    return smoothed_spikes


def load_from_file(
    data_path: Path,
):
    data = torch.load(data_path)
    spikes = data['spike']  # Shape: (num_time_bins, num_neurons)
    pos = data['pos']   # Shape: (num_time_bins, 2) for x and y positions
    vel = data['vel']   # Shape: (num_time_bins, 2) for x and y velocities
    smoothed_spikes_source = smooth_spikes(
        spikes,
        sigma=30,
        delta_ms=10,
        switch_causal=False,
        ignore_nans=True
    )
    # behavior
    z = torch.cat([pos, vel], dim=1)[:-1].numpy()
    y = smoothed_spikes_source[:-1].numpy()

    return y, z