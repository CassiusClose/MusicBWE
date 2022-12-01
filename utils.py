import torch
import matplotlib.pyplot as plt

def plot_spect(sig, fs, title=None, show=True, fig_ind=0):
    if len(sig.shape) > 1:
        sig = sig[0]

    n_win = 256
    n_hop = 128
    pad = 1
    win = torch.hann_window(n_win)
    stft = torch.stft(sig, n_fft=n_win*pad, hop_length=n_hop, win_length=n_win, window=win,
            center=False, return_complex=True).abs().clamp(1e-8).log10()


    # The max values of the x & y axes.
    # FFT measures frequency from 0 to fs/2
    # Calculate the max time based on the number of frames in the STFT. The stft() function
    # cuts off the last little bit of audio if it doesn't fit in frame, so can't just use the
    # length of the original audio signal.
    max_freq = fs/2
    max_time = (((stft.shape[1]-1)*n_hop + n_win) / fs)

    # Calculate the step so there are the proper number of points
    freq_step = max_freq / stft.shape[0]
    time_step = max_time / stft.shape[1]

    # Axis arrays
    freqs = torch.arange(0, max_freq, freq_step)
    times = torch.arange(0, max_time, time_step)

    plt.figure(fig_ind)
    plt.pcolormesh(times, freqs, stft)
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    if title:
        plt.title(title)

    plt.tight_layout()

    if(show):
        plt.show()

    return (freqs, times)
    


def plot_freqz(sos, fs, cutoff_freq, fig_ind=0):
    (w,h) = signal.sosfreqz(sos, fs=fs)
    h = torch.Tensor(np.abs(h)).log10()
    plt.figure(fig_ind)
    plt.plot(w, h)
    plt.axvline(cutoff_freq, color='r')
