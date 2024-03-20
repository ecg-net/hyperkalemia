# +
import base64
import struct

import numpy as np
import xmltodict
from scipy.ndimage import median_filter as scipy_ndimage_median_filter
import matplotlib.pyplot as plt
import pywt
# -

lead_order = [
    "I",
    "II",
    "III",
    "aVR",
    "aVL",
    "aVF",
    "V1",
    "V2",
    "V3",
    "V4",
    "V5",
    "V6",
]


def plot_12_lead_ecg(ecg_array, output_filename=None):
    """
    Plot each lead of the 12-lead ECG, and save the plot to a file.
    All leads share the x axis, but each lead gets its own chart.
    """
    fig, axs = plt.subplots(12, 1, sharex=True, figsize=(16, 9))
    for lead, lead_name in enumerate(lead_order):
        axs[lead].plot(ecg_array[:, lead])
        axs[lead].set_ylabel(str(lead_name))
    if output_filename is not None:
        plt.savefig(output_filename)
    plt.show()
    plt.close()


def get_median_filter_width(sampling_frequency, duration):
    res = int(sampling_frequency * duration)
    res += (res % 2) - 1  # needs to be an odd number
    return res


def remove_baseline_wander(waveform: np.ndarray, sampling_frequency: int) -> np.ndarray:

    """
    Remove baseline wander from ECG NPYs
    de Chazal et al. used two median filters to remove baseline wander.
    Median filters take the median value of a sliding window of a specified size
    One median filter of 200-ms width to remove QRS complexes and P-waves and other of
    600 ms width to remove T-waves.
    Do one filter and then the next filter. Then take the result and subtract it form the original signal
    https://pubmed.ncbi.nlm.nih.gov/15248536/
    Example of median filter:
    medfilt([2,6,5,4,0,3,5,7,9,2,0,1], 5) -> [ 2. 4. 4. 4. 4. 4. 5. 5. 5. 2. 1. 0.]
    >>> np.median([0, 0, 2, 6, 5])
    2.0
    >>> np.median([0, 2, 6, 5, 4])
    4.0

    """

    # Depending on the sampling frequency, the widths of the convolutional median filters changes
    filter_widths = [
        get_median_filter_width(sampling_frequency, duration) for duration in [0.2, 0.6]
    ]
    filter_widths = np.array(filter_widths, dtype="int")

    # make a copy of orignal signal
    original_waveform = waveform.copy()

    # apply median filters one by one on top of each other
    for filter_width in filter_widths:
        waveform = scipy_ndimage_median_filter(
            waveform, size=(filter_width, 1), mode="constant", cval=0.0
        )
    waveform = original_waveform - waveform  # finally subtract from orignal signal
    return waveform


def wavelet_denoise_signal(
    waveform: np.ndarray,
    dwt_transform: str = "bior4.4",
    dlevels: int = 9,
    cutoff_low: int = 1,
    cutoff_high: int = 7,
) -> np.ndarray:

    # cutoff_low determines how flat you want overall baseline to be.
    #   Higher means more flat baseline
    # cutoff_high determines within the small segments how much do
    #   you want to suppress the squiggliness. Lower cutoff_high
    #   suppresses more squiggliness but also suppresses R wave morphology

    coefficients = pywt.wavedec(
        waveform, dwt_transform, level=dlevels
    )  # wavelet transform 'bior4.4'
    # scale 0 to cutoff_low
    for low_cutoff_value in range(0, cutoff_low):
        coefficients[low_cutoff_value] = np.multiply(
            coefficients[low_cutoff_value], [0.0]
        )
    # scale cutoff_high to end
    for high_cutoff_value in range(cutoff_high, len(coefficients)):
        coefficients[high_cutoff_value] = np.multiply(
            coefficients[high_cutoff_value], [0.0]
        )
    waveform = pywt.waverec(coefficients, dwt_transform)  # inverse wavelet transform
    return waveform
