import numpy as np
import scipy as sp
from scipy.signal import find_peaks
from matplotlib import pyplot
from typing import Callable
import csv
import curie as cu

def calibration_2022(channel):
    return -0.267 +0.500*channel + -3.36E-08*pow(channel, 2) + 0.00E+00*pow(channel, 3)


def find_candidate_isotopes(
    energies,           # calibrated energies in keV
    counts,             # associated counts for each keV bin
    tolerance: float=5.0# amount of coverage for a peak in keV
):
    candidates = {}
    


def fit_spectrum(
    input_file: str, 
    calibration: Callable=calibration_2022,
    tolerance:  float=5.0
):
    bins = []
    with open(input_file, "r") as file:
        reader = csv.reader(file, delimiter=",")
        for row in reader:
            bins.append(float(row[0]))
    bins = np.array(bins)
    energies = np.array([calibration(ii) for ii in range(len(bins))])
    peaks, _ = find_peaks(bins, threshold=10)
    peak_bins = bins[peaks]
    peak_energies = energies[peaks]
    candidates = find_candidate_isotopes(
        peak_energies, peak_bins, tolerance
    )
