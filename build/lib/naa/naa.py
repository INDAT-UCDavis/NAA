import numpy as np
import scipy as sp
from scipy.signal import find_peaks
from scipy.stats import truncnorm
from scipy.optimize import curve_fit
from scipy.special import erfc
from matplotlib import pyplot
from matplotlib import pyplot as plt
from typing import Callable
import csv

def calibration_2022(channel):
    return -0.267 +0.500*channel + -3.36E-08*pow(channel, 2) + 0.00E+00*pow(channel, 3)


def find_candidate_isotopes(
    energies,               # calibrated energies in keV
    counts,                 # associated counts for each keV bin
    tolerance: float=5.0    # amount of coverage for a peak in keV
):
    candidates = {}
    
def truncated_normal(
    x, A, mu, sigma, a, b
):
    return A * np.exp(-0.5 * np.power(((x - mu)/sigma), 2)) + a + b * erfc((x - mu)/sigma)

def search_bins(x, value):
    for ii in range(len(x) - 1):
        if value >= x[ii] and value < x[ii+1]:
            return ii
    if value == x[-1]:
        return len(x) - 1
    return -1

def load_spectrum(
    input_file: str,
    calibration:    Callable=calibration_2022
):
    counts = []
    with open(input_file, "r") as file:
        reader = csv.reader(file, delimiter=",")
        for row in reader:
            counts.append(float(row[0]))
    counts = np.array(counts)
    energies = np.array([calibration(ii) for ii in range(len(counts))])
    return energies, counts

def fit_gaus(
    energies,
    counts,
    low:    float,
    center: float,
    high:   float,
    scale:  float=5,
):
    if (center < low) or (high < low) or (high < center):
        print("Fit parameters are not in order!")
        return
    mask = (energies >= low) & (energies <= high)

    fit_energies = energies[mask]
    fit_counts = counts[mask]

    low = fit_energies[0]
    high = fit_energies[-1]

    center_bin = search_bins(fit_energies, center)
    low_bin = search_bins(fit_energies, low)
    high_bin = search_bins(fit_energies, high)

    if center_bin == -1:
        print("Center value not in the range of the histogram!")
        return
    if low_bin == -1:
        print("Low value not in the range of the histogram!")
        return
    if high_bin == -1:
        print("High value not in the range of the histogram!")
        return
    
    A_init = fit_counts[center_bin] * 2
    A_low = fit_counts[center_bin]
    A_high = A_init * 5

    mu_init = center
    mu_low = center - 10
    mu_high = center + 10

    sigma_init = scale
    sigma_low = 0.01
    sigma_high = 20

    a_init = fit_counts[low_bin]
    a_low = a_init / 2
    a_high = a_init * 3

    b_init = fit_counts[center_bin] / 2
    b_low = 0
    b_high = fit_counts[center_bin] * 5

    popt, pcov = curve_fit(
        truncated_normal, 
        fit_energies, fit_counts,
        p0=[A_init, mu_init, sigma_init, a_init, b_init],
        bounds=(
            [A_low, mu_low, sigma_low, a_low, b_low], 
            [A_high, mu_high, sigma_high, a_high, b_high]
        )
    )
    integral = 2 * popt[0] * popt[2] * np.sqrt(np.pi)
    fit_counts = truncated_normal(fit_energies, popt[0], popt[1], popt[2], popt[3], popt[4])
    return popt, integral, fit_energies, fit_counts

def fit_spectrum(
    energies, 
    counts,
    tolerance:  float=5.0
):
    peaks, _ = find_peaks(counts, threshold=10)
    peak_counts = counts[peaks]
    peak_energies = energies[peaks]
    candidates = find_candidate_isotopes(
        peak_energies, peak_counts, tolerance
    )

def plot_spectrum(
    energies, 
    counts,
    peak_energies=[],
    peak_counts=[],
    integral=0.0
):
    fig, axs = plt.subplots(figsize=(10,6))
    axs.plot(energies, counts, linestyle='--', color='k', label='spectrum')
    if len(peak_energies) != 0:
        axs.scatter(peak_energies, peak_counts, color='r', label=f'Integral = {integral}')
    axs.set_yscale("log")
    axs.set_xlabel("Energy [keV]")
    axs.set_ylabel("Counts [n]")
    plt.legend()
    plt.tight_layout()
    plt.show()

# from pyne import data
# import math
# import numpy as np
# from matplotlib import pyplot as plt

# def get_gammas(Z, A):
#     child_string = str(Z + 1)
#     isotope_string = str(Z)
#     if A < 100:
#         isotope_string += '0'
#         child_string += '0'
#     isotope_string += str(A)
#     child_string += str(A)
#     isotope_string += '0'
#     isotope_string += '0'
#     isotope_string += '0'
#     isotope_string += '0'
#     child_string += '0'
#     child_string += '0'
#     child_string += '0'
#     child_string += '0'

#     intensities = data.gamma_photon_intensity(int(isotope_string))
#     decay_pairs = data.gamma_from_to_byparent(int(isotope_string))
#     energies = data.gamma_energy(int(isotope_string))

#     photonbr, photonbr_error = data.decay_photon_branch_ratio(int(isotope_string), int(child_string))
#     final_energies = []
#     final_intensities = []
#     #print 
#     for ii, item in enumerate(intensities):
#         if math.isnan(item[0]):
#             continue
#         # compute the intensities by multiplying the branch ratio and the relative intensity; ignore the errors 
#         final_energies.append(energies[ii][0])
#         final_intensities.append(photonbr*item[0])
#     final_energies = np.array(final_energies)
#     final_intensities = np.array(final_intensities)
#     final_intensities /= sum(final_intensities)
#     return final_energies, final_intensities

# final_energies, final_intensities = get_gammas(57, 140)

# fig, axs = plt.subplots()
# axs.scatter(final_energies, final_intensities)
# axs.set_xlabel("Gamma energy [keV]")
# axs.set_ylabel("Relative intensity")
# plt.show()
