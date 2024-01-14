import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.fft import fft



np.random.seed(0)
dates = pd.date_range(start='2000-01-01', end='2001-01-01', freq='D')
data = np.random.normal(size=len(dates))                                    # daily data

fft_results = fft(data)                                                     # Fourier Transform
power = np.abs(fft_results)
frequencies = np.fft.fftfreq(len(dates), d=(dates[1] - dates[0]).days)
positive_freqs = frequencies > 0
# mjo_range = (1/60 <= frequencies) & (frequencies <= 1/30)                   # MJO range (30 to 60 days)
mjo_range = (1/360 <= frequencies) & (frequencies <= 1/10)                   # MJO range (30 to 60 days)
filtered_power = power[positive_freqs & mjo_range]
filtered_frequencies = frequencies[positive_freqs & mjo_range]
periods_in_days = 1 / filtered_frequencies

plt.figure(figsize=(10, 6))
plt.plot(periods_in_days, filtered_power)                              # Plotting the power spectrum in the MJO frequency range
plt.title('Power Spectrum in MJO Frequency Range (30 to 60 days)')
plt.xlabel('Frequency days')
plt.ylabel('Power')
plt.grid(True)
plt.show()


np.random.seed(1)
dates = pd.date_range(start='2000-01-01', end='2001-01-01', freq='D')
data = np.random.normal(size=len(dates))                                    # daily data

fft_results = fft(data)                                                     # Fourier Transform
power = np.abs(fft_results)
frequencies = np.fft.fftfreq(len(dates), d=(dates[1] - dates[0]).days)
positive_freqs = frequencies > 0
# mjo_range = (1/60 <= frequencies) & (frequencies <= 1/30)                   # MJO range (30 to 60 days)
mjo_range = (1/360 <= frequencies) & (frequencies <= 1/10)                   # MJO range (30 to 60 days)
filtered_power = power[positive_freqs & mjo_range]
filtered_frequencies = frequencies[positive_freqs & mjo_range]
periods_in_days = 1 / filtered_frequencies

plt.figure(figsize=(10, 6))
plt.plot(periods_in_days, filtered_power)                              # Plotting the power spectrum in the MJO frequency range
plt.title('Power Spectrum in MJO Frequency Range (30 to 60 days)')
plt.xlabel('Frequency days')
plt.ylabel('Power')
plt.grid(True)
plt.show()








































