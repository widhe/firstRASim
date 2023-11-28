import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, filtfilt

ts = 1e-9
fs = 1e9
sig_len = 200000

V = 1
f = 2e9
tau_g = 0.25*1/f
tau_g = 0*1/f

#t = np.arange(0, 1000e-9, 1e-9)
t = np.linspace(0, sig_len*ts, sig_len)

v1 = V*np.cos(2*np.pi*f*(t-tau_g))
v2 = V*np.cos(2*np.pi*f*t)
vo = v1 * v2

# Define the cutoff frequency (in Hz)
cutoff_frequency = 1e3  # For example, 500 MHz

# Order of the filter
order = 2

# Get the filter coefficients
b, a = butter(order, cutoff_frequency, btype='low', analog=False, fs=fs)

r = filtfilt(b, a, vo)

x_plts = 1
y_plts = 4
sb_plt = 1

plt.subplot(y_plts, x_plts, sb_plt)
sb_plt += 1
plt.plot(t, v1)

plt.subplot(y_plts, x_plts, sb_plt)
sb_plt += 1
plt.plot(t, v2)

plt.subplot(y_plts, x_plts, sb_plt)
sb_plt += 1
plt.plot(t, vo)

plt.subplot(y_plts, x_plts, sb_plt)
sb_plt += 1
plt.plot(t, r)

plt.tight_layout()
plt.show()