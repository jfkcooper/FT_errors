""" Careful version """

import numpy as np
import matplotlib.pyplot as plt
from hogben.simulate import SimulateReflectivity
from refnx.reflect import SLD, ReflectModel


air = SLD(0, name='Air')
layer1 = SLD(8, name='Layer 1')(thick=100, rough=2)
substrate = SLD(2.047, name='Substrate')(thick=0, rough=2)

sample_1 = air | layer1 | substrate
model_1 = ReflectModel(sample_1)

angle_times = [(0.7, 100, 50),
               (2.0, 100, 200),
               (5.0, 100, 1000)] # (Angle, Points, Time) - lets be extreme here

# Simulate the experiment.
data_1 = SimulateReflectivity(model_1, angle_times).simulate()
length = len(data_1[0])
#  q, r, dr, counts
#  use q as x, r as data, and dr**2 as variance
#  Need to make something like xi for the FT data
############################
############################

x = np.linspace(0, 1, length)
n_xi = 1000
xi = np.linspace(0.0, 200.0, n_xi)

# Data with frequencies of 4 and 10 inverse x periods
#data = np.sin(2*np.pi*4*x) + np.sin(2*np.pi*10*x)
data = data_1[1]

# Noise amount, proportional to each component differently
# 5x the noise on f=10 component
#variance = np.sin(2*np.pi*4*x)**2 + 25*np.sin(2*np.pi*10*x)**2
variance = data_1[2]**2

covariance = np.diag(variance)

#
# Main calculation of fourier transform with errors
#

x_xi = 2 * np.pi * x.reshape(1, -1) * xi.reshape(-1, 1)

ft_matrix = np.concatenate((np.cos(x_xi), np.sin(x_xi)))
ft_data = ft_matrix @ data
ft_covariance = ft_matrix @ covariance @ ft_matrix.T

plt.figure("Covariance between real and imaginary components - not negligible")
plt.imshow(ft_covariance)


ft_power = ft_data[:n_xi]**2 + ft_data[n_xi:]**2

# Dubious assumption, we can treat real an imaginary parts separately

sq_jac = 2*np.diag(ft_data)
ft_sq_covariance = sq_jac @ ft_covariance @ sq_jac.T

plt.figure("Square covariance")
plt.imshow(ft_sq_covariance)

ft_power_covariance = ft_sq_covariance[:n_xi, :n_xi] + \
                      ft_sq_covariance[n_xi:, n_xi:] + \
                      ft_sq_covariance[:n_xi, n_xi:] + \
                      ft_sq_covariance[n_xi:, :n_xi]

plt.figure("Power covariance")
plt.imshow(ft_power_covariance)


ft_power_variance = np.diag(ft_power_covariance) # clearly dodgy

#
# Plots
#

plt.figure("Detailed plots")

plt.subplot(2,2,1)
plt.plot(x, data)
plt.yscale('log')
plt.fill_between(x, data - np.sqrt(variance), data+np.sqrt(variance), alpha=0.2)
# plt.errorbar(x, data, yerr=np.sqrt(variance))
plt.title("Input data")

plt.subplot(2,2,3)
plt.plot(x, variance)
plt.title("Input variance")

plt.subplot(2,2,2)
plt.plot(xi, ft_power)
plt.fill_between(xi,
                 ft_power - np.sqrt(ft_power_variance),
                 ft_power + np.sqrt(ft_power_variance),
                 alpha=0.6)
plt.title("FT power - much windowing")


plt.subplot(2,2,4)
plt.plot(xi, ft_power_variance)
plt.title("Input variance")


plt.figure("Just main plots")

plt.subplot(2,1,1)
plt.plot(x, data)
plt.yscale('log')
plt.fill_between(x, data - np.sqrt(variance), data+np.sqrt(variance), alpha=0.2)
plt.title("Input data")


plt.subplot(2,1,2)
plt.plot(xi, ft_power)
plt.yscale('log')
plt.fill_between(xi,
                 ft_power - np.sqrt(ft_power_variance),
                 ft_power + np.sqrt(ft_power_variance),
                 alpha=0.6)
plt.title("FT power")



plt.show()