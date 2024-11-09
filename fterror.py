import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 1, 1000)
xi = np.linspace(0.0, 20.0, 1000)

# Data with frequencies of 4 and 10 inverse x periods
data = np.sin(2*np.pi*4*x) + np.sin(2*np.pi*10*x)

# Noise amount, proportional to each component differently
# 5x the noise on f=10 component
variance = np.sin(2*np.pi*4*x)**2 + 25*np.sin(2*np.pi*10*x)**2
covariance = np.diag(variance)

#
# Main calculation of fourier transform with errors
#

ft_matrix = np.exp(-2j*np.pi * x.reshape(1, -1) * xi.reshape(-1, 1))
ft_data = np.dot(ft_matrix, data)
ft_covariance = np.dot(np.dot(ft_matrix, covariance), ft_matrix)


#
# End of actual calculation of fourier transform error stuff, the rest
# is just about power and plotting, a bit more fiddly
#

# Didn't write this down, but we're going plot the power spectrum here,
# so there is an extra step to propagate from f(xi) -> |f(xi)|^2 = y
# We need to be a bit careful here, because this function is not actually
# analytic. Bodge it until checked.
#

ft_power = ft_data.real**2 + ft_data.imag**2

# Dubious assumption, we can treat real an imaginary parts separately

sq_jac_real = 2*np.diag(ft_data.real)
sq_jac_imag = 2*np.diag(ft_data.imag)

ft_sq_covariance_real = np.dot(np.dot(sq_jac_real, ft_covariance.real), sq_jac_real)
ft_sq_covariance_imag = np.dot(np.dot(sq_jac_imag, ft_covariance.imag), sq_jac_imag)

ft_power_covariance = ft_sq_covariance_real + ft_sq_covariance_imag

ft_power_variance = np.diag(np.abs(ft_power_covariance)) # clearly dodgy

#
# Plots
#

plt.figure("Detailed plots")

plt.subplot(2,2,1)
plt.plot(x, data)
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
plt.fill_between(x, data - np.sqrt(variance), data+np.sqrt(variance), alpha=0.2)
plt.title("Input data")


plt.subplot(2,1,2)
plt.plot(xi, ft_power)
plt.fill_between(xi,
                 ft_power - np.sqrt(ft_power_variance),
                 ft_power + np.sqrt(ft_power_variance),
                 alpha=0.6)
plt.title("FT power")



plt.show()