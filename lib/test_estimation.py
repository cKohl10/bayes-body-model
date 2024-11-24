import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

# Prior mean
mu_prior = np.array([1.0, 2.0])

# Prior covariance matrix
Sigma_prior = np.array([
    [0.5, -0.1],
    [0.1, 0.5]
])

# Posterior mean
mu_posterior = np.array([1.5, 2.5])

# Posterior covariance matrix (typically smaller if data provides information)
Sigma_posterior = np.array([
    [0.3, 1.0],
    [0.05, 0.3]
])

def plot_covariance_ellipse(mean, cov, ax, n_std=1.0, **kwargs):
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = eigvals.argsort()[::-1]
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]
    angle = np.degrees(np.arctan2(*eigvecs[:,0][::-1]))
    width, height = 2 * n_std * np.sqrt(eigvals)
    ellipse = Ellipse(xy=mean, width=width, height=height, angle=angle, **kwargs)
    ax.add_patch(ellipse)

fig, ax = plt.subplots(figsize=(6,6))

# Plot prior
plot_covariance_ellipse(mu_prior, Sigma_prior, ax, edgecolor='blue', label='Prior')
ax.plot(*mu_prior, 'bo')

# Plot posterior
plot_covariance_ellipse(mu_posterior, Sigma_posterior, ax, edgecolor='red', label='Posterior')
ax.plot(*mu_posterior, 'ro')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.legend()
plt.title('Prior and Posterior Uncertainty Ellipses')
plt.grid(True)
plt.show()
