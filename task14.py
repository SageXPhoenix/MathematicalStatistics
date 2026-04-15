import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

alpha = 0.05
q = norm.ppf(1 - alpha)
sigma = np.sqrt(7/6)

def power(theta, q, sigma):
    return 1 - norm.cdf(q - theta / sigma)

theta = np.linspace(-2, 6, 500)
y = power(theta, q, sigma)

plt.figure(figsize=(8, 5))
plt.plot(theta, y, 'g-', linewidth=2)

plt.axhline(y=alpha, color='red', linestyle='--', alpha=0.5)
plt.axhline(y=1.0, color='blue', linestyle='--', alpha=0.3)
plt.axhline(y=0.5, color='blue', linestyle='--', alpha=0.3)

plt.xlabel('θ')
plt.ylabel('W(θ)')
plt.grid(True, alpha=0.3)
plt.ylim(-0.05, 1.05)
plt.xlim(-2, 6)

plt.tight_layout()
plt.savefig('PowerPlot14Task.png', dpi=300)
plt.show()