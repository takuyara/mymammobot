import torch
import matplotlib.pyplot as plt
K = 12.5
"""
x = torch.arange(0, 1, 0.001)
y = torch.sigmoid((x - 0.4) * K)
"""
x = torch.arange(-1, 1, 0.0001)
y = torch.sigmoid(x * K)
plt.plot(x, y)
ax = plt.gca()
ax.spines["right"].set_color("none")
ax.spines["top"].set_color("none")
ax.spines["left"].set_position(("data", 0))
plt.xlabel("Depth Value Difference to ABT", fontsize = 15)
plt.ylabel("Bifurcation Mask Value", fontsize = 15)
plt.show()
