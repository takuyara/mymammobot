import torch
import matplotlib.pyplot as plt
K = 12.5
x = torch.arange(0, 1, 0.001)
y = torch.sigmoid((x - 0.4) * K)
plt.plot(x, y)
plt.show()
