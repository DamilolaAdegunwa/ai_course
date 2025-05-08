import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

x = np.linspace(-5, 5, 50)
y = np.linspace(-5, 5, 50)
X, Y = np.meshgrid(x, y)
Z = np.sin(np.sqrt(X**2 + Y**2))

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')  # None, 'aitoff', 'hammer', 'lambert', 'mollweide', 'polar', 'rectilinear', str
ax.plot_surface(X, Y, Z, cmap='plasma')
ax.set_proj_type('persp', focal_length=0.2)
plt.show()
