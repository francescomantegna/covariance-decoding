
from mpl_toolkits.mplot3d import proj3d
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt
import numpy as np
plt.ion()

def f(x1, x2):
    return 0.1 * x1 + 0.1 * x2 - 0.5 * x1 * x1 - 0.1 * x1 * x2 - 1.0 * x2 * x2

x = np.linspace(-3, 3, 100)
y = np.linspace(-3, 3, 100)
xx, yy = np.meshgrid(x,y)
z = f(xx, yy)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.set_xlim(-3, 3)
ax.set_ylim(3, -3)
ax.set_zlim(-15,5)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")

ax.plot_surface(xx, yy, z, cmap="pink", alpha = 0.5)
ax.plot_surface(xx, yy, np.zeros_like(z), cmap="gray", alpha = 0.2)

ax.scatter(0,0,0,color="k",s=20)
ax.scatter(2,2,-6,color="k",s=20)
ax.scatter(2,2,0,color="k",s=20)

ax.text(3, 1.5, 8, "Pi",fontsize=50, fontfamily='cursive') 
ax.text(4.4, 4.6, 3, "Ei",fontsize=50, fontfamily='cursive')
ax.text(-9, 2, -18, r'$\delta$(Pi, Ei)', color='#a6611a', fontsize=50, fontfamily='cursive')
ax.text(4.4, 4.4, -3, "Ri",fontsize=50, fontfamily='cursive')
ax.text(2.4, 1.5, -2, r'$\Gamma$(Pi, Ri)', color='grey', fontsize=50, fontfamily='cursive')
ax.text(2, -0.5, -10.5, "M",fontsize=50, fontfamily='cursive')
ax.text(-3, -2, 4, "TxM",fontsize=50, fontfamily='cursive')

c1 = np.linspace(0, 2, 100)
ax.plot(c1, c1, f(0, 0), lw=2, linestyle='-', color='#a6611a', alpha=1)
ax.plot(c1, c1, f(c1, c1), lw=2, linestyle='--', color='grey', alpha=1)
ax.axis('off')
ax.view_init(azim=-28, elev=26)

plt.tight_layout(pad=-3)

#########################################################################

x = np.linspace(-3, 3, 100)
y = np.linspace(-3, 3, 100)
xx, yy = np.meshgrid(x,y)
z = f(xx, yy)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.set_xlim(-3, 3)
ax.set_ylim(3, -3)
ax.set_zlim(-15,5)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")

ax.plot_surface(xx, yy, z, cmap="pink", alpha = 0.5)
ax.plot_surface(xx, yy, np.zeros_like(z), cmap="gray", alpha = 0.2)

ax.scatter(0,0,0,color="k",s=20)
ax.scatter(-2,2,-5,color="k",s=20)
ax.scatter(-2,2,0,color="k",s=20)

ax.text(4.5, 2, 9, "Pi",fontsize=50, fontfamily='cursive') 
ax.text(-3.5,3.2, -1, "Ei",fontsize=50, fontfamily='cursive')
ax.text(-4, 1.2, -1.5, r'$\delta$(Pi, Ei)', color='#018571', fontsize=50, fontfamily='cursive')
ax.text(-4.2, 3, -11, "Ri",fontsize=50, fontfamily='cursive')
ax.text(-1.5, 1, -8, r'$\Gamma$(Pi, Ri)', color='grey', fontsize=50, fontfamily='cursive')
ax.text(2, -0.5, -10.5, "M",fontsize=50, fontfamily='cursive')
ax.text(-3, -2, 4, "TxM",fontsize=50, fontfamily='cursive')

c2 = np.linspace(0, -2, 100)
ax.plot(c2, -c2, f(0, 0), lw=2, linestyle='-', color='#018571', alpha=1)
ax.plot(c2, -c2, f(c2, -c2), lw=2, linestyle='--', color='grey', alpha=1)
ax.axis('off')
ax.view_init(azim=-28, elev=26)

plt.tight_layout(pad=-3)

#########################################################################

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.set_xlim(-3, 3)
ax.set_ylim(3, -3)
ax.set_zlim(-15,5)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")

ax.plot_surface(xx, yy, np.zeros_like(z), cmap="gray", alpha = 0.2)

ax.scatter(0,0,0,color="k",s=20)
ax.scatter(2,-2,0,color="k",s=20)

ax.text(3, 1.5, 8, "Pi",fontsize=50, fontfamily='cursive') 
ax.text(2.1, -2, 1.5, "Ei",fontsize=50, fontfamily='cursive')
ax.text(3.3, 2.5, 2, r'$\delta$(Pi, Ei)', color='#018571', fontsize=50, fontfamily='cursive')

c2 = np.linspace(0, 2, 100)
ax.plot(c2, -c2, f(0, 0), lw=2, linestyle='-', color='#018571', alpha=1)

ax.axis('off')
ax.view_init(azim=-28, elev=26)

plt.tight_layout(pad=-3)

#########################################################################

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.set_xlim(-3, 3)
ax.set_ylim(3, -3)
ax.set_zlim(-15,5)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")

ax.plot_surface(xx, yy, np.zeros_like(z), cmap="gray", alpha = 0.2)

ax.scatter(0,0,0,color="k",s=20)
ax.scatter(2.5,-1,0,color="k",s=20)

ax.text(2.4, 2.5, 8, "Pi",fontsize=50, fontfamily='cursive') 
ax.text(2.5, -1, 1.0, "Ei",fontsize=50, fontfamily='cursive')
ax.text(1.8, 2.5, -3, r'$\delta$(Pi, Ei)', color='#a6611a', fontsize=50, fontfamily='cursive')

c2 = np.linspace(0, 2.5, 100)
c3 = np.linspace(0, 1, 100)
ax.plot(c2, -c3, f(0, 0), lw=2, linestyle='-', color='#a6611a', alpha=1)

ax.axis('off')
ax.view_init(azim=-28, elev=26)

plt.tight_layout(pad=-3)
