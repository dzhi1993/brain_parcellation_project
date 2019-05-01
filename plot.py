import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# x = np.arange(0, 50, 50/1001)
# print(x)
# y_axis = pd.read_csv("square_average_list.csv", sep='\n', header=None)
# print(y_axis.shape, y_axis.describe())
#
# fig = plt.figure()
# #plt.xticks(x)
# plt.tick_params(
#     axis='x',          # changes apply to the x-axis
#     which='both',      # both major and minor ticks are affected
#     bottom=False,      # ticks along the bottom edge are off
#     top=False,         # ticks along the top edge are off
#     labelbottom=False)
# plt.title("image subtraction", fontweight="bold")
# plt.xlabel('Sigma: 0 to 50, interval 0.05')
# plt.ylabel('averaged subtraction square')
# plt.plot(x, y_axis, color='r')
# plt.show()

import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv("xyz_polt.csv", sep=',', header=None)
print(data, data.shape)

x_3d = data[[0]]
y_3d = data[[1]]
z_3d = data[[2]]
print(x_3d)
x_3d = np.asarray(x_3d)
y_3d = np.asarray(y_3d)
z_3d = np.asarray(z_3d)
np.reshape(x_3d, (86, ))
np.reshape(y_3d, (86, ))
np.reshape(z_3d, (86, ))
print(x_3d.shape)

mpl.rcParams['legend.fontsize'] = 10

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.plot_wireframe(x_3d, y_3d, z_3d)
#ax.plot(x_3d, y_3d, z_3d, label='parametric curve')
#ax.legend()

plt.show()
