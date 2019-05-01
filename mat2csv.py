import scipy.io as spio
import numpy as np
import pandas as pd

# Making the look-up table
# data = pd.read_csv("flatmap_vertices.csv", header=None)
#
# data = data.values
# data = data/100
#
# for i in range(len(data)):
#     data[i][2] = i + 1
#
# print(data.shape)
#
# stride = 2 / 800
# mapping = np.zeros(shape=(800, 800))
#
# for i in range(800):
#     for j in range(800):
#         for point in data:
#             if point[0] >= (-1 + j*stride) and point[0] < (-1 + (j+1)*stride) and point[1] > (1 - (i+1)*stride) and point[1] <= (1 - i*stride):
#                 mapping[i][j] = point[2]
#                 print(point[2])
#                 break
#             elif point[2] == 28935:
#                 print("[" + str(i) + ", " + str(j) + "] " + "Not found!")
#
# print(mapping)
#
# # for i in range(len(mapping)):
# np.savetxt(("mapping_800.csv"), mapping, delimiter=',', newline='\n')


# Filling the 0 values in the table if surrounded by non-zero values
mapping = pd.read_csv("mapping_250_refine.csv", header=None)

for i in range(15, len(mapping) - 15):
    for j in range(15, len(mapping) - 15):
        if mapping[i][j] == 0:
            count = 0
            if mapping[i-1][j] != 0:
                count += 1
            if mapping[i+1][j] != 0:
                count += 1
            if mapping[i][j-1] != 0:
                count += 1
            if mapping[i][j+1] != 0:
                count += 1

            if count >= 3:
                if mapping[i][j-1] != 0:
                    mapping[i][j] = mapping[i][j-1]
                else:
                    mapping[i][j] = mapping[i+1][j]

print(mapping)
np.savetxt(("mapping_250_rerefine.csv"), mapping, delimiter=',', newline='\n')