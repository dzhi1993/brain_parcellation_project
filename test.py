# # from nibabel.gifti import giftiio
# # import nibabel as nb
# # import numpy as np
# #
# # gifti_image = nb.load("Cond01_No-Go.func.gii")
# # # index = nifti_image.get_fdata()
# #
# # # print(index.shape)
# # img_data = [x.data for x in gifti_image.darrays]
# # #np_data = np.reshape(img_data, (len(img_data[0]),))
# # np_data = np.reshape(img_data, (len(img_data[0]), len(img_data[0][0])))
# # print(np_data, np_data.shape)
#
#
# import csv
#
# txt_file = "Mapping.txt"
# csv_file = "mycsv.csv"
#
#
# # with open(txt_file, 'r') as infile, open(csv_file, 'w') as outfile:
# #      stripped = (line.strip() for line in infile)
# #      lines = (line.split(",") for line in stripped if line)
# #      writer = csv.writer(outfile)
# #      writer.writerows(lines)
#
# #import os
# import pandas as pd
#
# df = pd.read_csv(txt_file, sep=",")
# df.to_csv(csv_file, index=False)

import time
import matplotlib.pyplot as plt
import numpy as np

from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import kneighbors_graph

# Generate sample data
n_samples = 1500
np.random.seed(0)
t = 1.5 * np.pi * (1 + 3 * np.random.rand(1, n_samples))
x = t * np.cos(t)
y = t * np.sin(t)


X = np.concatenate((x, y))
X += .7 * np.random.randn(2, n_samples)
X = X.T

# Create a graph capturing local connectivity. Larger number of neighbors
# will give more homogeneous clusters to the cost of computation
# time. A very large number of neighbors gives more evenly distributed
# cluster sizes, but may not impose the local manifold structure of
# the data
knn_graph = kneighbors_graph(X, 30, include_self=False)

for connectivity in (None, knn_graph):
    for n_clusters in (30, 3):
        plt.figure(figsize=(10, 4))
        for index, linkage in enumerate(('average',
                                         'complete',
                                         'ward',
                                         'single')):
            plt.subplot(1, 4, index + 1)
            model = AgglomerativeClustering(linkage=linkage,
                                            connectivity=connectivity,
                                            n_clusters=n_clusters)
            t0 = time.time()
            model.fit(X)
            elapsed_time = time.time() - t0
            plt.scatter(X[:, 0], X[:, 1], c=model.labels_,
                        cmap=plt.cm.nipy_spectral)
            plt.title('linkage=%s\n(time %.2fs)' % (linkage, elapsed_time),
                      fontdict=dict(verticalalignment='top'))
            plt.axis('equal')
            plt.axis('off')

            plt.subplots_adjust(bottom=0, top=.89, wspace=0,
                                left=0, right=1)
            plt.suptitle('n_cluster=%i, connectivity=%r' %
                         (n_clusters, connectivity is not None), size=17)


plt.show()
