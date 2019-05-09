################################################################################################
# This file is used to clustering human cerebellum parcellation using DBSCAN (Density-Based)   #
# algorithm.                                                                                   #
# The output file is a .csv file that contains the clustering result for both individual data  #
# and group-average data.                                                                      #
################################################################################################
from sklearn import datasets
import matplotlib
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import scipy.io as spio
import numpy as np
from sklearn.metrics.pairwise import cosine_distances, cosine_similarity

# ---------------------------- SC 1 ----------------------------- #
mat = spio.loadmat('../groupData_sc1.mat')
groupData = mat['X_C']
print(groupData.transpose().shape)

affinity_matrix = cosine_similarity(groupData.transpose())
print(affinity_matrix.shape, affinity_matrix.min(), affinity_matrix.max())

# plt.imshow(affinity_matrix, aspect='auto')
# plt.show()

# # spio.savemat('clusters.mat', {"Y": clustering.labels_})
clustering = DBSCAN(algorithm='auto', leaf_size=30, metric='precomputed',
                    metric_params=None, min_samples=2, n_jobs=None, p=None).fit(affinity_matrix + 1)

# clustering = AgglomerativeClustering(n_clusters=10,
#                                      affinity='manhattan',
#                                      memory=None,
#                                      connectivity=None,
#                                      compute_full_tree='auto',
#                                      linkage='average',
#                                      pooling_func='deprecated').fit(groupData.transpose())

# print(clustering.labels_.shape, clustering.labels_)


# Make the clustering result bestG type
clustering_result = np.zeros((25275, 10))
for i in range(clustering_result.shape[0]):
    clustering_result[i, clustering.labels_[i]] = 0.4

spio.savemat('spec_sc1_bestG.mat', {"bestG": clustering_result})
# plt.imshow(clustering.affinity_matrix_, aspect='auto')
# plt.show()
# np.savetxt("spec_sc1_bestG_affinity_rbf_arpack.csv", clustering.affinity_matrix_, delimiter=',', newline=',')
# spio.savemat('spec_sc1_bestG_affinity_10nn.mat', {"affinityMatrix": clustering.affinity_matrix_})

# print(clustering, clustering.affinity_matrix_.shape, clustering.affinity_matrix_)
# print(X[:, 0], y.shape)
print(clustering)

# # Test different eigen_solver
# clustering = SpectralClustering(n_clusters=10,
#                                 eigen_solver='lobpcg',
#                                 affinity="precomputed",
#                                 n_jobs=-1).fit(affinity.affinity_matrix_ * 2)
#
# # Make the clustering result bestG type
# clustering_result = np.zeros((25275, 10))
# for i in range(clustering_result.shape[0]):
#     clustering_result[i, clustering.labels_[i]] = 0.4
#
# spio.savemat('spec_sc1_bestG.mat', {"bestG": clustering_result})
# plt.imshow(clustering.affinity_matrix_, aspect='auto')
# plt.show()
# np.savetxt("spec_sc1_bestG_affinity_rbf_lobpcg.csv", clustering.affinity_matrix_, delimiter=',', newline=',')
# # spio.savemat('spec_sc1_bestG_affinity.mat', {"affinityMatrix": clustering.affinity_matrix_})
#
# # print(clustering, clustering.affinity_matrix_.shape, clustering.affinity_matrix_)
# # print(X[:, 0], y.shape)
# print(clustering)

# ---------------------------- SC 2 ----------------------------- #
mat = spio.loadmat('../groupData_sc2.mat')
groupData = mat['X_C']

affinity_matrix = cosine_similarity(groupData.transpose())
print(affinity_matrix.shape, affinity_matrix.min(), affinity_matrix.max())

# clustering = SpectralClustering(n_clusters=10,
#                                 eigen_solver='arpack',
#                                 affinity="nearest_neighbors",
#                                 n_neighbors=10,
#                                 n_jobs=-1).fit(groupData.transpose() * 2)
# # print(affinity.labels_.shape, affinity.labels_)
# # # spio.savemat('clusters.mat', {"Y": clustering.labels_})

clustering = DBSCAN(algorithm='auto', eps=3, leaf_size=30, metric='precomputed',
                    metric_params=None, min_samples=2, n_jobs=None, p=None).fit(affinity_matrix + 1)

print(clustering.labels_.shape, clustering.labels_)

# # clustering = AgglomerativeClustering(n_clusters=10,
# #                                      affinity='euclidean',
# #                                      memory=None,
# #                                      connectivity=None,
# #                                      compute_full_tree='auto',
# #                                      linkage='ward',
# #                                      pooling_func='deprecated').fit(groupData.transpose())
#
# print(clustering.labels_.shape, clustering.labels_)
#
# Make the clustering result bestG type
clustering_result = np.zeros((25275, 10))
for i in range(clustering_result.shape[0]):
    clustering_result[i, clustering.labels_[i]] = 0.6

spio.savemat('spec_sc2_bestG.mat', {"bestG": clustering_result})
print(clustering)
# spio.savemat('spec_sc2_bestG_affinity_10nn.mat', {"affinityMatrix": clustering.affinity_matrix_})

# plt.imshow(clustering.affinity_matrix_, aspect='auto')
# plt.show()
# Plotting the clustering result
# colors = ['red', 'blue']
# plt.figure()
# # plt.scatter(X[0, 0], y[0, 1])
# plt.scatter(X[:, 0], X[:, 1], c=clustering.labels_, marker='.', cmap=matplotlib.colors.ListedColormap(colors))
# plt.title('K = 2')
# plt.show()