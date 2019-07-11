from sklearn import datasets
import matplotlib
from sklearn.cluster import SpectralClustering, AgglomerativeClustering
import matplotlib.pyplot as plt
import scipy.io as spio
import numpy as np
from sklearn.metrics.pairwise import cosine_distances, cosine_similarity


# ---------------------------- Subject data ----------------------------- #
# subjectList = ['02', '03', '04', '06', '08', '09', '10', '12', '14', '15', '17', '18', '19', '20', '21', '22', '24', '25', '26', '27', '28', '29', '30', '31']
#
# # SC1
# for sub in subjectList:
#     # filename = "../Spectral clustering/subject data/subjectData_%s_sc1.mat" % sub
#     mat = spio.loadmat("../Spectral clustering/subject data/sc1/subjectData_%s_sc1.mat" % sub)
#     currentSubjectData = mat['A']
#     affinity_matrix = cosine_similarity(currentSubjectData.transpose())
#     print(affinity_matrix.shape, affinity_matrix.min(), affinity_matrix.max())
#
#     clustering = SpectralClustering(n_clusters=10, eigen_solver='arpack', affinity="precomputed", n_jobs=-1).fit(affinity_matrix + 1)
#
#     # Make the clustering result bestG type
#     clustering_result = np.zeros((25275, 10))
#     for i in range(clustering_result.shape[0]):
#         clustering_result[i, clustering.labels_[i]] = 0.4
#
#     outfilename = "../Spectral clustering/subject_clustering_results/sc1/spec_sub%s_sc1_bestG.mat" % sub
#     spio.savemat(outfilename, {"bestG": clustering_result})
#
#
# # SC2
# for sub in subjectList:
#     # filename = "../Spectral clustering/subject data/subjectData_%s_sc2.mat" % sub
#     mat = spio.loadmat("../Spectral clustering/subject data/sc2/subjectData_%s_sc2.mat" % sub)
#     currentSubjectData = mat['A']
#     affinity_matrix = cosine_similarity(currentSubjectData.transpose())
#     print(affinity_matrix.shape, affinity_matrix.min(), affinity_matrix.max())
#
#     clustering = SpectralClustering(n_clusters=10, eigen_solver='arpack', affinity="precomputed", n_jobs=-1).fit(affinity_matrix + 1)
#
#     # Make the clustering result bestG type
#     clustering_result = np.zeros((25275, 10))
#     for i in range(clustering_result.shape[0]):
#         clustering_result[i, clustering.labels_[i]] = 0.6
#
#     outfilename = "../Spectral clustering/subject_clustering_results/sc2/spec_sub%s_sc2_bestG.mat" % sub
#     spio.savemat(outfilename, {"bestG": clustering_result})


# # ---------------------------- SC 1 ----------------------------- #
mat = spio.loadmat('distSphere_sp.mat')
groupData = mat['avrgDs']
groupData = groupData.todense()
print(groupData.transpose().shape)

# affinity_matrix = cosine_similarity(groupData.transpose())
# print(affinity_matrix.shape, affinity_matrix.min(), affinity_matrix.max())

# plt.imshow(affinity_matrix, aspect='auto')
# plt.show()


# clustering = SpectralClustering(n_clusters=10,
#                                 eigen_solver='arpack',
#                                 affinity="cosine",
#                                 n_jobs=-1).fit(groupData.transpose())

# # spio.savemat('clusters.mat', {"Y": clustering.labels_})
clustering = SpectralClustering(n_clusters=60,
                                eigen_solver='arpack',
                                affinity="precomputed",
                                n_jobs=-1).fit(groupData)

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

# # ---------------------------- SC 2 ----------------------------- #
# mat = spio.loadmat('../groupData_sc2.mat')
# groupData = mat['X_C']
#
# affinity_matrix = cosine_similarity(groupData.transpose())
# print(affinity_matrix.shape, affinity_matrix.min(), affinity_matrix.max())
#
# # clustering = SpectralClustering(n_clusters=10,
# #                                 eigen_solver='arpack',
# #                                 affinity="nearest_neighbors",
# #                                 n_neighbors=10,
# #                                 n_jobs=-1).fit(groupData.transpose() * 2)
# # # print(affinity.labels_.shape, affinity.labels_)
# # # # spio.savemat('clusters.mat', {"Y": clustering.labels_})
#
# clustering = SpectralClustering(n_clusters=10,
#                                 eigen_solver='arpack',
#                                 affinity="precomputed",
#                                 n_jobs=-1).fit(affinity_matrix + 1)
#
# print(clustering.labels_.shape, clustering.labels_)
#
# # # clustering = AgglomerativeClustering(n_clusters=10,
# # #                                      affinity='euclidean',
# # #                                      memory=None,
# # #                                      connectivity=None,
# # #                                      compute_full_tree='auto',
# # #                                      linkage='ward',
# # #                                      pooling_func='deprecated').fit(groupData.transpose())
# #
# # print(clustering.labels_.shape, clustering.labels_)
# #
# # Make the clustering result bestG type
# clustering_result = np.zeros((25275, 10))
# for i in range(clustering_result.shape[0]):
#     clustering_result[i, clustering.labels_[i]] = 0.6
#
# spio.savemat('spec_sc2_bestG.mat', {"bestG": clustering_result})
# print(clustering)
# # spio.savemat('spec_sc2_bestG_affinity_10nn.mat', {"affinityMatrix": clustering.affinity_matrix_})
#
# # plt.imshow(clustering.affinity_matrix_, aspect='auto')
# # plt.show()
# # Plotting the clustering result
# # colors = ['red', 'blue']
# # plt.figure()
# # # plt.scatter(X[0, 0], y[0, 1])
# # plt.scatter(X[:, 0], X[:, 1], c=clustering.labels_, marker='.', cmap=matplotlib.colors.ListedColormap(colors))
# # plt.title('K = 2')
# # plt.show()
