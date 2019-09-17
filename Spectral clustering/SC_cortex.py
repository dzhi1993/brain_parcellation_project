from sklearn import datasets
import matplotlib
from sklearn.cluster import SpectralClustering, AgglomerativeClustering, KMeans
import matplotlib.pyplot as plt
import scipy.io as spio
import numpy as np
from scipy.spatial.distance import pdist
from sklearn.metrics.pairwise import cosine_distances, cosine_similarity
import nibabel as nb
from nibabel.gifti import giftiio

wb_dir = "D:/data/sc1/surfaceWB/group32k/"
subj_name = ['s01', 's02', 's03', 's04', 's05', 's06', 's07', 's08', 's09', 's10', 's11',
             's12', 's13', 's14', 's15', 's16', 's17', 's18', 's19', 's20', 's21', 's22', 's23', 's24',
             's25', 's26', 's27', 's28', 's29', 's30', 's31']
goodsubj = [2, 3, 4, 6, 8, 9, 10, 12, 14, 15, 17, 18, 19, 20, 21, 22, 24, 25, 26, 27, 28, 29, 30, 31]
resolution = 2562
hem = 'L'

# # ---------------------------- SC 1 ----------------------------- #

# clustering = SpectralClustering(n_clusters=10,
#                                 eigen_solver='arpack',
#                                 affinity="cosine",
#                                 n_jobs=-1).fit(groupData.transpose())

# # spio.savemat('clusters.mat', {"Y": clustering.labels_})

gifti_image = nb.load(wb_dir + "Icosahedron-%d.32k.%s.label.gii" % (resolution, hem))
img_data = [x.data for x in gifti_image.darrays]
np_data = np.reshape(img_data, len(img_data[0]))

# Get the group averaged data
for index in range(len(goodsubj)):
    sub = subj_name[goodsubj[index] - 1]
    mat = spio.loadmat("../data/betas_cortex/vertices/beta_avrg_noInstr_%s.mat" % sub)

    # group data across two sessions
    groupData_L = mat['avrgBeta_sc12_L']
    groupData_R = mat['avrgBeta_sc12_R']
    groupData = np.concatenate((groupData_L, groupData_R), axis=0)

    # # group data for sc1 only
    # groupData_L = mat['avrgBeta_sc1_L']
    # groupData_R = mat['avrgBeta_sc1_R']
    # groupData_sc1 = np.concatenate((groupData_L, groupData_R), axis=0)
    #
    # # group data for sc2 only
    # groupData_L = mat['avrgBeta_sc2_L']
    # groupData_R = mat['avrgBeta_sc2_R']
    # groupData_sc2 = np.concatenate((groupData_L, groupData_R), axis=0)

    # Loading the NaN value indices and transfer to 0-based array
    nanIndex_L = mat['nanIndex_L']
    nanIndex_L = np.reshape(nanIndex_L, len(nanIndex_L)) - 1
    nanIndex_R = mat['nanIndex_R']
    nanIndex_R = np.reshape(nanIndex_R, len(nanIndex_R)) - 1
    nanIndex = np.concatenate((nanIndex_L, nanIndex_R + 32492), axis=0)

    print(groupData.shape, nanIndex_L.shape, nanIndex_R.shape)

# Parcellation for group averaged data on vertices based space.




# Parcellation for each subject
for index in range(len(goodsubj)):
    sub = subj_name[goodsubj[index] - 1]
    mat = spio.loadmat("../data/betas_cortex/vertices/beta_avrg_noInstr_%s.mat" % sub)

    # group data across two sessions
    groupData_L = mat['avrgBeta_sc12_L']
    groupData_R = mat['avrgBeta_sc12_R']
    groupData = np.concatenate((groupData_L, groupData_R), axis=0)

    # group data for sc1 only
    groupData_L = mat['avrgBeta_sc1_L']
    groupData_R = mat['avrgBeta_sc1_R']
    groupData_sc1 = np.concatenate((groupData_L, groupData_R), axis=0)

    # group data for sc2 only
    groupData_L = mat['avrgBeta_sc2_L']
    groupData_R = mat['avrgBeta_sc2_R']
    groupData_sc2 = np.concatenate((groupData_L, groupData_R), axis=0)

    # Loading the NaN value indices and transfer to 0-based array
    nanIndex_L = mat['nanIndex_L']
    nanIndex_L = np.reshape(nanIndex_L, len(nanIndex_L)) - 1
    nanIndex_R = mat['nanIndex_R']
    nanIndex_R = np.reshape(nanIndex_R, len(nanIndex_R)) - 1
    nanIndex = np.concatenate((nanIndex_L, nanIndex_R + 32492), axis=0)

    print(groupData.shape, groupData_sc1.shape, groupData_sc2.shape, nanIndex_L.shape, nanIndex_R.shape)

    # Now, remove the NaN values from the raw data because the spectral clustering cannot take nan values
    groupData = np.delete(groupData, nanIndex, 0)

    affinity_matrix = cosine_similarity(groupData) + 1
    affinity_matrix = np.int16(affinity_matrix * 10000)
    print(affinity_matrix.shape)

    # Y = pdist(affinity_matrix, 'cosine')

    for k in range(7, 31):  # First try the range of cluster's number from 7 to 30
        clustering = SpectralClustering(n_clusters=k, eigen_solver='arpack', n_init=30, affinity="precomputed", n_jobs=-1).fit(affinity_matrix)
        # Make the clustering result bestG type
        clustering_result = np.zeros((clustering.labels_.shape[0], 1))
        for i in range(clustering_result.shape[0]):
            clustering_result[i][0] = clustering.labels_[i]

        outfilename = "../Spectral clustering/cortex_clustering_results/%s/vertices/spec_cosine_sc12_%d.mat" % (sub, k)
        spio.savemat(outfilename, {"parcel": clustering_result})

        # clustering = SpectralClustering(n_clusters=k, eigen_solver='arpack', affinity="precomputed", n_jobs=-1).fit(affinity_matrix_L)
        # # Make the clustering result bestG type
        # clustering_result = np.zeros((clustering.labels_.shape[0], 1))
        # for i in range(clustering_result.shape[0]):
        #     clustering_result[i][0] = clustering.labels_[i]
        #
        # outfilename = "../Spectral clustering/cortex_clustering_results/%s/spec_cosine_%d_L.mat" % (sub, k)
        # spio.savemat(outfilename, {"parcel": clustering_result})
        #
        # clustering = SpectralClustering(n_clusters=k, eigen_solver='arpack', affinity="precomputed", n_jobs=-1).fit(affinity_matrix_R)
        # # Make the clustering result bestG type
        # clustering_result = np.zeros((clustering.labels_.shape[0], 1))
        # for i in range(clustering_result.shape[0]):
        #     clustering_result[i][0] = clustering.labels_[i]
        #
        # outfilename = "../Spectral clustering/cortex_clustering_results/%s/spec_cosine_%d_R.mat" % (sub, k)
        # spio.savemat(outfilename, {"parcel": clustering_result})


# spio.savemat('spec_sc1_bestG.mat', {"bestG": clustering_result})
# np.savetxt("spec_sc1_bestG_affinity_rbf_arpack.csv", clustering.affinity_matrix_, delimiter=',', newline=',')
# spio.savemat('spec_sc1_bestG_affinity_10nn.mat', {"affinityMatrix": clustering.affinity_matrix_})


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
