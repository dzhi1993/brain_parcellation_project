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
idv_dir = "D:/python_workspace/brain_parcellation_project/data/"
subj_name = ['s01', 's02', 's03', 's04', 's05', 's06', 's07', 's08', 's09', 's10', 's11',
             's12', 's13', 's14', 's15', 's16', 's17', 's18', 's19', 's20', 's21', 's22', 's23', 's24',
             's25', 's26', 's27', 's28', 's29', 's30', 's31']
goodsubj = [2, 3, 4, 6, 8, 9, 10, 12, 14, 15, 17, 18, 19, 20, 21, 22, 24, 25, 26, 27, 28, 29, 30, 31]
resolution = 2562
hem = ['L', 'R']
D = ['A', 'B']
exp = [1, 2]

# # ---------------------------  Parcellation for each subject ------------------------------ #
# Find the index of medial wall
# gifti_image = nb.load(wb_dir + "Icosahedron-%d.32k.L.label.gii" % resolution)
# img_data = [x.data for x in gifti_image.darrays]
# labels = np.reshape(img_data, (len(img_data[0]), 1))
# nanIndex_L = np.where(labels == 0)[0]
#
# gifti_image = nb.load(wb_dir + "Icosahedron-%d.32k.R.label.gii" % resolution)
# img_data = [x.data for x in gifti_image.darrays]
# labels = np.reshape(img_data, (len(img_data[0]), 1))
# nanIndex_R = np.where(labels == 0)[0]
#
# # Making medial wall mask 32k
# MAP_L = np.empty((32492, 1))
# MAP_L[:] = np.nan
# MAP_L[nanIndex_L, :] = 0
#
# MAP_R = np.empty((32492, 1))
# MAP_R[:] = np.nan
# MAP_R[nanIndex_R, :] = 0
# MAP = np.concatenate((MAP_L, MAP_R), axis=0)
#
# # Main loop
# for sub in range(len(goodsubj)):
#     for ts in exp:
#         Data_L = []
#         Data_R = []
#         rawData_L = []
#         rawData_R = []
#         for h in range(len(hem)):
#             # Loading data
#             mat = spio.loadmat(idv_dir + subj_name[goodsubj[sub]-1] + "/%s.%s.swcon.exp%d.32k.mat" % (subj_name[goodsubj[sub]-1], hem[h], ts))
#             groupData = mat['data']
#             # gifti_data = nb.load(idv_dir + subj_name[goodsubj[sub]-1] + "/%s.%s.swcon.32k.func.gii" % (subj_name[goodsubj[sub]-1], hem[h]))
#             # img_data = [x.data for x in gifti_data.darrays]
#             # data = np.reshape(img_data, (len(img_data), len(img_data[0])))
#             # groupData = data.transpose()
#
#             verticesIdx = np.reshape(np.arange(len(groupData)), (len(groupData), 1))
#             groupData_labels = np.concatenate((groupData, labels, verticesIdx), axis=1)
#             # groupData_noNaN = np.delete(groupData_labels, nanIndex, 0)
#
#             groupData_reduced = np.empty((1, groupData.shape[1]))
#             groupData_reduced[:] = np.nan
#             for ico in np.unique(labels[:, 0]):
#                 if ico != 0:
#                     curr_tessellation = groupData_labels[groupData_labels[:, groupData.shape[1]] == ico]
#                     if curr_tessellation.shape[0] != 0:
#                         curr_avrg = np.reshape(curr_tessellation[:, 0:groupData.shape[1]].mean(0), (1, groupData.shape[1]))
#                         groupData_reduced = np.append(groupData_reduced, curr_avrg, axis=0)
#
#             groupData_reduced = np.delete(groupData_reduced, 0, 0)
#
#             if h == 0:
#                 Data_L = np.copy(groupData_reduced)
#                 rawData_L = np.copy(groupData)
#             elif h == 1:
#                 Data_R = np.copy(groupData_reduced)
#                 rawData_R = np.copy(groupData)
#
#         groupData = np.concatenate((Data_L, Data_R), axis=0)
#         raw_groupData = np.concatenate((rawData_L, rawData_R), axis=0)
#         affinity_matrix = cosine_similarity(groupData) + 1
#         print(affinity_matrix.shape)
#
#         for k in range(7, 31):
#             print('Running cluster %d ...' % k + 'for subject %d ' % goodsubj[sub] + 'of experiment %d.' % ts)
#             curr_map = np.copy(MAP)
#             clustering = SpectralClustering(n_clusters=k, eigen_solver='arpack', n_init=50, affinity="precomputed", n_jobs=-1).fit(affinity_matrix)
#             # Make the clustering result bestG type
#             clustering_result = clustering.labels_ + 1
#
#             raw_groupData[np.isnan(raw_groupData)] = 0
#             lookupTable = cosine_similarity(raw_groupData, groupData) + 1
#
#             for idx in range(len(raw_groupData)):
#                 this_label = clustering_result[np.argmax(lookupTable[idx])]
#                 if np.isnan(curr_map[idx][0]):
#                     curr_map[idx][0] = this_label
#
#             parcels = np.split(curr_map, 2)
#             parcels_L = parcels[0]
#             parcels_R = parcels[1]
#
#             outfilename = "../Spectral clustering/subject_clustering_results/sc%d/%s/spec_cosine_sc%d_all_%d.mat" % (ts, subj_name[goodsubj[sub]-1], ts, k)
#             spio.savemat(outfilename, {"parcels_L": parcels_L, "parcels_R": parcels_R})

# # ---------------------------  Parcellation for group average ------------------------------ #
mat = spio.loadmat("../data/betas_cortex/vertices/group_swcon.mat")
raw_groupData = np.concatenate((mat['A'], mat['B']), axis=0)
# raw_groupData = raw_groupData[:, 0:29]  # only SC1
raw_groupData = raw_groupData[:, 29:62]  # only SC2

MAP_L = np.empty((32492, 1))
MAP_L[:] = np.nan

MAP_R = np.empty((32492, 1))
MAP_R[:] = np.nan

for h in range(len(hem)):
    # Find the index of medial wall
    gifti_image = nb.load(wb_dir + "Icosahedron-%d.32k.%s.label.gii" % (resolution, hem[h]))
    img_data = [x.data for x in gifti_image.darrays]
    labels = np.reshape(img_data, (len(img_data[0]), 1))
    nanIndex = np.where(labels == 0)[0]

    if h == 0:
        MAP_L[nanIndex, :] = 0
    elif h == 1:
        MAP_R[nanIndex, :] = 0

MAP = np.concatenate((MAP_L, MAP_R), axis=0)

for k in range(10, 31):
    print('Running cluster %d ...' % k)
    curr_map = np.copy(MAP)
    clustering = AgglomerativeClustering(n_clusters=k, linkage='average', affinity='cosine').fit(raw_groupData)
    # Make the clustering result bestG type
    clustering_result = clustering.labels_ + 1
    clustering_result[np.where(MAP == 0)[0]] = 0
    clustering_result = np.reshape(clustering_result, (len(clustering_result), 1))

    parcels = np.split(clustering_result, 2)
    parcels_L = parcels[0]
    parcels_R = parcels[1]

    outfilename = "../agglomerative clustering/cortex_clustering_results/group/sc2/ac_cosine_single_sc2_%d.mat" % k
    spio.savemat(outfilename, {"parcels_L": parcels_L, "parcels_R": parcels_R})


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
