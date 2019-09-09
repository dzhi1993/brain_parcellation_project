from sklearn import datasets
import matplotlib
from sklearn.cluster import SpectralClustering, AgglomerativeClustering
import matplotlib.pyplot as plt
import scipy.io as spio
import numpy as np
from sklearn.metrics.pairwise import cosine_distances, cosine_similarity


# ---------------------------- Subject data ----------------------------- #
subjectList = ['02', '03', '04', '06', '08', '09', '10', '12', '14', '15', '17', '18', '19', '20', '21', '22', '24', '25', '26', '27', '28', '29', '30', '31']

# SC1
for sub in subjectList:
    # filename = "../Spectral clustering/subject data/subjectData_%s_sc1.mat" % sub
    mat = spio.loadmat("../Spectral clustering/subject data/sc1/subjectData_%s_sc1.mat" % sub)
    currentSubjectData = mat['A']
    affinity_matrix = cosine_similarity(currentSubjectData.transpose())
    print(affinity_matrix.shape, affinity_matrix.min(), affinity_matrix.max())

    clustering = SpectralClustering(n_clusters=10, eigen_solver='arpack', affinity="precomputed", n_jobs=-1).fit(affinity_matrix + 1)

    # Make the clustering result bestG type
    clustering_result = np.zeros((25275, 10))
    for i in range(clustering_result.shape[0]):
        clustering_result[i, clustering.labels_[i]] = 0.4

    outfilename = "../Spectral clustering/subject_clustering_results/sc1/spec_sub%s_sc1_bestG.mat" % sub
    spio.savemat(outfilename, {"bestG": clustering_result})


# SC2
for sub in subjectList:
    # filename = "../Spectral clustering/subject data/subjectData_%s_sc2.mat" % sub
    mat = spio.loadmat("../Spectral clustering/subject data/sc2/subjectData_%s_sc2.mat" % sub)
    currentSubjectData = mat['A']
    affinity_matrix = cosine_similarity(currentSubjectData.transpose())
    print(affinity_matrix.shape, affinity_matrix.min(), affinity_matrix.max())

    clustering = SpectralClustering(n_clusters=10, eigen_solver='arpack', affinity="precomputed", n_jobs=-1).fit(affinity_matrix + 1)

    # Make the clustering result bestG type
    clustering_result = np.zeros((25275, 10))
    for i in range(clustering_result.shape[0]):
        clustering_result[i, clustering.labels_[i]] = 0.6

    outfilename = "../Spectral clustering/subject_clustering_results/sc2/spec_sub%s_sc2_bestG.mat" % sub
    spio.savemat(outfilename, {"bestG": clustering_result})

