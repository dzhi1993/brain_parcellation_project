## Brain Parcellation Project

Diedrichsen Lab, Brain and Mind Institute, Western University (Slow progress)

Human brain parcellation aims to identify distinct functional regions to facilitate the understanding of the human brain as a modular structure. Practically, such regions are important to guide the analysis of human functional magnetic resonance imaging (fMRI) data. Numerous parcellations for the human neocortex and cerebellum have been proposed over the last years, based on anatomical boundaries, functional task-based data, or functional resting-state connectivity. 

we explore the pros and cons of different parcellation methods applied for finding functional boundaries in human brain. Our evaluation method relies on a novel Multi-Domain Task Battery [MDTB](http://www.diedrichsenlab.org/imaging/mdtb.htm) dataset (King et al., 2018, BioRxiv), which contains a wide range of tasks involving multiple motor, social, and cognitive functions.

### Dependencies

Most of the codes are written based on several popular scientific analysis packages, please make sure you have installed all necessary packages before running the code by pip tool:

    pip install numpy sklearn matplotlib scipy nibabel

If there are some packages not listed above, then use command below to install separately,

    pip install <package name>
    

### Running the code

the python files of the project mainly can be run independently since we explore many clustering methods for brain parcellation. So that we can simply run:

	python <filename>.py

to run each file.

### Clustering algorithms

Including multiple clustering or clustering-type algorithms.

Semi-NMF (semi non negative matrix factorization algorithm), NMF, spectral clustering, agglomerative clustering, DBSCAN
and regular k-means clustering algorithms, in order to make a comparison to get the clustering method that can best reflect the functional connectives between human's brain and individual tasks. 