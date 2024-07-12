import numpy as np
from sklearn.cluster import AgglomerativeClustering

class ClusteringEnsamble:
    def __init__(self, method='CSPA', n_clusters=4):
        self.method = method
        self.nclusters = n_clusters

    def transform(self, labels, weight_vector = []):
        if len(weight_vector) == 0:
            weight_vector = np.ones(len(labels[0]))


        self.__avg_similarity__(labels, weight_vector)
        return AgglomerativeClustering(
            metric='precomputed', 
            n_clusters=self.nclusters, 
            linkage='complete'
        ).fit(1-self.avg_sim).labels_

    def __avg_similarity__(self, labels, weight_vector: list):
        self.nclusterers_ = len(labels)
        self.samples = len(labels[0])
        self.avg_sim = np.zeros((self.samples, self.samples))

        for weight_index, labelVector in enumerate(labels):
            self.avg_sim = self.avg_sim + self.__similarity_matrix__(labelVector) * weight_vector[weight_index]

        self.avg_sim = self.avg_sim / self.nclusterers_

    def __similarity_matrix__(self, labels):
        if isinstance(labels, np.ndarray):
            sim_mat = []
            
            for i in range(self.samples):
                if labels[i] != -1:
                    sim_mat.append((labels == labels[i]).astype(np.int32))
                else:
                    temp = (np.zeros(self.samples)).astype(np.int32)
                    temp[i] = 1
                    sim_mat.append(temp)
            return np.array(sim_mat)
        else:
            raise TypeError("Labels Must Be Of Type numpy.ndarray")

