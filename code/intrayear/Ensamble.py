import numpy as np
from sklearn.cluster import AgglomerativeClustering, SpectralClustering, KMeans


class ClusteringEnsamble:
    def __init__(self, method='CSPA', n_clusters=4, random_state=41):
        self.method = method
        self.nclusters = n_clusters
        self.random_state=random_state

    def transform(self, labels, weight_vector = []):
        if len(weight_vector) == 0:
            weight_vector = np.ones(len(labels[0]))

        self.__avg_similarity__(labels, weight_vector)

        # return SpectralClustering(affinity='precomputed', n_clusters=self.nclusters, n_jobs=-1, random_state=self.random_state).fit_predict(self.avg_sim)
        return KMeans(n_clusters=self.nclusters, random_state=self.random_state).fit_predict(self.avg_sim)

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

