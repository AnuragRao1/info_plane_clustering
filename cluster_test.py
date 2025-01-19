# Measure mean distances from cluser centroids and generate plots to determine clustering claims with regards to specific estimates of information. 


import numpy as np
from sklearn.decomposition import PCA 
from matplotlib import pyplot as plt
from scipy.spatial.distance import cdist


def compute_cluster_distances(hidden_layers, labels):
    distances = np.zeros((len(hidden_layers),10))
    for i in range(len(hidden_layers)):
        hidden_layer = hidden_layers[i].numpy()
        for j in range(10):
            #compute mean centroids
            class_ind = np.where(labels==j)
            class_latent_representations = hidden_layer[class_ind]
            centroid = np.mean(class_latent_representations,axis=0)
            latent_distances = np.linalg.norm(class_latent_representations - centroid, axis=0)  # Compute distance of class latent reps from centroid
            mean_distance = np.mean(latent_distances)  # Compute the mean distance

            distances[i,j] = mean_distance

    return distances


def visualize(hidden_layers, labels, epoch):
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'cyan', 'magenta']
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    for i in range(len(hidden_layers)):
        hidden_layer = hidden_layers[i]
        pca = PCA(n_components=3)
        principal_comp = pca.fit_transform(hidden_layer)
        print(f"PCA for layer {i+1} explains {pca.explained_variance_} variance")
        for j in range(10):
            class_ind = np.where(labels==j)
            sample = principal_comp[class_ind]
            ax.scatter(sample[:,0], sample[:,1], sample[:,2],  c=colors[j], marker='o', label=f"{j}")

        ax.legend()
        ax.set_title(f"Clustering of layer {i+1}, epoch {epoch}")

        plt.savefig(f"layer_{i+1}_epoch_{epoch}.png")

def pairwise_dist(hidden_vectors, labels, layer, epoch):
    within_class_distances = []
    between_class_distances = []

    # WITHIN CLASS
    for j in range(10):
        class_indices = np.where(labels == j)
        class_vectors = hidden_vectors[class_indices]
        # Compute pairwise distances within the class
        distances = cdist(class_vectors, class_vectors, metric='euclidean')
        within_class_distances.extend(distances[np.triu_indices_from(distances, k=1)])  # Upper triangle to not double count symmetry

    # Loop through each pair of classes
    for i, c1 in enumerate(np.unique(labels)):
        for j, c2 in enumerate(np.unique(labels)):
            if i < j:  # we compute dist between new class pairs
                indices1 = np.where(labels == c1)
                indices2 = np.where(labels == c2)
                vectors1 = hidden_vectors[indices1]
                vectors2 = hidden_vectors[indices2]

                
                distances = cdist(vectors1, vectors2, metric='euclidean')
                between_class_distances.extend(distances.flatten())

    # Plot histograms
    plt.figure(figsize=(10, 6))

    # Within-class distances histogram
    plt.hist(within_class_distances, bins=150,range=(0,200), alpha=0.35, color='blue', label="Within")
    plt.hist(between_class_distances, bins=150,range=(0,200), alpha=0.35, color='orange', label="Between")

    plt.title(f'Epoch {epoch} - Layer {layer}')
    plt.xlabel('Distance')
    plt.ylabel('Density')
    plt.legend()

    plt.tight_layout()
    plt.savefig(f"tanh_pairwise_dist_{epoch}_{layer}.png")
    plt.close()

if __name__ == "__main__":
    print("test")