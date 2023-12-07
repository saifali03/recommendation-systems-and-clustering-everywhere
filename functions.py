import pandas as pd
import numpy as np

## general functions
def get_data_from_2_2(nrows):
    return pd.read_csv("pca.csv", nrows=nrows)


## for RQ 2.3
def calculate_distances(center, data, essential_cols):
    """
    takes a row of a dataframe (called center, which is in fact a dataframe with exactly one row)
    and the corresponding dataframe.
    returns the euclidean distances w.r.t to center and data.
    """
    distances = data.apply( lambda row : ((center-row[essential_cols])**2).sum(axis=1), axis=1) # calculating centroids w.r.t. first centroid
    return distances

def get_center_by_weighted_random(data, distances):
    weights = distances.to_numpy().reshape(-1)
    center = np.random.choice( np.arange(0,len(data)), p=weights/sum(weights) )
    return pd.DataFrame(data.iloc[center,]).T


def initialise_clusters_standard(k, data, essential_cols):
    centers = data[essential_cols].sample(n=k).copy() # drawing k samples uniformly random without replacement

    # add distances w.r.t to the last created center
    distances_df = pd.DataFrame()
    for i in range(k):
        center = pd.DataFrame(centers.iloc[i].copy()).T
        distances_df['center_' + str(i)] = calculate_distances(center, data, essential_cols)
    return distances_df.idxmin(axis=1)



def initialise_clusters_adv(k, data, essential_cols):
    distances_df = pd.DataFrame()
    centers = pd.DataFrame()

    for i in range(k):
        # create centers
        if len(centers) == 0:
            center = data[essential_cols].sample(n=1).copy()
        else:
            distances = distances_df.min(axis=1)
            center = get_center_by_weighted_random(data, distances)
        centers = pd.concat([centers, center])
        # add distances w.r.t to the last created center
        distances_df['center_' + str(i)] = calculate_distances(center, data, essential_cols)
    return distances_df.idxmin(axis=1) #  centers # data['cluster'] =


def calculate_new_centers(data, essential_cols): # this is the "reduce" of map-reduce
    return data.groupby('cluster')[essential_cols].mean()

def get_new_clusters(data, centers, essential_cols): # this is the "map" of map-reduce
    k = len(centers)
    distances_df = pd.DataFrame()
    for i in range(k):
        center = pd.DataFrame(centers.iloc[i,:]).T
        distances_df['center_' + str(i)] = calculate_distances(center, data, essential_cols)
        error = distances_df.min(axis=1).sum()

    return error, distances_df.idxmin(axis=1)


def iterate_clustering(data, essential_cols, eps=0.1, max_iterations=None ):
    error = eps + 1
    error_old = 2*error
    i=0
    while abs(error-error_old) > eps:
        if max_iterations is not None:
            i = i+1
            if i >= max_iterations:
                break
        error_old = error
        centers = calculate_new_centers(data, essential_cols)
        error, clusters = get_new_clusters(data, centers, essential_cols)
        data['cluster'] = clusters


def our_k_means_adv(k, data, essential_cols, eps=0.1, max_iterations=None):
    data = data.copy()
    data['cluster'] = initialise_clusters_adv(k, data, essential_cols)
    iterate_clustering(data, essential_cols, eps=eps, max_iterations=max_iterations)
    return data

def our_k_means_standard(k, data, essential_cols, eps=0.1, max_iterations=None):
    data = data.copy()
    data['cluster'] = initialise_clusters_standard(k, data, essential_cols)
    iterate_clustering(data, essential_cols, eps=eps,  max_iterations=max_iterations)
    return data




# data = get_data_from_2_2(100)
# our_k_means_adv(4, data, ['0', '1', '2'], eps=1, max_iterations=2)

# our_k_means_standard(4, data, ['0', '1', '2'], eps=1, max_iterations=2)



