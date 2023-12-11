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


def compute_wcss(k, data, essential_cols):
    """
    Takes a dataframe and the number of clusters you chose and compute the WCSS (the sum
    of the sum of squares between each point of a cluster and its center, for each center)
    This is a helper function that is used for 2.3.2
    """
    wcss = 0
    for cluster in data['cluster'].unique():
        cluster_pts = data[data['cluster'] == cluster]
        center = pd.DataFrame([cluster_pts.mean(numeric_only=True)])
        dist = calculate_distances(center, cluster_pts, essential_cols)
        wcss+= dist.sum()
    return wcss
   
   
   	
def our_k_means_standard(k, data_path, essential_cols, nrows=10**4, chunksize=10**3, upper_bound_for_iterations=50 ):
    centers = init_centers_standard(k, data_path, essential_cols, nrows=nrows)
    # data = pd.read_csv(data_path, nrows=nrows)
    # centers = data[essential_cols].head(k) # non-random-init
    # print(centers)
    previous_centers = None
    for _ in range(upper_bound_for_iterations):
        previous_centers = centers.copy()
        map_res = my_map(data_path, essential_cols, centers, chunksize, nrows)
        centers = my_reduce(map_res, essential_cols)
        if centers.equals(previous_centers):
            break
    data = pd.read_csv(data_path, nrows=nrows)
    data['cluster'] = cluster(data, essential_cols, centers)
    return data

def our_k_means_adv(k, data_path, essential_cols, nrows=10**4, chunksize=10**3, upper_bound_for_iterations=50 ):
    centers = init_centers_adv(k, data_path, essential_cols, nrows=nrows)
    # data = pd.read_csv(data_path, nrows=nrows)
    # centers = data[essential_cols].head(k) # non-random-init
    # print(centers)
    previous_centers = None
    for _ in range(upper_bound_for_iterations):
        previous_centers = centers.copy()
        map_res = my_map(data_path, essential_cols, centers, chunksize, nrows)
        centers = my_reduce(map_res, essential_cols)
        if centers.equals(previous_centers):
            break
    data = pd.read_csv(data_path, nrows=nrows)
    data['cluster'] = cluster(data, essential_cols, centers)
    return data



def init_centers_standard(k, data_path, essential_cols, nrows):
    data = pd.read_csv(data_path, nrows = nrows)
    centers = data[essential_cols].sample(k).copy()
    return centers

def init_centers_adv(k, data_path, essential_cols, nrows):
    distances_df = pd.DataFrame()
    data = pd.read_csv(data_path, nrows = nrows)
    centers = data[essential_cols].sample(1)
    for i in range(k-1):
        distances_df[str(i)] = np.linalg.norm(data[essential_cols].to_numpy() - centers.iloc[i].to_numpy(), axis=1)
        distances = distances_df.min(axis=1).to_numpy()
        center_index = np.random.choice( np.arange(0,len(data)), p=distances/sum(distances) )
        centers = pd.concat([centers, pd.DataFrame(data[essential_cols].iloc[center_index,:]).T])
    return centers


def cluster(data, essential_cols, centers):
    distance_df = pd.DataFrame()
    # print(data[essential_cols])
    for i, center in centers[essential_cols].iterrows():
        # print(center)
        distance_df[str(i)] = np.linalg.norm(data[essential_cols].to_numpy() - center.to_numpy(), axis=1)
    clustering = distance_df.idxmin(axis=1)
    return clustering

def my_map(data_path, essential_cols, centers, chunksize, nrows):
    map_res = pd.DataFrame()
    for chunk in pd.read_csv(data_path, chunksize=chunksize, nrows=nrows):
        chunk['cluster'] = cluster(chunk, essential_cols, centers).to_numpy()
        right = chunk.groupby('cluster').size().reset_index(name='count')
        left = chunk.groupby('cluster')[essential_cols].agg(lambda x : x.sum() ).reset_index()
        merged = pd.merge(left=left, right=right, left_on='cluster', right_on='cluster')
        map_res = pd.concat([map_res, merged])
        # map_res.reset_index(inplace=True, drop=True)
    return map_res


def my_reduce(map_res, essential_cols):
    map_res_grouped = map_res.groupby('cluster').agg(lambda x:  x.sum())
    return map_res_grouped.reset_index().apply( lambda row : row[essential_cols]/row['count'], axis=1)




def compute_wcss(k, data, essential_cols):
    """
    Takes a dataframe and the number of clusters you chose and compute the WCSS (the sum
    of the sum of squares between each point of a cluster and its center, for each center)
    This is a helper function that is used for 2.3.2
    """
    wcss = 0
    for cluster in data['cluster'].unique():
        cluster_pts = data[data['cluster'] == cluster]
        center = pd.DataFrame([cluster_pts.mean(numeric_only=True)])
        dist = calculate_distances(center, cluster_pts, essential_cols)
        wcss+= dist.sum()
    return wcss
   
   

