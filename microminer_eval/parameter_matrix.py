import networkx as nx
from ema_workbench import load_results
import pandas as pd
import pickle
from tqdm import tqdm
from scipy.spatial.distance import pdist, squareform
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from cdlib import evaluation
from cdlib.classes import NodeClustering
import pickle
from cdlib import algorithms
import scipy.spatial as spatial
import numpy as np
import os



def compute_parameter_distances(experiments_df, columns=None, parameters=[], metric='cosine', scaling=True):
    scaler = None
    data = experiments_df[parameters].values
    scaled_data = data
    # print(scaled_data.shape)
    if scaling:
        # scaler = StandardScaler()
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(data)

    dist = pdist(scaled_data, metric=metric)
    if columns is not None:
        dist_df = pd.DataFrame(squareform(dist), columns=columns, index=columns)
    else:
        dist_df = pd.DataFrame(squareform(dist))
    # print(dist_df.shape)
    return dist_df, scaler

# Omega index indicates the similarity between two partitions
# If omega = 1, the two partitions are identical (distance = 0), and omega = 0 (distance = 1) is the opposite case
# Thus, omega works as a similarity index
def compute_omega_index(partition_i, partition_j, graph, distance=False):
    clustering_i = NodeClustering(communities=list(partition_i.values()), graph=graph, overlap=True)
    clustering_j = NodeClustering(communities=list(partition_j.values()), graph=graph, overlap=True)
    if distance:
        return 1 - evaluation.omega(clustering_i, clustering_j).score
    else:
        return evaluation.omega(clustering_i, clustering_j).score

def get_noise_classes(partition, graph):
    reference_class_set = set(list(graph.nodes()))
    partition_class_set = set([x for xs in partition.values() for x in xs])
    difference_set = reference_class_set.difference(partition_class_set)
    return difference_set

def update_partition_with_noise(partition, graph):
    noise_classes = get_noise_classes(partition, graph)
    if len(noise_classes) > 0:
        partition[-1] = list(noise_classes) # -1 is the key for the noise classes

    return partition


#--------------------------------------------------------

# GRAPH_FILENAME = "./jpetstore/jpetstore_128scenarios_nopolicies_sobol_graph.pkl"
GRAPH_FILENAME = "../cargo/cargo_128scenarios_nopolicies_sobol_graph.pkl"

# MODEL_FILENAME = './jpetstore/jpetstore_128scenarios_nopolicies_sobol' #.tar.gz'
MODEL_FILENAME = './cargo/cargo_128scenarios_nopolicies_sobol' #.tar.gz'

# PARTITIONS_FILENAME = "./jpetstore/jpetstore_128scenarios_nopolicies_sobol_partitions.pkl"
PARTITIONS_FILENAME = "../cargo/cargo_128scenarios_nopolicies_sobol_partitions.pkl"

# DISTANCE_FILENAME = "./jpetstore/jpetstore_parameter_distances.csv"
DISTANCE_FILENAME = "../cargo/cargo_parameter_distances.csv"

# STABLE_SOLUTIONS_FILENAME = "./jpetstore/jpetstore_stable_solutions.pkl"
STABLE_SOLUTIONS_FILENAME = "../cargo/cargo_stable_solutions.pkl"

java_graph = None
with open(GRAPH_FILENAME, 'rb') as f:
     java_graph = pickle.load(f)

ALL_PARAMETERS = ['alpha',  'mfuzzy',  'microservice_threshold',  'resolution']

experiments_df, outcomes = load_results(MODEL_FILENAME+ '.tar.gz')
print(experiments_df.shape)
experiments_df = experiments_df[ALL_PARAMETERS].drop_duplicates(keep='first')
print(experiments_df.shape)

# Creating labels for the experiments
exp_labels = []
for idx, row in experiments_df.iterrows():
    lb = 'resolution_'+str(row['resolution'])+'_alpha_'+str(row['alpha'])+'_mfuzzy_'+str(row['mfuzzy'])+'_mthreshold_'+str(row['microservice_threshold'])
    exp_labels.append(lb)
# print(exp_labels)
experiments_df.index = exp_labels
print(experiments_df.head())

partitions_dict = None
with open(PARTITIONS_FILENAME, 'rb') as f:
     partitions_dict = pickle.load(f)
print("partitions:", len(partitions_dict))
key_0 = list(partitions_dict.keys())[10]
# print(partitions_dict[key_0])
print(key_0)

relevant_parameters = ['resolution', 'mfuzzy', 'microservice_threshold'] # ['resolution'] # ALL_PARAMETERS
# relevant_parameters = ['resolution', 'mfuzzy']

print("Computing parameter distances ...", relevant_parameters)
distances_df, scaler = compute_parameter_distances(experiments_df, experiments_df.index, parameters=relevant_parameters,
                                        metric='euclidean', scaling=True)
print(distances_df.shape)

distances_df.to_csv(DISTANCE_FILENAME)
# distances_df = pd.read_csv(DISTANCE_FILENAME, index_col=0)

distance_np = np.tril(distances_df).flatten()
min_non_zero = np.min(distance_np[np.nonzero(distance_np)])
print("min-max distances:", min_non_zero, distance_np.max())
print(distances_df.max().max())
