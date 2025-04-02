import pandas as pd
import numpy as np
import anytree
import sys
from anytree import Node
from scipy.cluster import hierarchy

from scipy.spatial import distance
from scipy.cluster import hierarchy

from anytree import RenderTree, PreOrderIter


def get_clone_name(sample, clone):
    return sample + "_" + str(clone)


def get_sample_from_clone_name(clone_name):
    return clone_name.split("_")[0]


def get_id_from_clone_name(clone_name):
    return clone_name.split("_")[-1]


def get_leaves(root, labels):
    num_labels = len(labels)
    output = []
    for node in PreOrderIter(root):
        if node.name < num_labels:
            output.append(labels[node.name])
    return output


# Get the clusters from hierarchical clustering using a branching distance cut.
def get_clusters_from_hierarchy_linkage(
    hierarchy_linkage, labels, distance_threshold=1
):

    num_labels = len(labels)
    assert num_labels - 1 == len(hierarchy_linkage)

    node_map = {}
    root = None
    distance = 0
    for i, pair in enumerate(hierarchy_linkage):
        label_1 = int(pair[0])
        if pair[0] < num_labels:
            label_1 = labels[label_1]
        label_2 = int(pair[1])
        if pair[1] < num_labels:
            label_2 = labels[label_2]

        if i > 0:
            distance = pair[2] - hierarchy_linkage[i - 1][2]

        parent = Node(i + num_labels, weight=distance)
        node_map[i + num_labels] = parent

        if pair[0] < num_labels:
            node_0 = Node(int(pair[0]), parent=parent)
        else:
            node_map[pair[0]].parent = parent

        if pair[1] < num_labels:
            node_1 = Node(int(pair[1]), parent=parent)
        else:
            node_map[pair[1]].parent = parent

        if i == num_labels - 2:
            root = parent

    nodes = get_leaves(root, labels)
    clusters = []

    cluster_ids = list(
        hierarchy.fcluster(hierarchy_linkage, distance_threshold, criterion="distance")
    )
    num_clusters = max(cluster_ids)

    start = 0
    for i in range(num_clusters):
        cluster_size = cluster_ids.count(i + 1)
        cluster = np.array(nodes)[start : start + cluster_size]
        clusters.append(list(cluster))
        start = start + cluster_size

    return clusters


# Get max over all elements except the diagonal.
def get_max_pairwise_distance(df_distances, label_set):
    submatrix = df_distances.loc[label_set, label_set]
    submatrix = submatrix.mask(np.eye(len(submatrix.index), dtype=bool))
    return submatrix.max().max()


def get_label_set(node_map, labels, value):
    if value < len(labels):
        return [labels[int(value)]]
    else:
        return get_leaves(node_map[value], labels)


def merge_children(node, threshold):
    if hasattr(node, "weight"):
        if not node.parent and node.weight <= threshold:  # if root
            return True  # merge everything
        if node.weight <= threshold and node.parent.weight > threshold:
            return True
        else:
            return False
    else:  # leaf
        if node.parent.weight > threshold:
            return True
    return False


## Get the clusters from hierarchical clustering based on pairwise similarity.
def get_similarity_clusters(
    hierarchy_linkage, labels, df_distances, distance_threshold=1
):

    num_labels = len(labels)
    assert num_labels - 1 == len(hierarchy_linkage)

    node_map = {}
    root = None

    for i, pair in enumerate(hierarchy_linkage):

        # Create tree with distance edge weights.
        distance = -1

        if pair[0] < num_labels and pair[1] < num_labels:
            distance = df_distances[labels[int(pair[0])]].loc[labels[int(pair[1])]]
        else:
            label_set_0 = get_label_set(node_map, labels, pair[0])
            label_set_1 = get_label_set(node_map, labels, pair[1])
            distance = get_max_pairwise_distance(
                df_distances, label_set_0 + label_set_1
            )

        parent = Node(i + num_labels, weight=distance)
        node_map[i + num_labels] = parent

        if pair[0] < num_labels:
            node_0 = Node(int(pair[0]), parent=parent, sample_name=labels[int(pair[0])])
        else:
            node_map[pair[0]].parent = parent

        if pair[1] < num_labels:
            node_1 = Node(int(pair[1]), parent=parent, sample_name=labels[int(pair[1])])
        else:
            node_map[pair[1]].parent = parent

        if i == num_labels - 2:
            root = parent

    clusters = []
    for node in PreOrderIter(root):
        # if root and weight below threshold, merge everything.
        if not node.parent and node.weight <= distance_threshold:
            return [get_leaves(node, labels)]
        if node.parent:  # if not root
            if merge_children(node, distance_threshold):
                cluster = get_leaves(node, labels)
                if len(cluster) != 1:
                    assert (
                        get_max_pairwise_distance(df_distances, cluster)
                        <= distance_threshold
                    )
                clusters.append(cluster)

    return clusters


# Relabel selected anytree nodes according to their similarity based on the Jaccard distance.
# The function updates the anytrees object with the new labels and returns the new labels.
def get_labeling(
    anytrees, node_label_map, distance_threshold, label_start_id=2, plot_statistic=False
):
    nodes = list(node_label_map.keys())
    df_distances = pd.DataFrame(0, columns=nodes, index=nodes).astype(float)
    for node_1 in nodes:
        for node_2 in nodes:
            if node_1 == node_2:
                df_distances[node_1][node_2] = 1
                continue
            list_1_genes = node_label_map[node_1]
            list_2_genes = node_label_map[node_2]
            intersection = len(set(list_1_genes).intersection(set(list_2_genes)))
            union = len(set(list_1_genes).union(set(list_2_genes)))
            assert union > 0
            iou = intersection / union
            df_distances[node_1][node_2] = 1 - iou

    clustering = hierarchy.linkage(
        distance.pdist(df_distances), metric="euclidean", method="ward"
    )
    node_clusters = get_similarity_clusters(
        clustering, nodes, df_distances, distance_threshold
    )

    labels = {}
    for idx in range(0, len(node_clusters)):
        node_cluster = node_clusters[idx]
        for node in node_cluster:
            labels[node] = idx + label_start_id

    for sample_name, root in anytrees.items():
        for node in PreOrderIter(root):
            node_id = sample_name + "_" + str(node.node_id)
            if node_id in labels:
                node.matching_label = labels[node_id]

    return labels
