import pandas as pd
import os
import csv
import json

from scipy.spatial import distance
from utils import get_files
from utils import path2name


def create_distance_df(df_embeddings, metric="cosine"):
    graphs = list(df_embeddings.index)
    df_distances = pd.DataFrame(0, columns=graphs, index=graphs).astype(float)

    for graph_1 in graphs:
        for graph_2 in graphs:
            embedding_1 = list(df_embeddings.loc[graph_1, :])
            embedding_2 = list(df_embeddings.loc[graph_2, :])
            if metric == "euclidean":
                df_distances[graph_1][graph_2] = distance.euclidean(
                    embedding_1, embedding_2
                )
            elif metric == "cosine":
                df_distances.loc[graph_1, graph_2] = distance.cosine(
                    embedding_1, embedding_2
                )
            else:
                print("No such metric", metric)
                assert False
    return df_distances


def get_max_distance_bw_duplicate_samples(df_embeddings, metric="cosine", suffix="#2"):
    unique_sample_list = set(
        [
            sample.split(suffix)[0]
            for sample in df_embeddings.index
            if suffix not in sample
        ]
    )
    scores = []
    for sample in unique_sample_list:
        embedding_1 = list(df_embeddings.loc[sample, :])
        embedding_2 = list(df_embeddings.loc[sample + suffix, :])
        if metric == "euclidean":
            scores.append(distance.euclidean(embedding_1, embedding_2))
        elif metric == "cosine":
            dist = distance.cosine(embedding_1, embedding_2)
            scores.append(dist)
        else:
            print("No such metric", metric)
            assert False
    return max(scores)


def list_intersection(lst1, lst2):
    return [value for value in lst1 if value in lst2]


def get_sample_vocabulary(corpus_dir, gexf_extn, skip_suffix):

    # Read label legends.
    label_legend_filename = os.path.join(corpus_dir, "label_legend.csv")

    reader = csv.reader(open(label_legend_filename, "r"))
    label_legend = {}
    for k, v in reader:
        label_legend[k] = v

    # Read the mapping between the filenames and sample names.
    treename_mapping = (
        pd.read_csv(corpus_dir + "/filename_index.csv", header=None, index_col=0)
        .squeeze("columns")
        .to_dict()
    )

    # Read vocabulary from file for each tree.
    map_tree_vocabulary = {}
    feature_files = get_files(
        dirname=corpus_dir, extn=".gexf." + gexf_extn, max_files=0
    )
    for filename in feature_files:
        file = open(filename, "r")
        file_index = filename.split("/")[-1].split(".")[0]
        if skip_suffix in file_index:
            continue
        file_vocabulary = file.read().split("\n")
        file_vocabulary.remove("")
        map_tree_vocabulary[treename_mapping[file_index]] = file_vocabulary

    return map_tree_vocabulary, label_legend


def generate_tree_visualization(
    df_distances, tree_clusters, in_json_trees_path, out_json_trees_path
):
    with open(in_json_trees_path, "r") as file:
        data = file.read()
    json_oncotreevis = json.loads(data)
    json_oncotreevis["clusters"] = tree_clusters

    pairwise_distances = []
    for sample_1 in df_distances.index:
        for sample_2 in df_distances.index:
            pairwise_distances.append(
                {
                    "sample_1": sample_1,
                    "sample_2": sample_2,
                    "distance": df_distances[sample_1][sample_2],
                }
            )
    json_oncotreevis["pairwise_tree_distances"] = pairwise_distances

    # Overwrite json file.
    file = open(out_json_trees_path, "w")
    file.write(json.dumps(json_oncotreevis))
    file.close()
