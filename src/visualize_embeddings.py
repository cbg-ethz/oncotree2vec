import ast
import os
import sys
import re
import os
import json
import copy
import csv
from itertools import combinations
import numpy as np
import argparse
import webbrowser
from distutils.dir_util import copy_tree
from anytree.importer import JsonImporter
from anytree.exporter import JsonExporter

import matplotlib.pyplot as plt
from matplotlib.pyplot import gcf
import seaborn as sns
import plotly.express as px

from umap import UMAP

from scipy.cluster import hierarchy
from sklearn.metrics import silhouette_score, silhouette_samples

from parse_metadata_aml_mutation_trees import *
from parse_metadata_evolution_trees import *
from parse_metadata_tupro_ovarian import *
from utils_score_ranks import *
from utils_clone_clustering import *
from utils_visualization import *

LOGS_FILENAME = "logs"


class TreeSample:
    # init method or constructor
    def __init__(
        self, sample_name, distance_to_reference, vocabulary_intersection_to_reference
    ):
        self.sample_name = sample_name
        self.similarity_to_reference = 1 - distance_to_reference
        self.vocabulary_intersection_to_reference = vocabulary_intersection_to_reference


def plot_heatmap(
    df_distances,
    df_embeddings,
    metric,
    sample_label_colors,
    color_codes,
    output_filename,
    clustering_threshold,
    cmap="Blues_r",
    ncol_legend=3,
    legend_box_position=(0, 0),
    predefined_order=None,
    tree_clusters=None,
    generate_heatmaps=True,
    plot_vocabulary_sizes=False,
):

    annotation = False
    annot_kws = {"size": 35 / np.sqrt(len(df_distances))}
    if plot_vocabulary_sizes:
        annotation = True

    if not clustering_threshold:
        clustering_threshold = 0.5

    trees = list(df_distances.index)
    if predefined_order:
        assert sorted(predefined_order) == sorted(df_distances.index)
        df_distances = df_distances.loc[predefined_order, predefined_order]
        clustering = None
        plot_dendrogram = False
        silhouette_filtered_score = 0
    else:
        plot_dendrogram = True
        clustering = hierarchy.linkage(
            distance.pdist(df_embeddings), metric=metric, method="ward"
        )
        tree_clusters = get_similarity_clusters(
            clustering, trees, df_distances, distance_threshold=clustering_threshold
        )
        if not plot_vocabulary_sizes:
            sample_order = [item for sublist in tree_clusters for item in sublist]
            with open(
                os.path.splitext(output_filename)[0] + "_sample_order.txt", "w"
            ) as filehandle:
                json.dump(sample_order, filehandle)

    # Compute silhouette score if there is more than one cluster with at least two samples.
    silhouette_overall_score = 0
    silhouette_filtered_score = 0
    if tree_clusters and len(tree_clusters) > 1 and len(tree_clusters) != len(trees):
        cluster_assignmens_map = {}
        for idx in range(len(tree_clusters)):
            for sample in tree_clusters[idx]:
                cluster_assignmens_map[sample] = idx + 1

        cluster_assignmens = []
        for sample in df_embeddings.index:
            cluster_assignmens.append(cluster_assignmens_map[sample])

        silhouette_overall_score = silhouette_score(
            df_embeddings, cluster_assignmens, metric=metric
        )
        silhouette_sample_scores = silhouette_samples(
            df_embeddings, cluster_assignmens, metric=metric
        )

        cluster_selected_vals = []
        for i, cluster in enumerate(np.unique(cluster_assignmens)):
            cluster_silhouette_vals = silhouette_sample_scores[
                cluster_assignmens == cluster
            ]
            if len(cluster_silhouette_vals) == 1:
                continue
            cluster_selected_vals = cluster_selected_vals + list(
                cluster_silhouette_vals
            )

            cluster_silhouette_vals.sort()

        if len(cluster_selected_vals):
            silhouette_filtered_score = sum(cluster_selected_vals) / len(
                cluster_selected_vals
            )

    min_score = df_distances.min().min()
    max_score = df_distances.max().max()

    if generate_heatmaps:
        sns.set(font_scale=0.4)
        plot = sns.clustermap(
            df_distances,
            row_cluster=plot_dendrogram,
            col_cluster=plot_dendrogram,
            row_linkage=clustering,
            col_linkage=clustering,
            row_colors=sample_label_colors,
            cmap=cmap,
            vmin=min_score,
            vmax=max_score,
            dendrogram_ratio=(0.1, 0.1),
            annot=annotation,
            annot_kws=annot_kws,
            xticklabels=True,
            yticklabels=True,
        )

        if color_codes:
            sns.set(font_scale=0.7)
            for label, color_code in color_codes.items():
                plot.ax_col_dendrogram.bar(
                    0, 0, color="white", label=label, linewidth=0
                )
                for key, color in color_code.items():
                    plot.ax_col_dendrogram.bar(
                        0, 0, color=color, label=key, linewidth=0
                    )
                plot.ax_col_dendrogram.bar(0, 0, color="white", label="", linewidth=0)

            l = plot.ax_col_dendrogram.legend(
                title="",
                loc="center",
                ncol=ncol_legend,
                bbox_to_anchor=legend_box_position,
                bbox_transform=gcf().transFigure,
                facecolor="white",
                framealpha=1,
            )

        plot.cax.set_visible(False)
        figure = plt.gcf()
        figure.set_size_inches(9.1, 9)
        plot.savefig(output_filename, format="png", dpi=300)
        plt.close()

    return tree_clusters, silhouette_filtered_score


def visualize_embeddings(
    df_embeddings,
    out_file_path_prefix,
    corpus_dir,
    metric="cosine",
    gexf_extn="vocabulary",
    dup_suffix="#2",
    generate_heatmaps=True,
    print_sub_heatmaps=False,
    last_iteration=True,
    clustering_threshold=None,
):

    output_dir_name = os.path.basename(os.path.dirname(out_file_path_prefix))

    # Get distances between clone samples and the rest of the cohort.
    max_distance_duplicates = get_max_distance_bw_duplicate_samples(
        df_embeddings, metric=metric, suffix=dup_suffix
    )

    # Remove the duplicate trees.
    df_embeddings = df_embeddings.loc[~df_embeddings.index.str.endswith(dup_suffix)]
    df_embeddings = df_embeddings.loc[~df_embeddings.index.str.endswith("_negative")]

    # Get min/max similarity scores.
    df_distances = create_distance_df(df_embeddings, metric=metric)
    min_distance = (
        df_distances.mask(np.eye(len(df_distances.index), dtype=bool)).min().min()
    )
    max_distance = (
        df_distances.mask(np.eye(len(df_distances.index), dtype=bool)).max().max()
    )

    # Get pairwise sample vocabulary intersections.
    map_tree_vocabulary, label_legend = get_sample_vocabulary(
        corpus_dir, gexf_extn, skip_suffix=dup_suffix
    )
    trees = list(df_embeddings.index)
    df_vocabulary_intersections = pd.DataFrame(columns=trees, index=trees)
    df_vocabulary_intersection_counts = pd.DataFrame(columns=trees, index=trees).astype(
        float
    )
    for tree_1 in trees:
        for tree_2 in trees:
            intersection = list_intersection(
                map_tree_vocabulary[tree_1], map_tree_vocabulary[tree_2]
            )
            df_vocabulary_intersections.loc[tree_1, tree_2] = [
                item + "_" + label_legend[item] for item in intersection
            ]
            df_vocabulary_intersection_counts.loc[tree_1, tree_2] = len(intersection)

    error_percentage = {}
    avg_error_scores = {}
    stdev_for_equal_scores = {}
    predefined_order = None
    tree_clusters = []
    sample_groups = []
    if (
        "neighborhood-matching-trees" in output_dir_name
        or "matching-vocabulary-sizes" in output_dir_name
        or "mutual-exclusivity" in output_dir_name
    ):
        score_map, error_percentage, avg_error_scores, stdev_for_equal_scores = (
            get_sorted_scores(df_distances, df_vocabulary_intersections)
        )
        predefined_order = []
        if "neighborhood-matching-trees" in output_dir_name:
            group_references = score_map.keys()
        elif "matching-vocabulary-sizes" in output_dir_name:
            group_references = [
                sample for sample in score_map.keys() if "reference" in sample
            ]
        elif "mutual-exclusivity" in output_dir_name:
            group_references = [sample for sample in score_map.keys() if "00" in sample]
        for reference in group_references:
            # Reverse order by rank, ascendin order by sample name.
            score_map[reference].sort(key=lambda x: (-x.rank, x.sample_name))
            sample_group = [reference] + [x.sample_name for x in score_map[reference]]
            sample_groups.append(sample_group)
            predefined_order = predefined_order + sample_group
            tree_clusters.append(sample_group)
    elif "comparison" in output_dir_name:
        predefined_order = sorted(df_distances.index)
        tree_clusters = [
            ["neighborhood", "neighborhood_match"],
            ["direct_edge1", "direct_edge1_match"],
            ["direct_edge2", "direct_edge2_match"],
            ["direct_edge3", "direct_edge3_match"],
            ["mutual1", "mutual1_match"],
            ["mutual2", "mutual2_match"],
            ["mutual3", "mutual3_match"],
            ["root_child", "root_child_match"],
            ["mutual4", "mutual4_match"],
            ["mutual5", "mutual5_match"],
            ["pair", "pair_match"],
            ["mutualx", "mutualx_match_swap"],
            ["xdirect_edgex", "xdirect_edgex_no_match", "xmix", "xmix_no_match"],
        ]

    # Plot vocabulary sizes.
    if last_iteration and generate_heatmaps:
        plot_heatmap(
            df_vocabulary_intersection_counts,
            df_embeddings,
            metric,
            None,
            None,
            out_file_path_prefix + "_vocabulary_sizes.png",
            clustering_threshold,
            cmap="Blues",
            predefined_order=predefined_order,
            plot_vocabulary_sizes=True,
        )

    # Get metadata.
    sample_label_colors = None
    color_codes = None
    if "trees_morita" in output_dir_name or "trees_etienne" in output_dir_name:
        sample_label_colors, color_codes = parse_metadata_aml_mutation_trees(
            df_distances.index
        )
        ncol = 4
        legend_box_position = (0.22, 0.75)
    elif "trees_noble" in output_dir_name:
        sample_label_colors, color_codes = parse_metadata_rob_trees(df_distances.index)
        ncol = 1
        legend_box_position = (0, 0)
    elif "tupro_ovarian" in output_dir_name:
        sample_label_colors, color_codes = parse_metadata_tupro_ovarian(
            df_distances.index
        )
        ncol = 4
        legend_box_position = (0.5, 1.15)
    else:
        ncol = 3
        legend_box_position = (0, 0)

    # Plot full heatmap.
    tree_clusters, silhouette_score = plot_heatmap(
        df_distances,
        df_embeddings,
        metric,
        sample_label_colors,
        color_codes,
        out_file_path_prefix + "_heatmap.png",
        clustering_threshold,
        ncol_legend=ncol,
        legend_box_position=legend_box_position,
        predefined_order=predefined_order,
        tree_clusters=tree_clusters,
        generate_heatmaps=generate_heatmaps,
    )

    # If a threshold for the hierarchical clustering is provided, compute additional outputs.
    # Otherwise, do so in the last iteration using a default threshold of 0.5 (hardcoded above).
    if clustering_threshold or last_iteration:

        def get_avg_score(cluster, df_distances):
            if len(cluster) == 1:
                return 1  # highest distance
            score = 0
            cnt = 0
            for x in cluster:
                for y in cluster:
                    if x == y:
                        continue
                    score = score + df_distances[x][y]
                    cnt = cnt + 1
            return score / cnt

        cluster_scores = [
            get_avg_score(cluster, df_distances) for cluster in tree_clusters
        ]
        tree_clusters_with_scores = sorted(
            zip(tree_clusters, cluster_scores), key=lambda tuple: tuple[1]
        )
        tree_clusters = [item[0] for item in tree_clusters_with_scores]

        if (
            "neighborhood-matching-trees" in output_dir_name
            or "matching-vocabulary-sizes" in output_dir_name
        ):
            tree_clusters = sample_groups

        # Plot UMAP.
        df_embeddings = df_embeddings.reindex(
            [item for sublist in tree_clusters for item in sublist]
        )
        embeddings = df_embeddings.to_numpy()
        umap_2d = UMAP()  # random_state=0
        umap_2d.fit(embeddings)
        projections = umap_2d.transform(embeddings)

        umap_colors_map = {}

        for idx, cluster in enumerate(tree_clusters):
            if len(cluster) == 1:
                umap_colors_map[cluster[0]] = len(tree_clusters)
            else:
                for sample in cluster:
                    umap_colors_map[sample] = idx
        umap_colors = [str(umap_colors_map[sample]) for sample in df_embeddings.index]

        fig = px.scatter(
            projections,
            x=0,
            y=1,
            color=umap_colors,
            labels={"color": "tree cluster"},
            hover_data={"sample": df_embeddings.index},
            opacity=0.75,
        )
        fig.write_image(out_file_path_prefix + "_umap.png")
        plt.close()

        # Get cluster summaries.
        cluster_summaries = {}
        for idx, cluster in enumerate(tree_clusters):
            if len(cluster) == 1:
                continue
            vocabulary_intersection = map_tree_vocabulary[cluster[0]]
            for sample in cluster:
                vocabulary_intersection = list_intersection(
                    vocabulary_intersection, map_tree_vocabulary[sample]
                )
            cluster_summary = ", ".join(
                [
                    label_legend[word]
                    for word in vocabulary_intersection
                    if "copy" not in label_legend[word]
                ]
            )
            cluster_summaries["cluster " + str(idx)] = {
                "samples": cluster,
                "vocabulary intersection": cluster_summary,
                "min similarity score": get_avg_score(cluster, df_distances),
            }

        with open("_".join([out_file_path_prefix, "clusters.txt"]), "w") as filehandle:
            print(
                "Clusters with at least 2 elements, in reverse order by size:\n",
                file=filehandle,
            )
            json.dump(cluster_summaries, filehandle, indent=2)

    # Plot sub-heatmaps for each group of trees in the synthetic dataset.
    if "neighborhood-matching-trees" in output_dir_name and print_sub_heatmaps:
        for group in sample_groups:
            sample = group[0]
            submatrix = df_distances.loc[group, group]
            plot_heatmap(
                submatrix,
                df_embeddings,
                metric,
                sample_label_colors,
                color_codes,
                out_file_path_prefix + "_" + sample + ".png",
                clustering_threshold,
                predefined_order=group,
            )

    # Generate javascript visualization.
    path_oncotreevis_file = os.path.join(corpus_dir, "trees.json")
    out_oncotreevis_file = "_".join([out_file_path_prefix, "oncotreevis.json"])
    generate_tree_visualization(
        df_distances,
        tree_clusters,
        in_json_trees_path=path_oncotreevis_file,
        out_json_trees_path=out_oncotreevis_file,
    )

    return (
        max_distance_duplicates,
        max_distance,
        min_distance,
        silhouette_score,
        error_percentage,
        avg_error_scores,
        stdev_for_equal_scores,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_embeddings", default="", type=str)
    parser.add_argument("--corpus_dir", default="", type=str)
    parser.add_argument("--threshold", default=None, type=float)
    parser.add_argument("--gexf_extn", default="vocabulary", type=str)
    parser.add_argument("--generate_heatmaps", default=True, type=bool)
    args = parser.parse_args()

    df = pd.read_csv(args.in_embeddings, index_col=0)
    visualize_embeddings(
        df,
        os.path.splitext(args.in_embeddings)[0],
        metric="cosine",
        corpus_dir=args.corpus_dir,
        clustering_threshold=args.threshold,
        gexf_extn=args.gexf_extn,
        generate_heatmaps=args.generate_heatmaps,
        print_sub_heatmaps=True,
    )
