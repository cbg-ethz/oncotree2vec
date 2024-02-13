import ast
import os
import sys
import re
import json
import copy
import csv
from itertools import combinations
import pandas as pd
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

from scipy.spatial import distance
from scipy.cluster import hierarchy
from sklearn.metrics import silhouette_score, silhouette_samples

from utils import get_files
from parse_metadata_aml_mutation_trees import *
from parse_metadata_evolution_trees import *
from utils_score_ranks import *
from utils_clone_clustering import *

LOGS_FILENAME = "logs"

def create_distance_df(df_embeddings, metric = "cosine"):
  graphs = list(df_embeddings.index)
  df_distances = pd.DataFrame(0, columns=graphs, index=graphs).astype(float)

  for graph_1 in graphs:
    for graph_2 in graphs:  
      embedding_1 = list(df_embeddings.loc[graph_1, :])
      embedding_2 = list(df_embeddings.loc[graph_2, :])
      if metric == "euclidean":
        df_distances[graph_1][graph_2] = distance.euclidean(embedding_1, embedding_2)
      elif metric == "cosine": 
        df_distances.loc[graph_1,graph_2] = distance.cosine(embedding_1, embedding_2)
      else:
        print("No such metric", metric)
        assert False
  return df_distances

def get_max_distance_bw_clone_samples(df_embeddings, metric="cosine", suffix="#2"):
  unique_sample_list = set([sample.split('#')[0]  for sample in df_embeddings.index if suffix not in sample])   
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

def plot_heatmap (df_distances, df_embeddings, metric, sample_label_colors, color_codes, output_filename, clustering_threshold, 
    cancer_type, cmap="Blues_r", predefined_order=None, tree_clusters=None, plot_vocabulary_sizes=False):

  annotation = False
  if plot_vocabulary_sizes:
    annotation = True

  trees = list(df_distances.index)
  if predefined_order:
    assert sorted(predefined_order) == sorted(df_distances.index)
    df_distances = df_distances.loc[predefined_order, predefined_order]
    clustering = None
    plot_dendrogram = False
    silhouette_filtered_score = 0
  else:
    plot_dendrogram = True
    clustering = hierarchy.linkage(distance.pdist(df_embeddings), metric=metric, method="ward")
    tree_clusters = get_similarity_clusters(clustering, trees, df_distances, distance_threshold=clustering_threshold)
    if not plot_vocabulary_sizes:
      sample_order = [item for sublist in tree_clusters for item in sublist]
      with open(os.path.splitext(output_filename)[0] + "_sample_order.txt", 'w') as filehandle:
        json.dump(sample_order, filehandle)

  # Compute silhouette score if there is more than one cluster.
  silhouette_overall_score = 0
  silhouette_filtered_score = 0
  if tree_clusters and len(tree_clusters) > 1 and len(tree_clusters) != len(trees): # for the silhouette score we need two samples in at least one cluster
    cluster_assignmens_map = {}
    for idx in range(len(tree_clusters)):
      for sample in tree_clusters[idx]:
        cluster_assignmens_map[sample] = idx + 1

    cluster_assignmens = []
    for sample in df_embeddings.index:
      cluster_assignmens.append(cluster_assignmens_map[sample])

    silhouette_overall_score = silhouette_score(df_embeddings, cluster_assignmens, metric=metric)
    silhouette_sample_scores = silhouette_samples(df_embeddings, cluster_assignmens, metric=metric)

    cluster_selected_vals = []
    for i,cluster in enumerate(np.unique(cluster_assignmens)):
      cluster_silhouette_vals = silhouette_sample_scores[cluster_assignmens == cluster]
      if len(cluster_silhouette_vals) == 1:
        continue
      cluster_selected_vals = cluster_selected_vals + list(cluster_silhouette_vals)

      cluster_silhouette_vals.sort()

    if len(cluster_selected_vals):
      silhouette_filtered_score = sum(cluster_selected_vals) / len(cluster_selected_vals)

  #print("silhouette_overall_score", silhouette_overall_score)
  #print("silhouette_filtered_score", silhouette_filtered_score)

  min_score = df_distances.min().min() 
  max_score = df_distances.max().max() 

  sns.set(font_scale=0.4)  
  plot = sns.clustermap(
      df_distances,
      row_cluster=plot_dendrogram,
      col_cluster=plot_dendrogram,
      row_linkage=clustering,
      col_linkage=clustering,
      col_colors=sample_label_colors,
      row_colors=sample_label_colors,
      cmap=cmap, #"Blues_r",#"coolwarm_r",#"YlOrBr_r",
      vmin=min_score,
      vmax=max_score,
      dendrogram_ratio=(0.1, 0.1),
      annot=annotation,
      #fmt=".2f",
      xticklabels=True,
      yticklabels=True)

  if color_codes: 
    sns.set(font_scale=0.7)
    for label, color_code in color_codes.items():
      plot.ax_col_dendrogram.bar(0, 0, color="white", label=label, linewidth=0)
      for key, color in color_code.items():
        plot.ax_col_dendrogram.bar(0, 0, color=color, label=key, linewidth=0)
      plot.ax_col_dendrogram.bar(0, 0, color="white", label="", linewidth=0)

    if cancer_type == "aml-mutation-trees":
      ncol = 4
      legend_box_position = (0.22, 0.75)
    elif cancer_type == "trees-rob":
      ncol = 1
      legend_box_position = (0, 0)
    else:
      ncol = 3
      legend_box_position = (0, 0)
 
    l = plot.ax_col_dendrogram.legend(title="", loc="center", ncol=ncol, bbox_to_anchor=legend_box_position, 
        bbox_transform=gcf().transFigure, facecolor='white', framealpha=1)

  #plot.ax_row_dendrogram.set_visible(False) 
  #plot.ax_col_dendrogram.set_visible(False)

  plot.cax.set_visible(False)
  plot.savefig(output_filename, format='png', dpi=300)
  plt.close()

  return tree_clusters, silhouette_filtered_score

def get_sample_vocabulary(corpus_dir, wl_extn, skip_suffix):

  # Read label legends.
  label_legend_filename = os.path.join(corpus_dir, "label_legend.csv")
  
  reader = csv.reader(open(label_legend_filename, 'r'))
  label_legend = {}
  for k, v in reader:
    label_legend[k] = v

  # Read the mapping between the filenames and sample names.
  treename_mapping = pd.read_csv(corpus_dir + "/filename_index.csv", header=None, index_col=0).squeeze('columns').to_dict()
 
  # Read vocabulary from file for each tree.
  map_tree_vocabulary = {}
  feature_files = get_files(dirname=corpus_dir, extn='.gexf.' + wl_extn, max_files=0)
  for filename in feature_files:
    file = open(filename, "r")
    file_index = int(filename.split('/')[-1].split('.')[0])
    if skip_suffix in treename_mapping[file_index]:
      continue
    file_vocabulary = file.read().split("\n")
    file_vocabulary.remove('')
    map_tree_vocabulary[treename_mapping[file_index]] = file_vocabulary
 
  return map_tree_vocabulary, label_legend

class TreeSample:
  # init method or constructor
  def __init__(self, sample_name, distance_to_reference, vocabulary_intersection_to_reference):
    self.sample_name = sample_name
    self.similarity_to_reference = 1-distance_to_reference
    self.vocabulary_intersection_to_reference = vocabulary_intersection_to_reference

def visualize_embeddings(df_embeddings, clustering_threshold, out_path_embeddings_prefix, path_trees_json, metric = "cosine",
      wl_extn="", print_sub_heatmaps=False, last_iteration=True):
 
  path_split = os.path.basename(os.path.dirname(out_path_embeddings_prefix)).split("_")
  timestamp = path_split[0]
  cancer_type = path_split[3]
  threshold = path_split[5]
  tree_type = path_split[6]

  # Get distances between clone samples and the rest of the cohort.
  suffix = "#2"
  max_distance_duplicates = get_max_distance_bw_clone_samples(df_embeddings, metric=metric, suffix=suffix)
  #print("[with duplicates] Max distance duplicates:", max_distance_duplicates)

  # Remove the duplicate trees.
  df_embeddings = df_embeddings.loc[~df_embeddings.index.str.endswith(suffix)]
  df_embeddings = df_embeddings.loc[~df_embeddings.index.str.endswith("_negative")]

  # Get min/max similarity scores.
  df_distances = create_distance_df(df_embeddings, metric=metric)
  min_distance = df_distances.mask(np.eye(len(df_distances.index), dtype = bool)).min().min()
  max_distance = df_distances.mask(np.eye(len(df_distances.index), dtype = bool)).max().max()
  #print("[w/o duplicates] Min distance:", min_distance, "Max distance:", max_distance)

  vocabulary_dir = os.path.dirname(path_trees_json)
  map_tree_vocabulary, label_legend = get_sample_vocabulary(vocabulary_dir, wl_extn, skip_suffix=suffix)
  trees = list(df_embeddings.index)

  # Get pairwise sample vocabulary intersections.
  df_vocabulary_intersections = pd.DataFrame(columns=trees, index=trees)
  df_vocabulary_intersection_counts = pd.DataFrame(columns=trees, index=trees).astype(float)
  for tree_1 in trees:
    for tree_2 in trees:
      intersection = list_intersection(map_tree_vocabulary[tree_1], map_tree_vocabulary[tree_2])
      df_vocabulary_intersections.loc[tree_1,tree_2] = [item + "_" + label_legend[item] for item in intersection]
      df_vocabulary_intersection_counts.loc[tree_1,tree_2] = len(intersection)

  error_percentage = {}
  avg_error_scores = {}
  stdev_for_equal_scores = {}
  predefined_order = None
  tree_clusters = []
  sample_groups = []
  if (cancer_type == "neighborhood-matching-trees" or cancer_type == "matching-vocabulary-sizes"
      or cancer_type == "mutual-exclusivity"):
    score_map, error_percentage, avg_error_scores, stdev_for_equal_scores = get_sorted_scores(df_distances, df_vocabulary_intersections)
    predefined_order = []
    if cancer_type == "neighborhood-matching-trees":
      group_references = score_map.keys()
    elif cancer_type == "matching-vocabulary-sizes":
      group_references = [sample for sample in score_map.keys() if "reference" in sample]
    elif cancer_type == "mutual-exclusivity":
      group_references = [sample for sample in score_map.keys() if "00" in sample]

    for reference in group_references:
      score_map[reference].sort(key=lambda x:(-x.rank, x.sample_name)) # Reverse order by rank, ascendin order by sample name.
      sample_group = [reference] +  [x.sample_name for x in score_map[reference]]
      sample_groups.append(sample_group)
      predefined_order = predefined_order + sample_group
      tree_clusters.append(sample_group)

  elif cancer_type == "comparison":
    predefined_order = sorted(df_distances.index) 
    tree_clusters = [["neighborhood", "neighborhood_match"], ["direct_edge1", "direct_edge1_match"], ["direct_edge2", "direct_edge2_match"], ["direct_edge3", "direct_edge3_match"], ["mutual1", "mutual1_match"], ["mutual2", "mutual2_match"], ["mutual3", "mutual3_match"], ["root_child", "root_child_match"], ["mutual4", "mutual4_match"], ["mutual5", "mutual5_match"], ["pair", "pair_match"], ["mutualx", "mutualx_match_swap"], ["xdirect_edgex", "xdirect_edgex_no_match", "xmix", "xmix_no_match"]]

  # Plot vocabulary sizes.
  if last_iteration:
    plot_heatmap(df_vocabulary_intersection_counts, df_embeddings, metric, None, None, out_path_embeddings_prefix + "_vocabulary_sizes.png",
        clustering_threshold, cancer_type, cmap="Blues", predefined_order=predefined_order, plot_vocabulary_sizes=True)

  # Get metadata.
  sample_label_colors = None
  color_codes = None
  if cancer_type == "aml-mutation-trees" or cancer_type == "aml-trees-etienne":
    sample_label_colors, color_codes = parse_metadata_aml_mutation_trees(df_distances.index)
  elif cancer_type == "trees-rob":
    sample_label_colors, color_codes = parse_metadata_rob_trees(df_distances.index)

  # Plot full heatmap.
  tree_clusters, silhouette_score = plot_heatmap(df_distances, df_embeddings, metric, sample_label_colors, color_codes, out_path_embeddings_prefix + ".png",
      clustering_threshold, cancer_type, predefined_order=predefined_order, tree_clusters=tree_clusters)

  tree_clusters = sorted(tree_clusters, key=lambda item: -len(item)) 

  def get_avg_score(cluster, df_distances):
    if len(cluster) == 1:
      return 1 # highest distance
    score = 0
    cnt = 0
    for x in cluster:
      for y in cluster:
        if x == y:
          continue
        score = score + df_distances[x][y]
        cnt = cnt + 1
    return score/cnt

  tree_clusters_with_scores = [(cluster, get_avg_score(cluster, df_distances)) for cluster in tree_clusters] 
  tree_clusters_with_scores = sorted(tree_clusters_with_scores, key=lambda tuple: tuple[1])  
  tree_clusters = [item[0] for item in tree_clusters_with_scores]

  with open(out_path_embeddings_prefix + "_clusters.txt", 'w') as filehandle:
    json.dump([item for item in tree_clusters_with_scores if len(item[0]) > 1], filehandle)

  if cancer_type == "neighborhood-matching-trees" or cancer_type == "matching-vocabulary-sizes":
    tree_clusters = sample_groups

  # Plot umap.
  df_embeddings = df_embeddings.reindex([item for sublist in tree_clusters for item in sublist])
  embeddings = df_embeddings.to_numpy()
  umap_2d = UMAP(random_state=0)
  umap_2d.fit(embeddings)
  projections = umap_2d.transform(embeddings)
  
  umap_colors_map = {}
  
  if re.match("synthetic-trees-.+-rob", cancer_type):
    categories = list(set([sample.split("_")[0] for sample in df_embeddings.index]))
    category_color = {}
    for idx in range(len(categories)):
      category_color[categories[idx]] = idx
    for sample in df_embeddings.index:
      umap_colors_map[sample] = category_color[sample.split("_")[0]]
  else:
    for idx, cluster in enumerate(tree_clusters):  
      if len(cluster) == 1:
        umap_colors_map[cluster[0]] = len(tree_clusters)
      else:
        for sample in cluster:
          umap_colors_map[sample] = idx
  umap_colors = [str(umap_colors_map[sample]) for sample in df_embeddings.index]

  fig = px.scatter(
      projections, x=0, y=1,
      color=umap_colors, labels={'color': 'tree cluster'}, hover_data={"sample": df_embeddings.index},
      opacity=0.75
  )
  fig.write_image(out_path_embeddings_prefix + "_umap.png")
  plt.close()

  # Get cluster summaries.
  cluster_summaries = []
  for cluster in tree_clusters:
    if len(cluster) == 1:
      cluster_summaries.append("")
      continue
    vocabulary_intersection = map_tree_vocabulary[cluster[0]]
    for sample in cluster:
      vocabulary_intersection = list_intersection(vocabulary_intersection, map_tree_vocabulary[sample])
    cluster_summary = ", ".join([label_legend[word] for word in vocabulary_intersection])
    cluster_summaries.append(cluster_summary)

  # Plot sub-heatmaps for each group of trees in the synthetic dataset.
  if cancer_type == "neighborhood-matching-trees" and print_sub_heatmaps:
    for group in sample_groups:
      sample = group[0]
      submatrix = df_distances.loc[group, group]
      plot_heatmap(submatrix, df_embeddings, metric, sample_label_colors, color_codes,
            out_path_embeddings_prefix + "_" + sample  + ".png",
            clustering_threshold, cancer_type, predefined_order = group)
 
  # Generate javascript visualization.
  clusters_json_file = out_path_embeddings_prefix + ".json"
  generate_tree_visualization(tree_clusters, cluster_summaries, sample_label_colors, color_codes, path_trees_json, clusters_json_file)

  if last_iteration:
    if path_trees_json and os.path.exists(path_trees_json):
      def create_visualization_folder(template_dir, target_dir, clusters_javascript_path, placeholder):
        os.makedirs(target_dir)
        copy_tree(template_dir, target_dir)

        # Replace PLACEHOLDER json path.
        html_file = os.path.join(target_dir, "index.html")
        with open(html_file, 'r') as file:
          data = file.read()  
        data = data.replace(placeholder, clusters_javascript_path)
        with open(html_file, 'w') as file:
          file.write(data) 
        return html_file

      current_path = os.getcwd()
      visualization_dir = os.path.join(os.path.abspath(os.path.join(current_path, os.pardir)), "visualization")
      placeholder_string = "JSON_PLACEHOLDER"

      # Add tree chohort visualization.
      template_dir = os.path.join(visualization_dir, "template_tree_cohort")
      target_dir = os.path.join(visualization_dir, timestamp, "tree_cohort")
      absolute_path_trees_json = os.path.join(current_path, path_trees_json)
      html_file = create_visualization_folder(template_dir, target_dir, absolute_path_trees_json, placeholder_string)

      # Add cluster visualization.
      template_dir = os.path.join(visualization_dir, "template_clusters")
      target_dir = os.path.join(visualization_dir, timestamp, "clusters")
      clusters_javascript_file_absolute_path = os.path.join(current_path, clusters_json_file)
      html_file = create_visualization_folder(template_dir, target_dir, clusters_javascript_file_absolute_path, placeholder_string)

      chrome_path = 'open -a /Applications/Google\ Chrome.app %s'
      webbrowser.get(chrome_path).open(html_file)

  '''
  with open(os.path.join(dir_path, LOGS_FILENAME), 'w') as logs_file:
    logs_file.write("Vocabulary parameters: " + json.dumps(vocabulary_params))
    logs_file.write("\n\n")
    logs_file.write("Number of trees:" + str(len(fnames)))
    logs_file.write("\n")
    logs_file.write("Longest root-leaf path: " + str(longest_path) + ".")
  '''

  return max_distance_duplicates, max_distance, min_distance, silhouette_score, error_percentage, avg_error_scores, stdev_for_equal_scores
  
def generate_tree_visualization(tree_clusters, cluster_summaries, sample_label_colors, color_codes, path_trees_json, out_path_javascript):
  # Write the javascript file with matching clusters.
  if path_trees_json and os.path.exists(path_trees_json):
    json_file = open(path_trees_json, "r")
    json_data = json_file.read().split('=')[1]
    trees = json.loads(json_data)

    #with open(path_trees_json) as json_file:
      #trees = json.load(json_file)

    importer = JsonImporter()
    exporter = JsonExporter(indent=2, sort_keys=False)

    # Add the trees from the same cluster to the tree lists. 
    for cluster in tree_clusters:
      for sample in cluster:
        matching_samples = copy.deepcopy(cluster)
        matching_samples.remove(sample)
        for matching_sample in matching_samples:
          matched_event_tree_json = trees[matching_sample]["trees"][0]
          trees[sample]["trees"].append(trees[matching_sample]["trees"][0])

    # Keep only clusters with more than one sample (not true) and set the node colors according to the matches. 
    html_colors = ["#FF0000", "#800000", "#E67E22", "#808000", "#00FF00", "#008000", "#00FFFF", "#008080", "#0000FF", "#000080", "#FF00FF", "#800080", "#CCCCFF"]
    tree_clusters_js = {}
    for idx in range(len(tree_clusters)-1):
      cluster = tree_clusters[idx]
      if len(cluster) > 0:
        first_sample = cluster[0]

        # Collect all the nodes that are matching between each pair of trees in the cluster.
        anytrees = [importer.import_(json.dumps(tree)) for tree in trees[first_sample]["trees"]]
        node_sets = [set([node.matching_label for node in PreOrderIter(root)]) for root in anytrees]
        node_set_pairs = [comb for comb in combinations(node_sets, 2)] 
        matching_node_labels = set()
        for pair in node_set_pairs:
          matching_node_labels.update(pair[0].intersection(pair[1]))

        color_map = {}
        for label in matching_node_labels:
          color_map[label] = html_colors[len(color_map) % len(html_colors)]    

        # Add color label to the anytrees.
        tree_clusters_js[first_sample] = {}
        tree_clusters_js[first_sample]["trees"] = []
        for root in anytrees:
          for node in PreOrderIter(root):
            label = node.matching_label
            if label in color_map:
              node.node_color = color_map[label]    
          tree_clusters_js[first_sample]["trees"].append(json.loads(exporter.export(root)))

        # Add metadata.
        tree_clusters_js[first_sample]["metadata"] = sample_label_colors.loc[cluster, :].T.to_dict()
        tree_clusters_js[first_sample]["metadata_color_codes"] = color_codes
        tree_clusters_js[first_sample]["cluster_summary"] = cluster_summaries[idx]

    file = open(out_path_javascript, "w")
    file.write("sample_map=")
    file.write(json.dumps(tree_clusters_js))
    file.close()

###
# Main
if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--in_embeddings", default='', type=str)
  parser.add_argument("--trees_json", default='', type=str)
  parser.add_argument("--threshold", type=float, required=True)
  parser.add_argument("--wl_extn", default='', type=str)
  args = parser.parse_args()

  df = pd.read_csv(args.in_embeddings, index_col=0)
  visualize_embeddings(df, args.threshold, os.path.splitext(args.in_embeddings)[0], metric="cosine", path_trees_json=args.trees_json, wl_extn=args.wl_extn, print_sub_heatmaps=True)
