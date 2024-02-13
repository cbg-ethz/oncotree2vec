# Code base reused from https://github.com/benedekrozemberczki/graph2vec

import os,sys,json,glob,copy,ast
from pprint import pprint
from time import time
import hashlib
import networkx as nx, numpy as np
import itertools
from collections import defaultdict
from joblib import Parallel,delayed
from copy import deepcopy
import pandas as pd
from functools import partial
from itertools import chain
from itertools import product
from itertools import starmap
from itertools import combinations

class IndexGenerator:
  def __init__(self):
    self.idx = 1

  def next(self):
    self.idx = self.idx + 1
    return self.idx - 1

# Reserved node labels.
RELABELED_ROOT_LABEL = -1
RELABELED_LABEL_TO_IGNORE = -1

LOGS_FILENAME = "logs"

node_label_map = {}
node_label_generator = IndexGenerator()
label_tags = {}
mutually_exclusive_pairs = {}
word_counts = {} # hashmap with bool values

def get_hash(str):
  hash_object = hashlib.md5(str.encode())
  return hash_object.hexdigest()

def get_node_label(g, node):
  return g.nodes[node]['neighborhood_label'][0]

def multiply_label(num_times, label):
  label_list = []
  prev_feature = label
  for it in range(int(num_times)):
    label_list.append(label + "_bis_" + str(it))
    '''
    compressed_feature = get_hash(prev_feature)
    label_list.append(compressed_feature)
    prev_feature = compressed_feature
    '''
  return label_list

def update_word_counts(label, word_counts):
  if label in word_counts:
    word_counts[label] = word_counts[label] + 1
  else:
    word_counts[label] = 1

def initial_relabel(g, node_label_attr_name='Label'):
    global node_label_map
    global node_label_generator
    global word_counts
   
    try:
        opfname = g+'.tmpg'
        g = nx.read_gexf(g)
    except:
        opfname = None
        pass

    nx.convert_node_labels_to_integers(g, first_label=0)  # this needs to be done for the initial interation only
    for node in g.nodes():
      g.nodes[node]['neighborhood_label'] = {}

    for node in g.nodes():
        try:
            node_id = g.nodes[node][node_label_attr_name]
        except:
            # no node label referred in 'node_label_attr_name' is present
            assert False

        if not node_id in node_label_map:
            node_label = node_label_generator.next() 
            if str(node) == "0": # root
              node_label = str(node_label) 
            node_label_map[node_id] = node_label #insert the new label to the label map
            g.nodes[node]['neighborhood_label'][0] = str(node_label)
        else:
            g.nodes[node]['neighborhood_label'][0] = str(node_label_map[node_id])

    for node in g.nodes():
      update_word_counts(g.nodes[node]['neighborhood_label'][0], word_counts)

    if opfname:
        nx.write_gexf(g,opfname)
    else:
        return g

def wl_relabel(g, it, debug=False):
    global node_label_map
    global node_label_generator
    global word_counts
    try:
        opfname = g+'.tmpg'
        g = nx.read_gexf(g+'.tmpg')
        new_g = deepcopy(g)
        for n in g.nodes():
            new_g.nodes[n]['neighborhood_label'] = ast.literal_eval(g.nodes[n]['neighborhood_label'])
        g = new_g
    except:
        opfname = None
        pass

    prev_iter = it - 1
    for node in g.nodes():
      neighbors = list(nx.all_neighbors(g, node))        
      neighborhood_label = sorted([g.nodes[nei]['neighborhood_label'][prev_iter] for nei in neighbors])

      new_node_features = [g.nodes[node]['neighborhood_label'][prev_iter]] + neighborhood_label
      new_node_features = "_".join(new_node_features)
      if new_node_features in node_label_map:
        next_label = node_label_map[new_node_features]
      else:
        next_label = str(node_label_generator.next())
        node_label_map[new_node_features] = next_label
      if str(node) == '0': # root
        next_label = next_label 
      g.nodes[node]['neighborhood_label'][it] = next_label #get_hash(new_node_features)
      update_word_counts(next_label, word_counts)

    if opfname:
        nx.write_gexf(g,opfname)
    else:
        return g


def add_label_tag(label, tag):
  global label_tags

  if label in label_tags:
    if tag != label_tags[label]:
      assert False
  label_tags[label] = tag
  return label_tags[label]


def write_labels_to_file(label, legend_tag, fh_vocabulary, fh_vocabulary_legend):
  print(label, file=fh_vocabulary)
  label_legend = add_label_tag(label, legend_tag)
  print(label_legend, file=fh_vocabulary_legend)

def dump_sg2vec_str (fname, max_h, g, vocabulary_params):

    global mutually_exclusive_pairs
    global RELABELED_ROOT_LABEL, RELABELED_LABEL_TO_IGNORE
    global word_counts

    opfname = fname + '.g2v' + str(max_h)
    fh_vocabulary = open(opfname,'w')
    fh_vocabulary_legend = open(opfname + ".legend",'w')

    # Add labels construted around each node.
    has_non_unique_words = False
    for n,d in g.nodes(data=True):
      # Neighborhood labels
      if vocabulary_params["neighborhoods"]:
        for i in d['neighborhood_label'].keys():
          # Add label tag.
          if i not in vocabulary_params["wlk_sizes"]:
            continue
          if vocabulary_params["remove_unique_words"] and word_counts[d['neighborhood_label'][i]] <= 2:
            continue
          if i == 0: # individual nodes
            if d['neighborhood_label'][i] != str(RELABELED_ROOT_LABEL):
              add_label_tag(d['neighborhood_label'][i], "single_node")
              write_labels_to_file(d['neighborhood_label'][i], "single_node", fh_vocabulary, fh_vocabulary_legend)
              has_non_unique_words = True
          else:
            add_label_tag(d['neighborhood_label'][i], "neighborhood_" + str(i))
            write_labels_to_file(d['neighborhood_label'][i], "neighborhood_" + str(i), fh_vocabulary, fh_vocabulary_legend)
            has_non_unique_words = True

      # Structural labels
      if vocabulary_params["structure"]:
        for i in d['structure_label'].keys():
          # Add label tag.
          if i == 0:# or i == 1:
            assert False
          if vocabulary_params["remove_unique_words"] and word_counts[d['structure_label'][i]] <= 2:
            continue
          write_labels_to_file(d['structure_label'][i], "structure_" + str(i), fh_vocabulary, fh_vocabulary_legend)
          has_non_unique_words = True

      # Repeat labels for individual nodes
      if not vocabulary_params["remove_unique_words"]:
        if vocabulary_params["individual_nodes"]:
          if str(d['neighborhood_label'][0]) == str(RELABELED_LABEL_TO_IGNORE):
            break
          for label_copy in multiply_label(vocabulary_params["individual_nodes"], d['neighborhood_label'][0]):
            if d['neighborhood_label'][0] != str(RELABELED_ROOT_LABEL):
              write_labels_to_file(label_copy, "single_node_copy", fh_vocabulary, fh_vocabulary_legend)
              has_non_unique_words = True
    
      # Nonadjacent_path_pair_labels: exclude root, the current node and the ancester.
      if vocabulary_params["non_adjacent_pairs"]:
        for label in d['nonadjacent_path_pair_labels']:
          if word_counts[label] <= 2: # remove unique words
            continue
          write_labels_to_file(label, "nonadjacent_path_pair", fh_vocabulary, fh_vocabulary_legend)
          has_non_unique_words = True
          for label_copy in multiply_label(vocabulary_params["non_adjacent_pairs"]-1, label):
             write_labels_to_file(label_copy, "nonadjacent_path_pair_copy", fh_vocabulary, fh_vocabulary_legend)


      # Direct_edge_labels: If no neighborhoods are considered, the add all the direct edges. Else, add only
      # the edges between nodes with node degree > 1 (the others are covered in the neighborhoods).
      if vocabulary_params["direct_edges"]:
        for label in d['direct_edge_labels']:
          if word_counts[label] <= 2: # remove unique words
            continue
          if not(vocabulary_params["neighborhoods"] and 1 in vocabulary_params["wlk_sizes"] and g.degree(n) == 1):
            write_labels_to_file(label, "direct_edge", fh_vocabulary, fh_vocabulary_legend) 
          has_non_unique_words = True
          for label_copy in multiply_label(vocabulary_params["direct_edges"]-1, label):
            write_labels_to_file(label_copy, "direct_edge_copy", fh_vocabulary, fh_vocabulary_legend)

    # Add roo-child relations.
    if vocabulary_params["root_child_relations"]:
      for label in g.nodes['0']['root_child_relations']:
        if word_counts[label] <= 2:
          continue
        write_labels_to_file(label, "root_child_relations", fh_vocabulary, fh_vocabulary_legend)
        has_non_unique_words = True
        for label_copy in multiply_label(vocabulary_params["root_child_relations"]-1, label):
           write_labels_to_file(label_copy, "root_child_relations_copy", fh_vocabulary, fh_vocabulary_legend)

    # Add mutually exclusive pairs.
    if vocabulary_params["mutually_exclusive_pairs"]:
      for label in mutually_exclusive_pairs[g]:
        if word_counts[label] <= 2: # remove unique words
          continue
        write_labels_to_file(label, "mutually_exclusive", fh_vocabulary, fh_vocabulary_legend)
        has_non_unique_words = True
        for label_copy in multiply_label(vocabulary_params["mutually_exclusive_pairs"]-1, label):
          write_labels_to_file(label_copy, "mutually_exclusive", fh_vocabulary, fh_vocabulary_legend)

    label_legend_filename = os.path.join(os.path.dirname(fname), "label_legend.csv")
    pd.DataFrame.from_dict(data=label_tags, orient='index').to_csv(label_legend_filename, header=False)

    if vocabulary_params["remove_unique_words"] and not has_non_unique_words:
      write_labels_to_file(d['neighborhood_label'][0], "single_node", fh_vocabulary, fh_vocabulary_legend)

    if os.path.isfile(fname+'.tmpg'):
        os.system('rm '+fname+'.tmpg')
    
def print_node_labels(g):
  print(g.nodes())
  print([g.nodes[node] for node in g.nodes()])

def encode_neighborhoods(graphs, node_label_attr_name, max_h, root_label=0, label_to_ignore=None):

    global RELABELED_ROOT_LABEL, RELABELED_LABEL_TO_IGNORE

    t0 = time()
    graphs = [initial_relabel(g,node_label_attr_name) for g in graphs]
    print('initial relabeling done in {} sec'.format(round(time() - t0, 2)))
    RELABELED_ROOT_LABEL = int(node_label_map[root_label])
    if label_to_ignore: 
      RELABELED_LABEL_TO_IGNORE = int(node_label_map[label_to_ignore])

    for it in range(1, max_h + 1): # range from [1 to max_h]
        print("##### iteration", it, "#####")
        t0 = time()
        for g in graphs:
          wl_relabel(g, it)
        print('WL iteration {} done in {} sec.'.format(it, round(time() - t0, 2)))    
    return graphs

def wlk_relabel_and_dump_memory_version(fnames, max_h, vocabulary_params, node_label_attr_name='Label'):
    global node_label_generator
    global mutually_exclusive_pairs
    global RELABELED_LABEL_TO_IGNORE
    global word_counts
 
    t0 = time()
    graphs = [nx.read_gexf(fname) for fname in fnames]
    print('loaded all graphs in {} sec'.format(round(time() - t0, 2)))

    # Encode neighborhoods.
    print("Encoding neighborhoods....")
    graphs = encode_neighborhoods(graphs, node_label_attr_name, max_h,
        root_label=vocabulary_params["root_label"], label_to_ignore=vocabulary_params["ignore_label"])

    # Encoding graph structures. This is equivalent with encoding the neighborhoods
    # for non-labeled graphs (or graphs with the same label for all nodes). The root will be labeled differenty.
    print("Encoding tree structures....")
    non_labeled_graphs = deepcopy(graphs)
    root_relabel = node_label_generator.next()
    regular_node_relabel = node_label_generator.next()
    for g in non_labeled_graphs:
      for n in g.nodes():
        if str(n) == "0": # root
          g.nodes[n][node_label_attr_name] = root_relabel
        else:
          g.nodes[n][node_label_attr_name] = regular_node_relabel
    non_labeled_graphs = encode_neighborhoods(non_labeled_graphs, node_label_attr_name, max_h)
     
    # Add the new labels as structure labels in the old trees.
    for g1, g2 in zip(graphs, non_labeled_graphs):
      for n1, n2 in zip(g1.nodes(), g2.nodes()):
        g1.nodes[n1]['structure_label'] = g2.nodes[n2]['neighborhood_label']
        # Delete the structure encoding for single nodes.
        del g1.nodes[n1]['structure_label'][0]

    # Encode root-child node relations.
    print("Encoding root-child relations....")
    for g in graphs:
      g.nodes['0']['root_child_relations'] = []
      root_node = g.nodes()['0']
      root_label = get_node_label(g, "0")
      for node in nx.all_neighbors(g,'0'):
        child_label = get_node_label(g, node)
        if str(child_label) == str(RELABELED_LABEL_TO_IGNORE):
          continue
        feature = "_".join([root_label, child_label, "root-child"])
        g.nodes['0']['root_child_relations'].append(feature)      
        update_word_counts(feature, word_counts) 

    # Encode pairwise relations on the same path.
    print("Encoding co-occurence....")
    for g in graphs:
      for node in g.nodes():
        g.nodes[node]['nonadjacent_path_pair_labels'] = []
        g.nodes[node]['direct_edge_labels'] = []
 
        if str(node) == "0": # root 
          continue

        node_label = get_node_label(g, node)
        if str(node_label) == str(RELABELED_LABEL_TO_IGNORE):
          continue
   
        path_to_root = list(nx.all_simple_paths(g, source='0', target=node))
        assert len(path_to_root) == 1 # There is only one path to root
        path_to_root = path_to_root[0]
        assert path_to_root[-1] == node

        # Add reachable non-neighbouring pairs, excluding root and direct edges.
        for reachable_node in path_to_root[1:-2]: # exclude root, the current node and the ancester. The ancester will be added though the direct edges.
          reachable_node_label = get_node_label(g, reachable_node)
          if str(reachable_node_label) == str(RELABELED_LABEL_TO_IGNORE):
            continue
          feature = "_".join([reachable_node_label, node_label, "na-pair-path"])
          g.nodes[node]['nonadjacent_path_pair_labels'].append(feature)
          update_word_counts(feature, word_counts)    

        # Add direct edges. If no neighborhoods are considered, then add all the direct edges. Else, add only
        # the edges between nodes with node degree > 1 (the others are covered in the neighborhoods).
        # Also exclude root-child edges.
        if len(path_to_root) < 2:
          continue

        ancester = path_to_root[-2]
        if str(ancester) == "0": # root
          continue

        ancester_label = get_node_label(g, ancester)
        if str(ancester_label) == str(RELABELED_LABEL_TO_IGNORE):
          continue
        feature = "_".join([ancester_label, node_label, "direct-edge"])
        g.nodes[node]['direct_edge_labels'].append(feature)
        update_word_counts(feature, word_counts)    

    longest_path = 0
    # Encode pairwise relations on different paths.
    print("Encoding mutual exclusivity....")
    for g in graphs:
      features = set()
      mutually_exclusive_pairs[g] = []

      leaves = (v for v, d in g.degree() if d == 1 and v != 0)
      all_paths = partial(nx.all_simple_paths, g)
      chains = chain.from_iterable
      list_all_paths = list(chains(starmap(all_paths, product(['0'], leaves))))
      if list_all_paths:
        longest_path = max(longest_path,max([len(path) for path in list_all_paths]))

      # For each pair of paths.
      path_pairs = list(combinations(list_all_paths, 2))
      for path_pair in path_pairs:
        # Extract the branches.
        path_pair = list(set(path_pair[0]) - set(path_pair[1])), list(set(path_pair[1]) - set(path_pair[0]))
        mutual_exclusive_pairs = list(itertools.product(path_pair[0], path_pair[1]))
        for pair in mutual_exclusive_pairs:
          if str(get_node_label(g, pair[0])) == str(RELABELED_LABEL_TO_IGNORE) or str(get_node_label(g, pair[1])) == str(RELABELED_LABEL_TO_IGNORE):
            continue
          # Add the pair (pair[0], pair[1]) to the vocabulary. The order doesn't matter, therefore I sort by node label.
          feature_1 = get_node_label(g, pair[0])
          feature_2 = get_node_label(g, pair[1])
          feature = "_".join(sorted([feature_1, feature_2]) + ["mutually_exclusive"])
          features.add(feature)
      for feature in features:
        mutually_exclusive_pairs[g].append(feature)
        update_word_counts(feature, word_counts)   

    t0 = time()
    # Write subgraph labels to file.
    for fname, g in zip(fnames, graphs):
        dump_sg2vec_str(fname, max_h, g, vocabulary_params)
    print('dumped sg2vec sentences in {} sec.'.format(round(time() - t0, 2)))

    dir_path = os.path.dirname(fnames[0])
    with open(os.path.join(dir_path, "label_tags.json"), 'w') as convert_file:
      convert_file.write(json.dumps(label_tags))

    vocabulary_size = 0
    for n,d in g.nodes(data=True):
      for key in d:
        if key.lower() != "label":
          vocabulary_size = vocabulary_size + len(d[key])
    vocabulary_size = vocabulary_size / 2 # the number of trees is doubled.

    with open(os.path.join(dir_path, LOGS_FILENAME), 'w') as logs_file:
      logs_file.write("Vocabulary parameters: " + json.dumps(vocabulary_params))
      logs_file.write("\n\n")
      logs_file.write("Number of trees:" + str(len(fnames)/2)) # the number of trees is doubled. 
      logs_file.write("\n")
      logs_file.write("Longest root-leaf path: " + str(longest_path) + ".")
      logs_file.write("\n")
      logs_file.write("Vocabulary size: " + str(vocabulary_size) + " words.")
      logs_file.write("\n")

      emb_size = 32
      if vocabulary_size > 1000 and vocabulary_size <= 10000:
        emb_size = 64
      elif vocabulary_size > 10000:
        emb_size = 128
      logs_file.write("Recommended embedding size: " + str(emb_size) + ".")       

def main():

    ip_folder = '../data/kdd_datasets/ptc'
    max_h = 3

    all_files = sorted(glob.glob(os.path.join(ip_folder,'*gexf')))#[:100]
    print('loaded {} files in total'.format(len(all_files)))

    wlk_relabel_and_dump_memory_version(all_files,max_h)


if __name__ == '__main__':
    main()
