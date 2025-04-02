# Code base reused from https://github.com/benedekrozemberczki/graph2vec

import os, json, sys
import networkx as nx
from anytree import Node
from anytree.exporter import JsonExporter


def path2name(path):
    base = os.path.basename(path)
    return os.path.splitext(base)[0]


def get_files(dirname, extn, max_files=0):
    all_files = [
        os.path.join(dirname, f) for f in os.listdir(dirname) if f.endswith(extn)
    ]
    for root, dirs, files in os.walk(dirname):
        for f in files:
            if f.endswith(extn):
                all_files.append(os.path.join(root, f))

    all_files = list(set(all_files))
    all_files.sort()
    if max_files:
        return all_files[:max_files]
    else:
        return all_files


def save_graph_embeddings(corpus, final_embeddings, opfname):
    dict_to_save = {}
    for i in range(len(final_embeddings)):
        graph_fname = corpus._id_to_graph_name_map[i]
        graph_embedding = final_embeddings[i, :].tolist()
        dict_to_save[graph_fname] = graph_embedding

    with open(opfname, "w") as fh:
        json.dump(dict_to_save, fh, indent=4)


def gexf_to_anytree(
    paths, treename_mapping, node_label="Label", discard_sample_suffix="#2"
):
    exporter = JsonExporter()
    anytrees = {}
    for path in paths:
        sample_name = treename_mapping[path2name(path)]

        nx_tree = nx.read_gexf(path)
        node_labels = nx.get_node_attributes(nx_tree, node_label)
        node_map = {}
        for node_id in nx_tree.nodes:
            node_map[node_id] = Node(node_id, node_id=node_id)
            if node_labels:
                node_map[node_id].matching_label = node_labels[node_id]
        for edge in nx_tree.edges:
            node_map[edge[1]].parent = node_map[edge[0]]
        for key, node in node_map.items():
            if not node.parent:  # root
                anytrees[sample_name] = {}
                anytrees[sample_name]["tree"] = json.loads(exporter.export(node))
                break
    return {"trees": anytrees}
