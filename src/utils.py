# Code base reused from https://github.com/benedekrozemberczki/graph2vec

import os,json,sys

def get_files(dirname, extn, max_files=0):
    all_files = [os.path.join(dirname, f) for f in os.listdir(dirname) if f.endswith(extn)]
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
        graph_embedding = final_embeddings[i,:].tolist()
        dict_to_save[graph_fname] = graph_embedding

    with open(opfname, 'w') as fh:
        json.dump(dict_to_save,fh,indent=4)
