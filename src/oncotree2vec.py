# Code base reused from https://github.com/benedekrozemberczki/graph2vec

import argparse,os,logging,time
from utils import get_files
from train_utils import train_skipgram
from make_graph2vec_corpus import *
from time import time
import random

from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.models.callbacks import CallbackAny2Vec
from visualize_embeddings import visualize_embeddings

logger = logging.getLogger()
logger.setLevel("INFO")

def save_embedding(output_path, model, files, tree_mapping, dimensions):
    """
    Function to save the embedding.
    :param output_dir: Path to the output directory.
    :param model: The embedding model object.
    :param files: The list of files.
    :param dimensions: The embedding dimension parameter.
    """
    out = []
    indices = []
    for f in files:
        identifier = path2name(f)
        if not identifier[0].isdigit():
          continue
        out.append(list(model.dv["g_"+identifier]))
        indices.append(int(identifier))
    column_names = [str(dim) for dim in range(dimensions)]
    out = pd.DataFrame(out, index=indices, columns=column_names)
    out.rename(index = tree_mapping, inplace = True)
    out.index.name = "graphs"
    out.columns.name = "features"
    out.to_csv(output_path)
    return out

def path2name(path):
    base = os.path.basename(path)
    return os.path.splitext(base)[0]

def main(args):
    corpus_dir = args.corpus
    output_dir = args.output_dir
    if not os.path.exists(output_dir): 
      os.makedirs(output_dir) 

    batch_size = args.batch_size
    epochs = args.epochs
    embedding_size = args.embedding_size
    num_negsample = args.num_negsample
    learning_rate = args.learning_rate
    wlk_h = max(args.wlk_sizes)
    suffix = args.suffix
    label_filed_name = "Label" 

    wl_extn = 'g2v'+str(wlk_h)

    assert os.path.exists(corpus_dir), "File {} does not exist".format(corpus_dir)
    assert os.path.exists(output_dir), "Dir {} does not exist".format(output_dir)

    graph_files = get_files(dirname=corpus_dir, extn='.gexf', max_files=0)
    logging.info('Loaded {} graph file names form {}'.format(len(graph_files),corpus_dir))

    t0 = time()
    vocabulary_params = {
        'wlk_sizes': args.wlk_sizes,
        'individual_nodes': args.augment_individual_nodes, 
        'neighborhoods': args.augment_neighborhoods, 
        'non_adjacent_pairs': args.augment_pairwise_relations, 
        'mutually_exclusive_pairs': args.augment_mutually_exclusive_relations,
        'direct_edges': args.augment_direct_edges, 
        'structure': args.augment_tree_structure,
        'root_child_relations': args.augment_root_child_relations,
        'root_label': args.root_label,
        'ignore_label': args.ignore_label,
        'remove_unique_words': args.remove_unique_words}
    print(vocabulary_params)
    wlk_relabel_and_dump_memory_version(graph_files, max_h=wlk_h, vocabulary_params=vocabulary_params, node_label_attr_name=label_filed_name)
    logging.info('dumped sg2vec sentences in {} sec.'.format(time() - t0))

    t0 = time()
    timestamp = str(int(t0)) 
    filename_prefix = '_'.join([timestamp,
        os.path.basename(corpus_dir),
        "struct" + str(int(vocabulary_params["structure"])), 
        "neigh" + str(int(vocabulary_params["neighborhoods"])),
        "nodes" + str(int(vocabulary_params["individual_nodes"])),
        "root-child" + str(int(vocabulary_params["root_child_relations"])),
        "edges" + str(int(vocabulary_params["direct_edges"])),
        "non-adj" + str(int(vocabulary_params["non_adjacent_pairs"])),
        "mutual" + str(int(vocabulary_params["mutually_exclusive_pairs"])),
        'dims' + str(embedding_size),
        'epochs' + str(epochs),
        'lr' + str(learning_rate),
        'wlk' + str(wlk_h), 
        'ns' + str(num_negsample)]) 
    dir_path = os.path.join(args.output_dir, filename_prefix)
    os.makedirs(dir_path)
    embeddings_fname = filename_prefix + suffix + '.csv'
    embeddings_path = os.path.join(dir_path, embeddings_fname)

    if args.use_package:
      treename_mapping = pd.read_csv(corpus_dir + "/filename_index.csv", header=None, index_col=0, squeeze=True).to_dict()
      feature_files = get_files(dirname=corpus_dir, extn='.gexf.' + wl_extn, max_files=0)
      documents = []
      for filename in feature_files:
        file = open(filename, "r")
        file_index = int(filename.split('/')[-1].split('.')[0])
        file_vocabulary = file.read().split("\n")
        file_vocabulary.remove('')
        documents.append(TaggedDocument(words=file_vocabulary, tags=["g_" + str(file_index)]))

      class callback(CallbackAny2Vec):
        '''Callback to print loss after each epoch.'''
        def __init__(self):
          self.epoch = 0
        def on_epoch_end(self, model):
          loss = model.get_latest_training_loss()
          print('Loss after epoch {}: {}'.format(self.epoch, loss))
          self.epoch += 1      

      model = Doc2Vec(documents,
                    vector_size=args.embedding_size,
                    window=2, # the maximum distance between the current and predicted word within a sentence.
                    min_count=0, #args.min_count,
                    dm=0, # if 1, distributed memory (PV-DM) is used; otherwise, distributed bag of words (PV-DBOW) is employed
                    dbow_words=1, # trains word-vectors (in skip-gram fashion) simultaneous with DBOW doc-vector training
                    seed=random.randint(0, 10000000),
                    sample=0, #args.down_sampling,
                    workers=4, #args.workers,
                    epochs=epochs,
                    alpha=learning_rate,
                    #min_alpha = 0.1,
                    hs=0, #if set to 0, and negative is non-zero, negative sampling will be used.
                    negative=num_negsample,
                    compute_loss=True,
                    callbacks=[callback()])

      print("Final loss:", model.get_latest_training_loss())

      graphs = glob.glob(os.path.join(corpus_dir, "*.gexf"))
      df_embeddings = save_embedding(embeddings_path, model, graphs, treename_mapping, embedding_size)

    else:
      df_embeddings = train_skipgram(corpus_dir, wl_extn, learning_rate, embedding_size, num_negsample, epochs, batch_size, wlk_h, embeddings_path)
    logging.info('Trained the skipgram model in {} sec.'.format(round(time()-t0, 2)))

    # Visualize embeddings.
    command = ("python3 visualize_embeddings.py --in_embeddings " + embeddings_path + " --trees_json " + corpus_dir +
        "/trees.json --threshold 0.4 --wl_extn " +  wl_extn)
    print(command)

    visualize_embeddings(df_embeddings, 0.4, os.path.splitext(embeddings_path)[0], 
        corpus_dir + "/trees.json", "cosine", wl_extn, print_sub_heatmaps=True)


def parse_args():
    args = argparse.ArgumentParser("graph2vec")
    args.add_argument("-c","--corpus", required=True, 
                      help="Path to directory containing graph files to be used for  clustering")

    args.add_argument('-o', "--output_dir", default = "../embeddings",
                      help="Path to directory for storing output embeddings")

    args.add_argument('-b',"--batch_size", default=128, type=int,
                      help="Number of samples per training batch")

    args.add_argument('-e',"--epochs", default=1000, type=int,
                      help="Number of iterations the whole dataset of graphs is traversed")

    args.add_argument('-d',"--embedding_size", default=1024, type=int,
                      help="Intended graph embedding size to be learnt")

    args.add_argument('-neg', "--num_negsample", default=10, type=int,
                      help="Number of negative samples to be used for training")

    args.add_argument('-lr', "--learning_rate", default=0.3, type=float,
                      help="Learning rate to optimize the loss function")

    args.add_argument("--wlk_sizes", nargs="*", default=[0,1,2,3], type=int, help="Seizes of WL kernel (i.e., degree of rooted subgraph "
                                                           "features to be considered for representation learning)")

    args.add_argument('-s','--suffix', default='',
                      help='Suffix to be added to the output filenames.')

    args.add_argument('-x0', "--augment_individual_nodes", default=1, type=float, 
                      help="Number of times to augment the vocabulary for the individual nodes.")

    args.add_argument('-x1', "--augment_neighborhoods", default=1, type=float,
                      help="Number of times to augment the vocabulary for the tree neighborhoods.")

    args.add_argument('-x2', "--augment_pairwise_relations", default=1, type=float, 
                      help="Number of times to augment the vocabulary for the non-adjacent pairwise relations.")

    args.add_argument('-x3', "--augment_direct_edges", default=1, type=float, 
                      help="Number of times to augment the vocabulary for the direct edges.")

    args.add_argument('-x4', "--augment_mutually_exclusive_relations", default=1, type=float, 
                      help="Number of times to augment the vocabulary for the mutually exclusive relations.")

    args.add_argument('-x5', "--augment_tree_structure", default=1, type=float,
                      help="Number of times to augment the vocabulary for the tree structure.")

    args.add_argument('-x6', "--augment_root_child_relations", default=1, type=float,
                      help="Number of times to augment the vocabulary for the root-child node relations.")

    args.add_argument('-rlabel', "--root_label", default=0, type=int,
                      help="Label of the neutral node (used for discarding certain node relations).")

    args.add_argument('-ilabel', "--ignore_label", type=int,
                      help="Label to be ignored when matching individual nodes or pairwise relations. (usecase: ignoring neutral clones).")

    args.add_argument('--use_package', action='store_true')
    args.add_argument('--no_use_package', dest='use_package', action='store_false')
    args.set_defaults(use_package=False)

    args.add_argument('--remove_unique_words', action='store_true')
    args.set_defaults(remove_unique_words=False)

    return args.parse_args()

if __name__=="__main__":
    args = parse_args()
    main(args)
