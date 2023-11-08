import os,logging
import numpy as np
import pandas as pd
from corpus_parser import Corpus
from utils import save_graph_embeddings
from skipgram import skipgram
import sys

def train_skipgram (corpus_dir, extn, learning_rate, embedding_size, num_negsample, epochs, batch_size, wlk, op_fname): #output_dir, timestamp, suffix):
    '''
    :param corpus_dir: folder containing WL kernel relabeled files. All the files in this folder will be relabled
    according to WL relabeling strategy and the format of each line in these folders shall be: <target> <context 1> <context 2>....
    :param extn: Extension of the WL relabled file
    :param learning_rate: learning rate for the skipgram model (will involve a linear decay)
    :param embedding_size: number of dimensions to be used for learning subgraph representations
    :param num_negsample: number of negative samples to be used by the skipgram model
    :param epochs: number of iterations the dataset is traversed by the skipgram model
    :param batch_size: size of each batch for the skipgram model
    :param op_fname: path where to save the embeddings
    :param output_dir: the folder where embedding file will be stored
    :return: name of the file that contains the subgraph embeddings (in word2vec format proposed by Mikolov et al (2013))
    '''

    '''
    op_fname = '_'.join([timestamp, os.path.basename(corpus_dir), 'dims', str(embedding_size), 'epochs',
                         str(epochs),'lr',str(learning_rate),'wlk',str(wlk), 'ns', str(num_negsample), 'embeddings', suffix + '.txt'])
    op_fname = os.path.join(output_dir, op_fname)
    if os.path.isfile(op_fname):
        logging.info('The embedding file: {} is already present, hence NOT training skipgram model '
                     'for subgraph vectors'.format(op_fname))
        return op_fname
    '''

    logging.info("Initializing SKIPGRAM...")
    treename_mapping = pd.read_csv(corpus_dir + "/filename_index.csv", header=None, index_col=0, squeeze=True).to_dict()
    corpus = Corpus(corpus_dir, treename_mapping, extn = extn, max_files=0)  # just load 'max_files' files from this folder
    corpus.scan_and_load_corpus()

    out_path_prefix = os.path.splitext(op_fname)[0]
    model_skipgram = skipgram(
        num_graphs=corpus.num_graphs,
        num_subgraphs=corpus.num_subgraphs,
        learning_rate=learning_rate,
        embedding_size=embedding_size,
        num_negsample=num_negsample,
        num_steps=epochs,  # no. of time the training set will be iterated through
        corpus=corpus,  # data set of (target,context) tuples
        corpus_dir=corpus_dir,
        out_path_prefix = out_path_prefix,
        wl_extn=extn
    )

    df_embeddings = model_skipgram.train(corpus=corpus,batch_size=batch_size)
    embeddings_file = out_path_prefix + ".csv"
    df_embeddings.to_csv(embeddings_file)
    return df_embeddings

if __name__ == '__main__':
    pass
