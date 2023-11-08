import pandas as pd
import numpy as np
#import tensorflow as tf
import tensorflow.compat.v1 as tf
import tensorflow.compat.v1 as tf
import matplotlib.pyplot as plt
import math,logging
from pprint import  pprint
from time import time
import sys

from visualize_embeddings import visualize_embeddings

class skipgram(object):
    '''
    skipgram model - refer Mikolov et al (2013)
    '''

    def __init__(self,num_graphs,num_subgraphs,learning_rate,embedding_size,
                 num_negsample,num_steps,corpus,corpus_dir,out_path_prefix,wl_extn):
        self.num_graphs = num_graphs
        self.num_subgraphs = num_subgraphs
        self.embedding_size = embedding_size
        self.num_negsample = num_negsample
        self.learning_rate = learning_rate
        self.num_steps = num_steps
        self.corpus = corpus
        self.corpus_dir = corpus_dir
        self.wl_extn = wl_extn
        self.out_path_prefix = out_path_prefix,
        self.graph, self.batch_inputs, self.batch_labels,self.normalized_embeddings,\
        self.loss,self.optimizer, self.out_path_prefix = self.trainer_initial()

    def trainer_initial(self):
        graph = tf.Graph()
        with graph.as_default():
            batch_inputs = tf.placeholder(tf.int32, shape=([None, ]))
            batch_labels = tf.placeholder(tf.int64, shape=([None, 1]))
 
            graph_embeddings = tf.Variable(
                    tf.random_uniform([self.num_graphs, self.embedding_size], -0.5 / self.embedding_size, 0.5/self.embedding_size))

            batch_graph_embeddings = tf.nn.embedding_lookup(graph_embeddings, batch_inputs) #hidden layer

            weights = tf.Variable(tf.truncated_normal([self.num_subgraphs, self.embedding_size],
                                                          stddev=1.0 / math.sqrt(self.embedding_size))) #output layer wt
            biases = tf.Variable(tf.zeros(self.num_subgraphs)) #output layer biases

            #negative sampling part
            loss = tf.reduce_mean(
                tf.nn.nce_loss(weights=weights,
                               biases=biases,
                               labels=batch_labels,
                               inputs=batch_graph_embeddings,
                               num_sampled=self.num_negsample,
                               num_classes=self.num_subgraphs,
                               sampled_values=tf.nn.fixed_unigram_candidate_sampler(
                                   true_classes=batch_labels,
                                   num_true=1,
                                   num_sampled=self.num_negsample,
                                   unique=True,
                                   range_max=self.num_subgraphs,
                                   distortion=0.75,
                                   unigrams=self.corpus.subgraph_id_freq_map_as_list)#word_id_freq_map_as_list is the
                               # frequency of each word in vocabulary
                               ))

            global_step = tf.Variable(0, trainable=False)
            learning_rate = tf.train.exponential_decay(self.learning_rate,
                                                       global_step, 100000, 0.96, staircase=True) #linear decay over time

            learning_rate = tf.maximum(learning_rate,0.001) #cannot go below 0.001 to ensure at least a minimal learning

            optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,global_step=global_step)

            norm = tf.sqrt(tf.reduce_mean(tf.square(graph_embeddings), 1, keep_dims=True))
            normalized_embeddings = graph_embeddings/norm

            out_path_prefix = tf.Variable(self.out_path_prefix[0])

        return graph,batch_inputs, batch_labels, normalized_embeddings, loss, optimizer, out_path_prefix

    def train(self,corpus,batch_size):
        with tf.Session(graph=self.graph,
                        config=tf.ConfigProto(log_device_placement=True,allow_soft_placement=False)) as sess:
            def embeddings_to_dataframe(embeddings, sample_names, corpus):
              df = pd.DataFrame(embeddings, index = sample_names)
              df.index.name = "graphs"
              df.columns.name = "features"
              return df

            init = tf.global_variables_initializer()
            sess.run(init)

            loss = 0
            loss_values = []
            silhouette_scores = []
            max_distances_duplicates = []
            max_distances = []
            min_distances = []
            sample_names = [corpus._id_to_sample_name[i] for i in range(len(corpus._id_to_sample_name))]
            error_percentages_all_iterations = {}
            avg_error_scores_all_iterations = {}
            stdev_for_equal_scores_all_iterations = {}

            for i in range(self.num_steps):
                t0 = time()
                step = 0
                while corpus.epoch_flag == False:
                    batch_data, batch_labels = corpus.generate_batch_from_file(batch_size)# get (target,context) wordid tuples
                    feed_dict = {self.batch_inputs:batch_data,self.batch_labels:batch_labels}
                    _, loss_val, out_path_prefix = sess.run([self.optimizer,self.loss, self.out_path_prefix],feed_dict=feed_dict)
                    out_path_prefix = out_path_prefix.decode("utf-8") 

                    loss += loss_val
                    if step % 100 == 0:
                        if step > 0:
                            average_loss = loss/step
                            logging.info( 'Epoch: %d : Average loss for step: %d : %f'%(i,step,average_loss))
                    step += 1

                corpus.epoch_flag = False
                epoch_time = time() - t0
                logging.info('#########################   Epoch: %d :  %f, %.2f sec.  #####################' % (i, loss/step,epoch_time))
                if i % 100 == 0:
                  df = embeddings_to_dataframe(self.normalized_embeddings.eval(), sample_names, corpus)
                  filename_prefix = out_path_prefix + "iter_" + str(i)
                  max_distance_duplicates, max_distance, min_distance, silhouette_score, error_percentages, avg_error_scores, stdev_for_equal_scores = visualize_embeddings(df, 0.5,
                      filename_prefix, path_trees_json=self.corpus_dir+"/trees.json", wl_extn=self.wl_extn, last_iteration=False, metric="cosine")
                  df.to_csv(filename_prefix + ".csv", index=True)

                  for sample in error_percentages:
                    if sample not in error_percentages_all_iterations:
                      error_percentages_all_iterations[sample] = [error_percentages[sample]]
                      avg_error_scores_all_iterations[sample] = [avg_error_scores[sample]]
                      stdev_for_equal_scores_all_iterations[sample] = [stdev_for_equal_scores[sample]]
                    else:
                      error_percentages_all_iterations[sample].append(error_percentages[sample])
                      avg_error_scores_all_iterations[sample].append(avg_error_scores[sample])
                      stdev_for_equal_scores_all_iterations[sample].append(stdev_for_equal_scores[sample])

                loss_values.append(loss/step)
                max_distances_duplicates.append(max_distance_duplicates)
                max_distances.append(max_distance)
                min_distances.append(min_distance)
                silhouette_scores.append(silhouette_score)
                loss = 0
 
            # Plot loss.
            fig = plt.figure()
            plt.plot(loss_values)
            plt.gca().set_ylim([0, 4])
            plt.xlabel("num iterations")
            plt.ylabel("loss")
            fig.savefig(out_path_prefix + "loss_values.png", format='png', dpi=300)

            # Plot min/max distances and silhouette scores.
            fig = plt.figure()
            if not(len(set(silhouette_scores)) == 1 and silhouette_scores[0] == 0):
              plt.plot(silhouette_scores, label="silhouette")
            plt.plot(max_distances_duplicates, label="max dist dup")
            plt.plot(max_distances, label="max dist")
            plt.plot(min_distances, label="min dist")
            plt.xlabel("num iterations")
            plt.ylabel("scores")
            plt.legend(loc="upper right")
            fig.savefig(out_path_prefix + "other_scores.png", format='png', dpi=300)

            # Plot error scores if applicable.
            def plot_error_scores(error_scores_all_iterations, filename, ylabel, y_axis_upper_limit = None):
              fig = plt.figure()
              if error_scores_all_iterations: 
                first_sample = list(error_scores_all_iterations.keys())[0]
                x_axis_values = np.arange(100, (len(error_scores_all_iterations[first_sample])+1)*100, 100)
                for sample in error_scores_all_iterations:
                  if len(set([s.split("_")[0] for s in error_scores_all_iterations.keys()])) == 1:
                    plt.plot(x_axis_values, error_scores_all_iterations[sample], linewidth=0.5)
                  else:
                    plt.plot(x_axis_values, error_scores_all_iterations[sample], linewidth=0.5, label=sample)
                if y_axis_upper_limit:
                  plt.gca().set_ylim([0, y_axis_upper_limit])
                plt.xlabel("num iterations")
                plt.ylabel(ylabel)
                plt.legend(loc="upper right")
                fig.savefig(filename, format='png', dpi=300)
 
            plot_error_scores(error_percentages_all_iterations, out_path_prefix + "error_percentages.png", 
                "percentages of samples with similarity score shifted from expected ordering")
            plot_error_scores(avg_error_scores_all_iterations, out_path_prefix + "avg_deviation.png", 
                "average similarity score deviation from expected ordering", y_axis_upper_limit=0.3)
            plot_error_scores(stdev_for_equal_scores_all_iterations, out_path_prefix + "stdev_for_equal_scores.png",
                "stdev of similarityies for sample pairs with same vocabulary intersection size")

            # Done with training.
            # final_embeddings = self.normalized_embeddings.eval()
            df = embeddings_to_dataframe(self.normalized_embeddings.eval(), sample_names, corpus)
            return df

        return final_embeddings
