import sys
import statistics

class TreeSample:
  # init method or constructor
  def __init__(self, sample_name, distance_to_reference, vocabulary_intersection_to_reference):
    self.sample_name = sample_name
    self.similarity_to_reference = 1-distance_to_reference
    self.vocabulary_intersection_to_reference = vocabulary_intersection_to_reference
    self.rank = len(vocabulary_intersection_to_reference)

  def print(self):
    print(self.sample_name, self.similarity_to_reference, self.rank)

# Compute deviation to correct order.
def compute_rank_deviation(scores_sorted_by_rank):
  ranks = [item.rank for item in scores_sorted_by_rank]
  unique_ranks = sorted(set(ranks), reverse=True)
  map_previous_rank = {}
  map_previous_rank[unique_ranks[0]] = unique_ranks[0]
  for idx in range(1,len(unique_ranks)):
    map_previous_rank[unique_ranks[idx]] = unique_ranks[idx-1]
   
  current_similarity_scored = [item.similarity_to_reference for item in scores_sorted_by_rank]

  min_rank_score = {}
  for rank in ranks:
    min_rank_score[rank] = min([item.similarity_to_reference for item in scores_sorted_by_rank if item.rank == rank])
  min_rank_score[ranks[0]] = current_similarity_scored[0]

  cummulated_error = 0
  cnt = 0
  reference_max_values = [min_rank_score[map_previous_rank[rank]] for rank in ranks]

  print(reference_max_values)

  # Each score should be lower than the minimum score from the previous rank.
  for idx in range(0, len(current_similarity_scored)):
    difference = current_similarity_scored[idx] - reference_max_values[idx]
    if difference > 0:
      cummulated_error = cummulated_error + difference
      cnt = cnt + 1

  avg_error = 0
  if cnt != 0:
    avg_error = cummulated_error / cnt

  error_percentage = cnt/len(scores_sorted_by_rank)

  return error_percentage, avg_error

def get_sorted_scores(df_distances, df_vocabulary_intersections):
  prefixes = set([sample.split('_')[0] for sample in df_distances.index])

  print("prefixes", prefixes)

  compare_all_trees = False
  if len(prefixes) == 1:
    prefixes = df_distances.index
    compare_all_trees = True

  prefixes = sorted(list(prefixes))

  # Compute score map.
  score_map = {}
  row_names = []
  cummulated_error_scores = {}
  avg_error_scores = {}
  stdev_for_equal_scores = {}

  for sample in prefixes:
    if compare_all_trees:
      sample_trees = list(df_distances.index)
    else:
      sample_trees = [s for s in df_distances.index if sample in s]

    scores = []
    for tree in sample_trees:
      if tree == sample:
        continue
      distance_to_reference = df_distances[sample][tree]
      vocabulary_intersection_to_reference = df_vocabulary_intersections[sample][tree]
      tree_variation = TreeSample(tree, distance_to_reference, vocabulary_intersection_to_reference)
      scores.append(tree_variation)
    scores.sort(key=lambda x:(x.rank, x.similarity_to_reference), reverse=True)
    score_map[sample] = scores

    # Get the stddev for the distance scores of the samples with the same rank as the reference sample.
    print("scores", scores)
    max_rank = max([item.rank for item in scores])
    stdev_for_equal_scores[sample] = statistics.pstdev([x.similarity_to_reference for x in scores if x.rank == max_rank])

    cummulated_error_scores[sample], avg_error_scores[sample] = compute_rank_deviation(scores)

  return score_map, cummulated_error_scores, avg_error_scores, stdev_for_equal_scores


