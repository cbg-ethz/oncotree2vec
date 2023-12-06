import pandas as pd
import sys

def parse_metadata_rob_trees(samples):

  metadata = pd.read_csv("../data/modes_of_evolution/metadata_evolution_trees.csv", sep = ',').fillna("").applymap(str)

  metadata_colors = {}
  metadata_colors["cancer_type"] = {
      'AML': '#cc3b2e',
      'uveal melanoma': '#a0c0dd',
      'braca': '#cdbfcf',
      'non-small-cell lung cancer': '#f9cb9c',
      'ccRCC': '#7b9a4a',
      'mesothelioma': '#008080'}

  color_map = {sample:"white" for sample in samples}
  for sample in samples:
    met = metadata.loc[metadata['sample'] == sample]
    color_map[sample] = metadata_colors["cancer_type"][met["cancer_type"].item()]    

  color_series = pd.Series(color_map)
  color_series.name = "cancer_type"

  return pd.concat([color_series], axis=1), metadata_colors
