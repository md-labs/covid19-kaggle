"""
no need to run this file as a sbatch command. It doesn't make use of GPU amd is very light weight.

1. Read the token embeddings generated by covid19-embeddings-simple-code.py/embeddings_from_fine_tuned_models.py
in "word_embeddings/" folder.
2. Read the seed embeddings generated by covid19-embeddings-simple-code.py/embeddings_from_fine_tuned_models.py
in "word_embeddings/" folder.
3. Read a file containing all the tokens present across all the abstracts from "inputs/ or fine_tuned_result/" folder
4. Writes a dictionary of the form {token: embedding} where <token> are those tokens that have a similar
embedding to the seed words in "similar_words/" folder.

word_embeddings/, inputs/, fine_tuned_result/ =======>  similar_words/
"""

import argparse
import glob
import pickle
import os
from collections import defaultdict
from queue import PriorityQueue

from sklearn.metrics.pairwise import cosine_similarity
import torch

import logging
logger = logging.getLogger(__name__)
logging.basicConfig(
  format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
  datefmt="%m/%d/%Y %H:%M:%S",
  level=logging.INFO)


def find_distance(func, list_1, list_2):
  return func(list_1, list_2)


def get_embeddings_for_seed_words(file_name):
  with open(file_name, "rb") as f:
    return pickle.load(f)


# returns the distance of <token> from all the seed words
def similarity_with_seeds(distance_metric, token, seed_to_embedding, token_to_embedding):
  dist = 0
  for seed, seed_embedding in seed_to_embedding.items():
    dist += distance_metric(seed_embedding.reshape(1, -1), token_to_embedding[token].reshape(1, -1))
  return dist


# takes the sharded dictionaries of {token: sum of embeddings} and converts them into a single dict
def get_combined_embeddings(token_to_embeddings_files, token_frequencies):
  combined_token_to_embeddings = defaultdict(torch.tensor)
  for token_to_embeddings_file in token_to_embeddings_files:
    with open(token_to_embeddings_file, "rb") as f:
      token_to_embeddings = pickle.load(f)
    for token, embedding in token_to_embeddings.items():
      if token in combined_token_to_embeddings:
        combined_token_to_embeddings[token] += embedding
      else:
        combined_token_to_embeddings[token] = embedding
  for token, embedding in combined_token_to_embeddings.items():
    combined_token_to_embeddings[token] = embedding / float(token_frequencies[token])
  return combined_token_to_embeddings



def main():
  parser = argparse.ArgumentParser()

  parser.add_argument(
    "--data_dir",
    type=str,
    required=True,
    help="The directory storing the word embeddings of the tokens (as python dictionary {token : embedding}) in pickle format")

  parser.add_argument(
    "--output_file",
    type=str,
    required=True,
    help="The directory storing the word embeddings of the tokens which are most similar (as python dictionary {token : embedding}) in pickle format")

  parser.add_argument(
    "--distance_metric",
    default="cosine",
    type=str,
    required=False,
    help="The distance metric to be used to find the similarity between the embeddings")

  parser.add_argument(
    "--k_similar",
    default=5,
    type=int,
    required=False,
    help="The number of similar words needed.")

  args = parser.parse_args()
  k = args.k_similar

  distance_metric = cosine_similarity
  if args.distance_metric == "cosine":
    distance_metric = cosine_similarity

  # stores only those tokens as the key which are in the priority queue
  token_to_embedding_pq_words = {}

  token_to_embeddings_files = glob.glob(f"word_embeddings/{args.data_dir}/word_embeddings_averaged_*.pickle")
  seed_to_embedding = get_embeddings_for_seed_words(os.path.join("word_embeddings", args.data_dir, "seed_embeddings_averaged.pickle"))

  pq = PriorityQueue(k)

  logger.info("seed tokens are %s", str(list(seed_to_embedding.keys())))

  with open(f"word_embeddings/{args.data_dir}/token_count.pickle", "rb") as f:
    token_frequencies = pickle.load(f)

  combined_token_to_embeddings = get_combined_embeddings(token_to_embeddings_files, token_frequencies)

  '''
  for every token:
    find cosine similarity with the seed tokens
    keep the k most similar words
  '''
  for token, embedding in list(combined_token_to_embeddings.items()):
    # don't consider seed words
    # if token in seed_to_embedding:
    #   continue

    if pq.full() is True:
      most_far_word_similarity, most_far_word = pq.get()
      token_to_embedding_pq_words.pop(most_far_word)

      similarity_current_token = similarity_with_seeds(distance_metric, token, seed_to_embedding, combined_token_to_embeddings)
      if similarity_current_token < most_far_word_similarity:
        pq.put((most_far_word_similarity, most_far_word))
        token_to_embedding_pq_words[most_far_word] = combined_token_to_embeddings[most_far_word]
      else:
        pq.put((similarity_current_token, token))
        token_to_embedding_pq_words[token] = embedding

    else:
      similarity_current_token = similarity_with_seeds(distance_metric, token, seed_to_embedding, combined_token_to_embeddings)
      pq.put((similarity_current_token, token))
      token_to_embedding_pq_words[token] = embedding

  with open(f'similar_words/{args.output_file}.pickle', 'wb') as handle:
    pickle.dump(token_to_embedding_pq_words, handle, protocol=pickle.HIGHEST_PROTOCOL)

  while pq.empty() is False:
    dist, word = pq.get()
    logger.info("%s %f", word, dist)


if __name__ == "__main__":
  main()
