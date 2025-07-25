from yake import KeywordExtractor
import numpy as np

def keyword_extraction(
    text,
    language = "en",
    max_ngram_size = 2,
    deduplication_thresold = 0.85,
    deduplication_algo = 'seqm',
    windowSize = 30,
    numOfKeywords = 25):

    kw_extractor = KeywordExtractor(
        lan=language,
        n=max_ngram_size,
        dedupLim=deduplication_thresold,
        dedupFunc=deduplication_algo,
        windowsSize=windowSize,
        top=numOfKeywords)
    return kw_extractor.extract_keywords(text)

def overlap_penalty(extracted_keywords, jaccard_threshold = 0.7):
  keywords = [word[0].lower() for word in extracted_keywords]
  num_keywords = len(keywords)

  if num_keywords <= 1:
    return 0.0

  penalty = 0

  # Created for Normalization
  max_keyword_combination = (num_keywords * (num_keywords- 1)) / 2

  # Handelling for Empty combination and preventing divide by Zero Error
  if max_keyword_combination == 0:
    return 0.0

  for i, ki in enumerate(keywords):
    for j, kj in enumerate(keywords):

      # Preventing repeats like for (A, B) but not (B, A)
      if i >= j: continue

      if ki in kj or kj in ki:
        penalty += 1
        continue

      # Create Substrings for comparisons
      words1 = set(ki.split())
      words2 = set(kj.split())

      # Handel Empty Strings
      if not words1 or not words2: continue

      intersection = len(words1.intersection(words2))
      union = len(words1.union(words2))

      # Handel Empty Strings and prevent divide by Zero Error
      if union == 0:
        continue

      jaccard_similarity = intersection / union

      if jaccard_similarity >= jaccard_threshold: penalty += 1

  # Normalizing the values to not skew the final score by alot
  normalized_penalty = 1 - (penalty / max_keyword_combination)

  return normalized_penalty

def n_gram_optimizer(text, n_grams = [2, 3, 4]):
    # Collect the Extracted keywords
    extracted_keywords = []
    # Avg Yake Score
    average_yake_score = []
    # Overrlap Penalty
    overlap = []

    # Adjusted Weights
    YAKE_SCORE_WEIGHT = 1.0
    OVERLAP_PENALTY_WEIGHT = 1.0

    # Loop through all the n_grams
    for index, n_gram in enumerate(n_grams):
        # Extract Keywords for a specific n_gram
        extracted_keywords.append(keyword_extraction(
        text = text,
        language = "en",
        max_ngram_size = n_gram,
        deduplication_thresold = 0.85,
        deduplication_algo = 'seqm',
        windowSize = 30,
        numOfKeywords = 25)
        )
        # Calculate the Avg Yake Score for that specific n_gram
        yake_score = np.average([score for kw, score in extracted_keywords[index]])
        # Normalizing the Yake scores for Edge Cases
        if yake_score < 0 or yake_score > 1:
            yake_score = 1
        # Subtracting to create Goodness score
        average_yake_score.append(np.clip(1 - yake_score, 0.0, 1.0))
        # Calculating the overlap penalty
        overlap.append(overlap_penalty(extracted_keywords[index]))

    # Calculating the goodness Score
    ####### INCOMPLETE #######
    ####### REQUIRES THE USE OF WEIGHTS ######
    goodness_score = [average_yake_score[i] * overlap[i] for i in range(len(n_grams))]

    return goodness_score