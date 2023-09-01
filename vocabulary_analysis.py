from Levenshtein import distance
import numpy as np
import pandas as pd
from tqdm import tqdm

from joblib import Parallel, delayed

def parallel_distance(s1, list2):
    distances = [distance(s1, s2, score_cutoff=distance_threshold) for s2 in list2]

    return [np.argpartition(distances, 1)[1],
            np.partition(distances, 1)[1]]


def compute_closest_words(voc1, voc2=None, threshold=None):
    """Compute closest words.

        Args:
            voc1(List[str]): List of words
            voc2(List[str]): List of words (voc1 is compared to voc2 if provided)

        Returns:
            closest_words(pd.DataFrame): DataFrame of closest word from voc1 in voc1 (voc2) with associated distance
    """

    global distance_threshold
    distance_threshold = threshold

    if not voc2:
        voc2 = voc1

    distances = Parallel(n_jobs=20)(delayed(parallel_distance)(voc1[i], voc2) for i in tqdm(range(len(voc1))))

    closest_words = (pd.DataFrame(distances)
                     .merge(right=pd.Series(voc1,
                                            name="word_voc1"),
                            left_index=True,
                            right_index=True,
                            how="left")
                     .merge(right=pd.Series(voc1,
                                            name="closest_word_voc2"),
                            left_on=0,
                            right_index=True,
                            how="left")
                     .drop(columns=[0])
                     .rename(columns={1: "distance"})
                     )

    return closest_words