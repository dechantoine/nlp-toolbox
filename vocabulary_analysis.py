from Levenshtein import distance
import numpy as np
import pandas as pd

def compute_closest_words(voc1, voc2=None):
    """Compute closest words.

        Args:
            voc1(array-like): Array of words
            voc2(array-like): Array of words (voc1 is compared to voc2 if provided)

        Returns:
            closest_words(pd.DataFrame): DataFrame of the closest word from voc1 in voc1 (voc2) with associated distance
    """

    if not voc2:
        voc2 = voc1

    distance_func = distance

    distances = [[voc1[i],
                  voc2[j],
                  distance_func(voc1[i],
                                voc2[j])]
                 for j in range(len(voc1))
                 for i in range(j if np.array_equal(voc1, voc2)
                                else len(voc2))]

    closest_words = (pd.DataFrame(data=distances,
                                 columns=["word1", "word2", "distance"])
                    .sort_values(by=["word1", "distance"],
                                 ascending=True)
                    .drop_duplicates(subset=["word1"])
                    )

    return closest_words