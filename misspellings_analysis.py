import pandas as pd

from metaphone import doublemetaphone
from Levenshtein import distance

def distance_double_metaphone(word1, word2):
    """Compute the distance between two words using the Double Metaphone algorithm

        Args:
            word1(str): First word
            word2(str): Second word

        Returns:
            distance(float): Distance between the two words
    """
    distances = []
    for uni1 in doublemetaphone(word1):
        for uni2 in doublemetaphone(word2):
            if len(uni1) > 0 and len(uni2) > 0:
                distances.append(distance(uni1, uni2))
    return sum(distances)/len(distances)


def compute_misspelled_words(voc, distance_func, threshold=1):
    """Compute misspelled words.

        Args:
            voc(array-like): Array of words sort by frequency
            distance_func(function): Function to compute the distance between two words
            threshold(float): Threshold to determine if a word is misspelled. Depends on the distance function used.

        Returns:
            misspelled_words(dict): Dictionary of (supposedly) correct words with their closest mispelled words
    """
    distances = [[voc[i],
                  voc[j],
                  distance_func(voc[i],
                                voc[j])]
                 for j in range(len(voc))
                 for i in range(0, j)]

    df_distances = pd.DataFrame(data=distances,
                                columns=["word1", "word2", "distance"])

    misspelled_words = (df_distances[df_distances.distance <= threshold]
                        .drop_duplicates(subset="word2")
                        .reset_index(drop=True))[["word1", "word2"]]

    return misspelled_words