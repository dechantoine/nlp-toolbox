import pandas as pd
from text_cleaning import clean_texts_fr
from more_itertools import sliding_window, windowed_complete


def n_wise(texts: pd.Series, n: int, replacements: dict = None, stopwords: list[str] = None) -> pd.Series:
    """Extracts unique n-grams with counts from a series of texts

    Args:
        texts(pd.Series): Series of texts
        n(int): Number of words in n-gram
        replacements(dict): Dictionary of replacements to be made in the texts, e.g. for misspellings or abbreviations. Dict should be of the form {"misspelled_word": "correct_word"}
        stopwords(list): List of stopwords to be removed from the texts

    Returns:
        n_wise(pd.Series): Series of unique n-grams with their counts

    """
    n_grams = (clean_texts_fr(texts)
               .str.split(" ")
               )
    if stopwords:
        n_grams = (n_grams
                   .apply(lambda x: [k for k in x if k not in stopwords])
                   )

    if replacements:
        n_grams = (n_grams.
                   apply(lambda x: [k if k not in replacements.keys()
                                    else replacements[k]
                                    for k in x])
                   )

    n_grams = (n_grams
               .apply(lambda x: list(sliding_window(x, n)))
               .explode()
               .dropna()
               .value_counts()
               .rename(index="count")
               .rename_axis(index="n_gram")
               )

    return n_grams


def beginning_middle_end(n_grams: pd.Series) -> pd.DataFrame:
    """ Cut n-grams into (beginning + n-1 gram) or (n-1 gram + end) and count them

        Args:
            n_grams(pd.Series): Series of unique n-grams with their counts
        Returns:
            b_m_e(pd.DataFrame): DataFrame of n-grams cut into (beginning + n-1 gram) or (n-1 gram + end) with their counts

    """
    n = len(n_grams.index[0])

    b_m_e = (pd.concat([(n_grams
                         .reset_index()
                         # .rename(columns={"index": "n_gram"})
                         ),
                        (pd.Series(n_grams
                                   .index)
                         .apply(lambda x: list(windowed_complete(x, n - 1)))
                         .rename("n-1_gram")
                         )
                        ],
                       axis=1)
             .explode(column="n-1_gram")
             )

    b_m_e = (pd.concat([pd.DataFrame([list(x)
                                      for x in b_m_e["n-1_gram"].values]),
                        (b_m_e[["n_gram", "count"]]
                         .reset_index(drop=True))
                        ],
                       axis=1,
                       ignore_index=True)
             .rename(columns={0: "begin",
                              1: "middle",
                              2: "end",
                              3: "n_gram",
                              4: "count"})
             .applymap(lambda x: None if x == tuple() else x)
             )

    return b_m_e