import pandas as pd

def clean_texts_fr(texts: pd.Series) -> pd.Series:
    """Clean texts by removing accents and special characters, and lowercasing.

    Args:
        texts(pd.Series): Series of texts

    Returns:
        texts_clean(pd.Series): Series of cleaned texts

    """
    texts_clean = (texts
                   .str.normalize('NFKD')
                   .str.encode('ascii', errors='ignore')
                   .str.decode('utf-8')
                   .str.lower()
                   )

    return texts_clean