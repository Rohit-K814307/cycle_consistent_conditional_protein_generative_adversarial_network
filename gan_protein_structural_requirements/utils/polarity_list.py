def polarity_list():
    """
    Polarity Type Codes:

    0: non polar

    1: polar

    2: ionic
    """

    return {
        "A": {"name":"Alanine", "polarity_type":0},
        "R": {"name":"Arginine", "polarity_type":2},
        "N": {"name":"Asparagine", "polarity_type":1},
        "D": {"name":"Aspartic Acid", "polarity_type":2},
        "C": {"name":"Cysteine", "polarity_type":1},
        "E": {"name":"Glutamic Acid", "polarity_type":2},
        "Q": {"name":"Glutamine", "polarity_type":1},
        "G": {"name":"Glycine", "polarity_type":0},
        "H": {"name":"Histidine", "polarity_type":2},
        "I": {"name":"Isoleucine", "polarity_type":0},
        "L": {"name":"Leucine", "polarity_type":0},
        "K": {"name":"Lysine", "polarity_type":2},
        "M": {"name":"Methionine", "polarity_type":0},
        "F": {"name":"Phenylalanine", "polarity_type":0},
        "P": {"name":"Proline", "polarity_type":0},
        "S": {"name":"Serine", "polarity_type":1},
        "T": {"name":"Threonine", "polarity_type":1},
        "W": {"name":"Tryptophan", "polarity_type":0},
        "Y": {"name":"Tyrosine", "polarity_type":0},
        "V": {"name":"Valine", "polarity_type":0}
    }