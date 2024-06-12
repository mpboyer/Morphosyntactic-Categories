from collections import OrderedDict


def str_to_dict(sequence: str) -> dict[str, str | int]:
    """Sequence must end by \n"""
    features = {}
    sequence = sequence[1:-2]
    sequence = sequence.split(", ")
    for w in sequence:
        feature = w.split(": ")
        if len(feature) == 1:
            print(feature)
            pass
        elif len(feature[1]) > 1:
            features[feature[0][1:-1]] = feature[1][1:-1]
        else:
            features[feature[0][1:-1]] = feature[1]
    return features


def sorter(t):
    if t[0] == "Other_Word_Type":
        return -1
    return t[1]


def format_case_stats(d):
    return OrderedDict(sorted(d.items(), key=sorter, reverse=True))


def format_property_stats(d):
    for f in d:
        d[f] = format_case_stats(d[f])
    return d


def frequency_on_grammar_feature_checking_reldep(filename, grammar_feature):
    with open(filename, 'r') as f:
        features = f.readlines()
    features = list(map(lambda t: str_to_dict(t), features))
    frequency_dict = {}
    other_types = {}
    for word in features:
        c = word.get(grammar_feature, 'Other_Word_Type')
        if c == 'Other_Word_Type':
            for f in word:
                if f not in ['predecessor', 'position', 'grapheme', 'lemma', 'edge_type']:
                    if f not in other_types:
                        other_types[f] = {}
                    v = str(word[f])
                    other_types[f][v] = other_types[f].get(v, 0) + 1
        frequency_dict[c] = frequency_dict.get(c, 0) + 1
    dom = max(frequency_dict, key=frequency_dict.get, default='')
    return frequency_dict, other_types
