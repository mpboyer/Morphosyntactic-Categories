def str_to_dict(sequence: str) -> dict[str, str | int]:
    """Sequence must end by \n"""
    features = {}
    sequence = sequence[1:-2]
    sequence = sequence.split(", ")
    for w in sequence:
        feature = w.split(": ")
        if len(feature[1]) > 1:
            features[feature[0][1:-1]] = feature[1][1:-1]
        else:
            features[feature[0][1:-1]] = feature[1]
    return features


def frequency_on_case(filename):
    with open(filename, 'r') as f:
        features = f.readlines()
    features = list(map(lambda t: str_to_dict(t), features))
    frequency_dict = {}
    for word in features:
        frequency_dict[word.get('Case', 'Other_Word_Type')] = frequency_dict.get(word.get('Case', 'Other_Word_Type'), 0) + 1
    dom = max(frequency_dict, key=frequency_dict.get, default='')
    return frequency_dict, dom
