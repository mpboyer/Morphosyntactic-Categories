import re


def is_prefix(s1: str, s2: str) -> bool:
    """
    Looks at len(s1) first chars in b to check if s1 is a prefix of s2.
    :rtype: bool
    :param s1: pattern
    :param s2: text
    :return: True iff s1 is a prefix of s2.
    """
    if len(s1) > len(s2):
        return False
    for i in range(len(s1)):
        if s1[i] != s2[i]:
            return False
    return True


def vectorize(filename):
    """
    Takes a UD-Treebank filename (in reality we use a CLI parameter such that `deep/filename/all.conllu` exists),
    and multiple optional arguments to prepare stats, and returns the trees in the treebank.
    :param filename: .conllu
    file containing a UD-Treebank
    :return: Stores in multiple files in out directory the vectors representing cases in the corpus.
    """
    with open(filename, "r") as f:
        lines = f.readlines()
    sentences = []
    tmp = []
    number_of_studied_sentences = 0
    number_of_cpt_sentences = 0
    for l in lines:
        if l == "\n":
            sentences.append(tmp)
            tmp = []
            number_of_studied_sentences += 1
        elif l[0] == "#":
            pass
        else:
            tmp.append(l)

    trees = []
    all_features = set()

    for sentence in sentences:
        sentence_dict = {}
        word = 0
        offset = 1
        # Equal to the difference between the position of the word in the sentence (considered from the
        # parsing, i.e. including `am = an dem` as two words) minus the index in the list of parsed words.
        c = re.compile("-")
        while word < len(sentence):
            annotations = sentence[word]
            annotations = annotations.split("\t")
            if re.search(c, annotations[0]):
                word += 1
                offset -= 1
            else:

                attributes = {
                    "position": word + offset,
                    "grapheme": annotations[1],
                    "lemma": annotations[2],
                    "part_of_speech": annotations[3],
                    "edge_type": annotations[7]}
                features = annotations[5]

                if annotations[6] != "0":
                    attributes["predecessor"] = annotations[6]
                else:
                    attributes["predecessor"] = str(word + offset)
                if features == "_":
                    pass
                else:
                    features = features.split("|")
                    for feature in features:
                        feature = feature.split("=")
                        attributes[feature[0]] = feature[1]
                        all_features.add((feature[0], feature[1]))
                sentence_dict[attributes["position"]] = attributes
            word += 1
        trees.append(sentence_dict)

    results = []
    for feature in all_features:
        reldep_for_grammar_feature = ""
        reldep_for_grammar_feature += (
            f"We have studied {number_of_studied_sentences} sentences and failed on {number_of_cpt_sentences} in "
            f"treebank `{filename}`.\n We get the following distribution "
            f"of RelDep for words matching `{feature[0]}={feature[1]}`:\n")
        res_dict = {}
        for t in trees:
            for w in t:
                if is_prefix(feature[1], t[w].get(feature[0], "")):
                    res_dict[t[w]["edge_type"]] = res_dict.get(t[w]["edge_type"], 0) + 1

        for v in res_dict:
            reldep_for_grammar_feature += f"RelDep {v}: {res_dict[v]}\n"

        results.append((feature, reldep_for_grammar_feature))

    return results


if __name__ == "__main__":
    for c in vectorize(f"../ud-treebanks-v2.14/UD_Russian-GSD/ru_gsd-ud-dev.conllu")[:1]:
        print(c[1])
