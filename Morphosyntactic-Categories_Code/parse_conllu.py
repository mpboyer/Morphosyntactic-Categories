import os
import argparse
import re

parser = argparse.ArgumentParser()
parser.add_argument("-fi", "--filename", help=".conllu file containing a UD-Treebank.", required=True)
parser.add_argument("-v", "--verbose", required=False)
parser.add_argument("-p", "--property", help="UD-RELDEP which we're studying.", required=False)
parser.add_argument("-f", "--feature_type", help="Morphosyntactic category we want to study.", required=True)
parser.add_argument("-val", "--value", help="Value of the studied feature", required=True)
args = parser.parse_args()


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


def vectorize(filename, grammar_feature, ud_reldep: str = None, out=None):
    """
    Takes a UD-Treebank filename (in reality we use a CLI parameter such that `deep/filename/all.conllu` exists),
    and multiple optional arguments to prepare stats, and returns the trees in the treebank.
    :param filename: .conllu
    file containing a UD-Treebank
    :param ud_reldep: UD_RelDep we want to get all the representatives from. None if we
    don't want to.
    :param grammar_feature: tuple containing a grammatical feature name (e.g. Case, Tense, Person),
    and an associated value of which we want the representatives.
    :param out: Directory name in which to drop the results. Acts as a boolean in the function.
    :return: Stores in multiple files in out directory the trees which are in the .conllu files,
    the associated DAGs, and representatives of the ud_reldep and grammar_feature.
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

    property_matches = 0
    trees = []

    for sentence in sentences:
        sentence_dict = {}
        word = 0
        offset = 1
        # Equal to the difference between the position of the word in the sentence (considered from the
        # parsing, i.e. including `am = an dem` as two words) minus the index in the list of parsed words.
        while word < len(sentence):
            try:
                annotations = sentence[word]
                annotations = annotations.split("\t")
                c = re.compile("-")
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
                    if annotations[7] == ud_reldep:
                        property_matches += 1

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
                    sentence_dict[attributes["position"]] = attributes

            except IndexError:
                number_of_cpt_sentences += 1
                print(number_of_cpt_sentences)
            word += 1
        trees.append(sentence_dict)

    grammar_feature_for_reldep = (f"We have studied {number_of_studied_sentences} sentences and failed on "
                                  f"{number_of_cpt_sentences} in treebank `{filename}`.\n"
                                  f"Of those, {property_matches} words match `{ud_reldep}`.\n")
    res_dict = {}
    for t in trees:
        for w in t:
            if is_prefix(ud_reldep, t[w]["edge_type"]):
                if grammar_feature[0] in t[w]:
                    res_dict[t[w][grammar_feature[0]]] = res_dict.get(t[w][grammar_feature[0]], 0) + 1
            grammar_feature_for_reldep += (
                f"We get the following distribution of values for {grammar_feature[0]} "
                f"matching `{ud_reldep}`:\n")
    for v in res_dict:
        grammar_feature_for_reldep += f"\t{v}: {res_dict[v]}\n"

    reldep_for_grammar_feature = ""
    reldep_for_grammar_feature += (
        f"We have studied {number_of_studied_sentences} sentences and failed on {number_of_cpt_sentences} in"
        f"treebank `{filename}`.\n We get the following distribution "
        f"of RelDep for words matching `{grammar_feature[0]}={grammar_feature[1]}`:\n")
    res_dict = {}
    for t in trees:
        for w in t:
            if is_prefix(grammar_feature[1], t[w].get(grammar_feature[0], "")):
                res_dict[t[w]["edge_type"]] = res_dict.get(t[w]["edge_type"], 0) + 1

    for v in res_dict:
        reldep_for_grammar_feature += f"RelDep {v}: {res_dict[v]}\n"

    if out:
        if ud_reldep:
            return grammar_feature_for_reldep
        return reldep_for_grammar_feature
    else:
        return grammar_feature_for_reldep, reldep_for_grammar_feature


if __name__ == "__main__":
    gf = (args.feature_type, args.value)
    if args.out:
        with open(args.out, "w") as f:
            f.write(vectorize(args.filename, gf, args.property, args.out))
