import os
import graphviz
import re

INFINITY: int = 2 ** 64 - 1
MAX_STUDIED_SENTENCES: int = INFINITY

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


def reduce(v: str, index: bool) -> list[str]:
    """

    :param index: Is true if word v is at the end of the sentence. :type index: bool :param v: Word of a sentence
    :type v: str :rtype: list[str] :return: List containing the different parts of speech in the considered word.
    This is often the word itself, but punctuation can come in if the world is at the end of the sentence.
    """
    if len(v) == 0:
        return []
    if v[0] in ["„"]:
        return [v[0]] + reduce(v[1:], index)
    if v[-1] == "\n":
        return reduce(v[:-1], index)
    if v[-1] in [",", "?", "!", "…", "“"]:
        return reduce(v[:-1], index) + [v[-1]]
    elif v[-3:] == "...":
        return reduce(v[:-3], index) + ["..."]
    elif v[-1] in ["."] and index:
        return reduce(v[:-1], index) + [v[-1]]
    else:
        return [v]


def treeifier(filename, ud_reldep=None, grammar_feature=None, out=None):
    """
    Takes a UD-Treebank filename (in reality we use a CLI parameter such that `deep/filename/all.conllu` exists), and multiple optional arguments to prepare stats, and returns the trees in the treebank.
    :param filename: .conllu file containing a UD-Treebank
    :param ud_reldep: UD_RelDep we want to get all the representatives from. None if we don't want to.
    :param grammar_feature: tuple containing a grammatical feature name (e.g. Case, Tense, Person), and an associated value of which we want the representatives.
    :param out: Directory name in which to drop the results
    :return: Stores in multiple files in out directory the trees which are in the .conllu files, the associated DAGs, and representatives of the ud_reldep and grammar_feature.
    """
    # try :
    #     with open(out + ".txt", "r") as f:
    #         feature_dicts = f.readlines()
    #     trees = [str_to_dict(f) for f in feature_dicts]
    #     for t in trees:
    #         print(t)
    #         for w in t:
    #             t[w] = str_to_dict(t[w] + "\n")
    #     with open(out + "/UD_RelDep/" + ud_reldep + ".txt", "w") as f:
    #         for t in trees:
    #             for w in t:
    #                 # noinspection PyTypeChecker
    #                 if is_prefix(ud_reldep, t[w]["edge_type"]):
    #                     f.write(str(t[w]) + "\n")
    # except FileNotFoundError:
    with open(filename, "r") as f:
        lines = f.readlines()
    trees = []
    tmp = []
    index = 0
    number_of_studied_sentences = 0
    number_of_cpt_sentences = 0
    while number_of_studied_sentences < MAX_STUDIED_SENTENCES and index < len(lines):
        l = lines[index]
        if l == "\n":
            trees.append(tmp)
            tmp = []
            number_of_studied_sentences += 1
        elif l[:8] == "# text =":
            tmp.append(l[9:])
        elif l[0] == "#":
            pass
        else:
            tmp.append(l)
        index += 1

    for tree in range(len(trees)):
        annotated_sentence = trees[tree]
        at_least_a_space_regex: re.Pattern[str] = re.compile(" +")
        annotated_sentence[0] = re.sub(at_least_a_space_regex, " ", annotated_sentence[0])
        sentence = annotated_sentence[0].split(" ")
        # sentence = sentence[:-1] + [sentence[-1][:-3], sentence[-1][-3]]
        tmp_sentence = []
        # TODO: Consider end of line words with dots in them, e.g. German `u.s.w.`
        for i in range(len(sentence)):
            v = sentence[i]
            tmp_sentence = tmp_sentence + reduce(v, i == len(sentence) - 1)

        sentence_dict = {}
        sentence = tmp_sentence
        word = 0
        # TODO: Consider the case where words are agglutinations: cf German `am` instead of `an dem` which is written
        #  on 3 different lines.
        while word < len(sentence):
            try:
                annotations = annotated_sentence[word + 1]
                annotations = annotations.split("\t")
                attributes = {
                    "position": word + 1,
                    "grapheme": annotations[1],
                    "lemma": annotations[2],
                    "part_of_speech": annotations[3],
                    "edge_type": annotations[7]}
                features = annotations[5]

                if annotations[6] != "0":
                    attributes["predecessor"] = annotations[6]
                else:
                    attributes["predecessor"] = str(word + 1)
                if features == "_":
                    pass
                else:
                    features = features.split("|")
                    for feature in features:
                        feature = feature.split("=")
                        attributes[feature[0]] = feature[1]
                sentence_dict[sentence[word]] = attributes
            except IndexError:
                number_of_cpt_sentences += 1
                # print(number_of_cpt_sentences)
            word += 1
        trees[tree] = sentence_dict
        graphical = graphviz.Digraph()
        for word in sentence_dict:
            word = sentence_dict[word]

            graphical.node(str(word["position"]), word["grapheme"])

        for word in sentence_dict:
            word = sentence_dict[word]
            graphical.edge(str(word["position"]), str(word["predecessor"]), label=str(word["edge_type"]))
        trees[tree] = (trees[tree], graphical)

    if out is None:
        print([str(t[0]) for t in trees])
    else:
        with open(out + ".txt", "w") as f:
            for i in range(len(trees)):
                t = trees[i][0]
                f.write(str(i) + " : " + str(t) + "\n")
        for i in range(len(trees)):
            with open(out + "/Graph_Sources/" + str(i) + ".dot", "w") as f:
                f.write(str(trees[i][1]))
        if ud_reldep is not None:
            with open(out + "/UD_RelDep/" + ud_reldep + ".txt", "w") as f:
                for t in trees:
                    for w in t[0]:
                        # noinspection PyTypeChecker
                        if is_prefix(ud_reldep, t[0][w]["edge_type"]):
                            f.write(str(t[0][w]) + "\n")
        if grammar_feature is not None:
            with open(out + "/Features/" + f"{grammar_feature[0]}={grammar_feature[1]}" + ".txt", "w") as f:
                for t in trees:
                    for w in t[0]:
                        # noinspection PyTypeChecker
                        if is_prefix(grammar_feature[1], t[0][w][grammar_feature[0]]):
                            f.write(str(t[0][w]) + "\n")
    return number_of_cpt_sentences


def show_graph(index, directory, view=False):
    s = graphviz.Source.from_file(directory + f"/Graph_Sources/{index}.dot")
    s.render(directory + f"/Graphs/{index}", format="pdf")
    try:
        os.remove(directory + f"/Graphs/{index}")
    except FileNotFoundError:
        pass
    if view:
        s.view()
