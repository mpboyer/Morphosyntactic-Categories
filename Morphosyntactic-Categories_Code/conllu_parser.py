import os
import graphviz
import re

from tqdm import tqdm, trange

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


# def reduce(v: str, index: bool) -> list[str]:
#     """
#
#     :param index: Is true if word v is at the end of the sentence. :type index: bool :param v: Word of a sentence
#     :type v: str :rtype: list[str] :return: List containing the different parts of speech in the considered word.
#     This is often the word itself, but punctuation can come in if the world is at the end of the sentence.
#     """
#     if len(v) == 0:
#         return []
#     if v[0] in ["„"]:
#         return [v[0]] + reduce(v[1:], index)
#     if v[-1] == "\n":
#         return reduce(v[:-1], index)
#     if v[-1] in [",", "?", "!", "…", "“"]:
#         return reduce(v[:-1], index) + [v[-1]]
#     elif v[-3:] == "...":
#         return reduce(v[:-3], index) + ["..."]
#     elif v[-1] in ["."] and index:
#         return reduce(v[:-1], index) + [v[-1]]
#     else:
#         return [v]


def tree_ifier(filename, ud_reldep=None, grammar_feature=None, out=None, graphical=False):
    """
    Takes a UD-Treebank filename (in reality we use a CLI parameter such that `deep/filename/all.conllu` exists), and multiple optional arguments to prepare stats, and returns the trees in the treebank.
    :param filename: .conllu file containing a UD-Treebank
    :param ud_reldep: UD_RelDep we want to get all the representatives from. None if we don't want to.
    :param grammar_feature: tuple containing a grammatical feature name (e.g. Case, Tense, Person), and an associated value of which we want the representatives.
    :param out: Directory name in which to drop the results
    :return: Stores in multiple files in out directory the trees which are in the .conllu files, the associated DAGs, and representatives of the ud_reldep and grammar_feature.
    """
    with open(filename, "r") as f:
        lines = f.readlines()
    trees = []
    tmp = []
    index = 0
    number_of_studied_sentences = 0
    number_of_cpt_sentences = 0
    # pbar = tqdm(total=len(lines), colour='#7d1dd3', leave=True)
    while number_of_studied_sentences < MAX_STUDIED_SENTENCES and index < len(lines):
        l = lines[index]
        if l == "\n":
            trees.append(tmp)
            tmp = []
            number_of_studied_sentences += 1
        elif l[0] == "#":
            pass
        else:
            tmp.append(l)
        index += 1
    #     pbar.update(1)
    # pbar.close()

    # for tree in trange(len(trees), colour='#7d1dd3', leave=True):
    for tree in range(len(trees)):
        sentence = trees[tree]
        sentence_dict = {}
        word = 0
        offset = 1  # Equal to the difference between the position of the word in the sentence (considered from the
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
                # print(number_of_cpt_sentences)
            word += 1
        trees[tree] = (sentence_dict, "")

        if graphical:
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
        if graphical:
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
                        if is_prefix(grammar_feature[1], t[0][w].get(grammar_feature[0], "")):
                            f.write(str(t[0][w]) + "\n")
    return number_of_cpt_sentences


def show_graph(index, directory, view=False):
    """
    Renders a graph from a directory containing indexed dot files.
    :param index: Index of the file in Graph_Sources
    :param directory: Directory containing Graph_Sources
    :param view: If [default=False], show the resulting graph.
    :return: Writes the graph in .pdf in the directory.
    """
    s = graphviz.Source.from_file(directory + f"/Graph_Sources/{index}.dot")
    s.render(directory + f"/Graphs/{index}", format="pdf")
    try:
        os.remove(directory + f"/Graphs/{index}")
    except FileNotFoundError:
        pass
    if view:
        s.view()
