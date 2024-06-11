import os
import graphviz
import argparse
import re

INFINITY: int = 2 ** 64 - 1
MAX_STUDIED_SENTENCES: int = 10000

parser = argparse.ArgumentParser()
parser.add_argument("filename", help=".conllu file containing a UD-Treebank")
parser.add_argument("-p", "--property", help="UD-RELDEP which we're studying, e.g. obl:tmod")
parser.add_argument("-o", "--out", help="File in which to store the result", required=False)
parser.add_argument("-i", "--index",
                    help=f"Number of the sentences to study. Must be smaller than MAX_STUDIED_SENTENCES={MAX_STUDIED_SENTENCES}",
                    required=False)
args = parser.parse_args()


def is_prefix(a, b):
    if len(a) > len(b):
        return False
    for i in range(len(a)):
        if a[i] != b[i]:
            return False
    return True


def treeifier(filename, out=None):
    with open(filename, "r") as f:
        lines = f.readlines()
    trees = []
    tmp = []
    index = 0
    t = 0
    while t < MAX_STUDIED_SENTENCES and index < len(lines):
        l = lines[index]
        if l == "\n":
            trees.append(tmp)
            tmp = []
            t += 1
        elif l[:6] == "# text":
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
        for i in range(len(sentence)):
            v = sentence[i]
            if v[-1] == "\n":
                v = v[:-1]
            if v[-1] in [",", "?", "!", "…"]:
                tmp_sentence = tmp_sentence + [v[:-1], v[-1]]
            elif v[-1] == "." and i == len(sentence) - 1:
                tmp_sentence = tmp_sentence + [v[:-1], v[-1]]
            else:
                tmp_sentence = tmp_sentence + [v]

        sentence_dict = {}
        sentence = tmp_sentence
        for word in range(len(sentence)):
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
                print(sentence, word + 1, tree)
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
        return [str(t[0]) for t in trees]
    else:
        with open(out + ".txt", "w") as f:
            for i in range(len(trees)):
                t = trees[i][0]
                f.write(str(i) + " : " + str(t) + "\n")
        for i in range(len(trees)):
            with open(out + "/Graph_Sources/" + str(i) + ".dot", "w") as f:
                f.write(str(trees[i][1]))
        with open(out + "/Properties/" + args.property + ".txt", "w") as f:
            for t in trees:
                for w in t[0]:
                    # noinspection PyTypeChecker
                    if is_prefix(args.property, t[0][w]["edge_type"]):
                        f.write(str(t[0][w]) + "\n")


def show_graph(index, directory, view=False):
    s = graphviz.Source.from_file(directory + f"/Graph_Sources/{index}.dot")
    s.render(directory + f"/Graphs/{index}", format="pdf")
    try:
        os.remove(directory + f"/Graphs/{index}")
    except FileNotFoundError:
        pass
    if view:
        s.view()


if __name__ == "__main__":
    try:
        os.mkdir(args.filename)
    except FileExistsError:
        pass
    out = "deep/" + args.filename + "/" + args.out
    for c in ["", "Graph_Sources", "Graphs", "Properties"]:
        try:
            os.mkdir(out + f"/{c}/")
        except FileExistsError:
            pass
    treeifier("deep/" + args.filename + "/all.conllu", out=out)
    if args.out and args.index:
        number_of_sentences = len(os.listdir(out + f"/Graph_Sources"))
        for n in range(number_of_sentences):
            show_graph(n, "deep/" + args.filename + "/" + args.out)
