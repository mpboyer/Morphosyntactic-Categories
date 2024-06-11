import os
import graphviz
import argparse

INFINITY: int = 2**64 - 1
MAX_STUDIED_SENTENCES: int = 10

parser = argparse.ArgumentParser()
parser.add_argument("filename", help=".conllu file containing a UD-Treebank")
parser.add_argument("--out", help="File in which to store the result")
parser.add_argument("--index", help=f"Number of the sentences to study. Must be smaller than MAX_STUDIED_SENTENCES={MAX_STUDIED_SENTENCES}")
args = parser.parse_args()

def treeifier(filename, out = None):
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
        sentence = annotated_sentence[0].split(" ")
        # sentence = sentence[:-1] + [sentence[-1][:-3], sentence[-1][-3]]
        tmp_sentence = []
        for i in range(len(sentence)): 
            v = sentence[i]
            if v[-1] == "\n":
                v = v[:-1]
            if v[-1] in [",", ".", "?", "!", "…"]:
                tmp_sentence = sentence[:i] + [v[:-1], v[-1]] + sentence[i + 1:]
        
        sentence_dict = {}
        sentence = tmp_sentence
        for word in range(len(sentence)):
            annotations = annotated_sentence[word + 1]
            annotations = annotations.split("\t")
            attributes = {"position" : word, "grapheme": annotations[1], "lemma": annotations[2], "part_of_speech": annotations[3], "predecessor": annotations[6], "edge_type": annotations[7]}
            features = annotations[5]
            if features == "_":
                pass
            else: 
                features = features.split("|")
                for feature in features: 
                    feature = feature.split("=")
                    attributes[feature[0]] = feature[1]
            sentence_dict[sentence[word]] = attributes
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
            for t in trees:
                f.write(str(t[0])+"\n")
        try :
            os.mkdir(out) 
        except FileExistsError : pass
        for i in range(len(trees)):
            with open(out + "/" + str(i) + ".dot", "w") as f: 
                f.write(str(trees[i][1]))

def show_graph(index, directory, view=False):
    s = graphviz.Source.from_file(directory + f"/{index}.dot")
    s.render(directory + f"/{index}", format="pdf")
    try :
        os.remove(directory + f"/{index}")
    except FileNotFoundError : pass
    if view:
        s.view()

if __name__ == "__main__":
    treeifier(args.filename, out=args.out)    
    if args.out and args.index: 
        show_graph(args.index, args.out)
    


