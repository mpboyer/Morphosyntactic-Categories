import os
import argparse
import conllu_parser
import word2vec

parser = argparse.ArgumentParser()
parser.add_argument("filename", help=".conllu file containing a UD-Treebank")
parser.add_argument("-p", "--property", help="UD-RELDEP which we're studying, e.g. obl:tmod")
parser.add_argument("-f", "--feature", help="Morphosyntactic category we want to study.")
parser.add_argument("-o", "--out", help="File in which to store the result", required=False)
parser.add_argument("-i", "--index",
                    help=f"Number of the sentences to study. Must be smaller than MAX_STUDIED_SENTENCES={conllu_parser.MAX_STUDIED_SENTENCES}",
                    required=False)
args = parser.parse_args()

if __name__ == "__main__":
    try:
        os.mkdir(args.filename)
    except FileExistsError:
        pass
    out = "deep/" + args.filename + "/" + args.out
    try:
        os.mkdir(out)
    except FileExistsError:
        pass
    for c in ["/Graph_Sources", "/Graphs", "/Properties"]:
        try:
            os.mkdir(out + c)
        except FileExistsError:
            pass
    conllu_parser.treeifier("deep/" + args.filename + "/all.conllu", args.property, out=out)
    if args.out and args.index:
        number_of_sentences = len(os.listdir(out + f"/Graph_Sources"))
        for n in range(number_of_sentences):
            conllu_parser.show_graph(n, "deep/" + args.filename + "/" + args.out)
    print(word2vec.frequency_on_case(out + f"/Properties/{args.property}.txt"))
