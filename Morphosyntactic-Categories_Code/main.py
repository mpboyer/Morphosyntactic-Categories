import os
import argparse
import conllu_parser
import statifier

parser = argparse.ArgumentParser()
parser.add_argument("filename", help=".conllu file containing a UD-Treebank. Use all to iterate on all banks.")
parser.add_argument("-v", "--verbose", required=False)
parser.add_argument("-p", "--property", help="UD-RELDEP which we're studying, e.g. obl:tmod")
parser.add_argument("-f", "--feature_type", help="Morphosyntactic category we want to study.")
parser.add_argument("-val", "--value", help="Value of the studied feature")
parser.add_argument("-o", "--out", help="File in which to store the result", required=False, default="Sentence_Graphs")
parser.add_argument("-i", "--index",
                    help=f"Number of the sentences to study. Must be smaller than MAX_STUDIED_SENTENCES={conllu_parser.MAX_STUDIED_SENTENCES}",
                    required=False)
args = parser.parse_args()

if __name__ == "__main__":
    grammar_feature = None
    if args.feature_type and args.value:
        grammar_feature = (args.feature_type, args.value)
    if args.feature_type and args.verbose:
        if args.filename == 'all':
            treebanks = os.listdir("deep/")
            right_type_dict = {}
            other_type_dict = {}
            number_of_sentences = 0
            property_matches = 0
            for t in treebanks:
                cpt = 0
                out = "deep/" + t + "/" + args.out
                try:
                    os.mkdir(out)
                except FileExistsError:
                    pass
                for c in ["/Graph_Sources", "/Graphs", "/UD_RelDep", "/Features"]:
                    try:
                        os.mkdir(out + c)
                    except FileExistsError:
                        pass
                try:
                    with open(out + f"/UD_RelDep/{args.property}.txt", "r") as f:
                        pass
                except FileNotFoundError:
                    cpt += conllu_parser.treeifier("deep/" + t + "/all.conllu", ud_reldep=args.property,
                                        grammar_feature=grammar_feature, out=out)
                print(t)
                n = len(os.listdir(f"deep/{str(t)}/{args.out}/Graph_Sources"))
                number_of_sentences += n
                d = statifier.frequency_on_grammar_feature_checking_reldep(
                    f"deep/{t}/{args.out}/UD_RelDep/{args.property}.txt", args.feature_type)
                property_matches += sum(map(lambda l: l[1], d[0].items()))
                for v in d[0]:
                    right_type_dict[v] = right_type_dict.get(v, 0) + d[0][v]
                for feature in d[1]:
                    if feature not in other_type_dict:
                        other_type_dict[feature] = {}
                    for value in d[1][feature]:
                        other_type_dict[feature][value] = other_type_dict[feature].get(value, 0) + d[1][feature][value]
        else :
            out = "deep/" + args.filename + "/" + args.out
            try:
                os.mkdir(out)
            except FileExistsError:
                pass
            for c in ["/Graph_Sources", "/Graphs", "/UD_RelDep", "/Features"]:
                try:
                    os.mkdir(out + c)
                except FileExistsError:
                    pass

            number_of_sentences = len(os.listdir(out + "/Graph_Sources"))

            if args.out and args.index:
                for n in range(number_of_sentences):
                    conllu_parser.show_graph(n, "deep/" + args.filename + "/" + args.out)
            right_type_dict, other_type_dict = statifier.frequency_on_grammar_feature_checking_reldep(out + f"/UD_RelDep/{args.property}.txt",
                                                                        args.feature_type)
            property_matches = (sum(map(lambda l: l[1], right_type_dict.items())))
        print(f"We have studied {number_of_sentences} sentences and failed on {cpt} in treebank `{args.filename}`"
              f"Of those, {property_matches} words match `{args.property}`.")
        if property_matches != 0:
            right_type_dict = statifier.format_case_stats(right_type_dict)
            print(f"We get the following distribution of values for `{args.feature_type}` matching `{args.property}`:")
            for v in right_type_dict:
                print(f"\t{v}: {right_type_dict[v]}")
            if right_type_dict.get("Other_Word_Type", 0) != 0:
                print("The Other Words have the following distribution of grammatical features:")
                other_type_dict = statifier.format_property_stats(other_type_dict)
                for v in other_type_dict:
                    print(f"Feature {v}:")
                    for a in other_type_dict[v]:
                        print(f"\t{a}: {other_type_dict[v][a]}")
