import os
import argparse
import conllu_parser
import statifier

UDDIR = "ud-treebanks-v2.14"

parser = argparse.ArgumentParser()
parser.add_argument("filename", help=".conllu file containing a UD-Treebank. Use all to iterate on all banks.")
parser.add_argument("-v", "--verbose", required=False)
parser.add_argument("-p", "--property", help="UD-RELDEP which we're studying, e.g. obl:tmod", required=True)
parser.add_argument("-f", "--feature_type", help="Morphosyntactic category we want to study.", required=True)
parser.add_argument("-val", "--value", help="Value of the studied feature", required=True)
parser.add_argument("-o", "--out", help="File in which to store the result", required=False, default="Sentence_Graphs")
parser.add_argument("-i", "--index",
                    help=f"Number of the sentences to study. Must be smaller than MAX_STUDIED_SENTENCES={conllu_parser.MAX_STUDIED_SENTENCES}",
                    required=False)
args = parser.parse_args()


def empacking(filename, reldep, gf, index, verbose):
    if verbose is None:
            print(filename)
    filename = (filename).split("/")

    pwd = f"{UDDIR}/{filename[0]}"
    out = pwd + f"/{filename[1][:-7]}/"
    try:
        os.mkdir(out)
    except FileExistsError:
        pass

    for c in ["/Graph_Sources", "/Graphs", "/UD_RelDep", "/Features", "/RelDep_Matches", "/Grammar_Matches"]:
        try:
            os.mkdir(out + c)
        except FileExistsError:
            pass

    cpt = conllu_parser.treeifier(pwd + "/" + filename[1], ud_reldep=reldep,
                                  grammar_feature=gf, out=out)
    number_of_sentences = len(os.listdir(out + "/Graph_Sources"))
    if index:
        conllu_parser.show_graph(index, "deep/" + filename[0] + "/" + args.out)
    right_type_dict, other_type_dict = statifier.frequency_on_grammar_feature_checking_reldep(
        out + f"/UD_RelDep/{args.property}.txt",
        args.feature_type)
    property_matches = (sum(map(lambda l: l[1], right_type_dict.items())))
    reldep_dict = statifier.frequency_on_reldep_checking_grammar_feature(
        f"{out}/Features/{gf[0]}={gf[1]}.txt")

    reldep_for_grammar_feature = ""
    grammar_feature_for_reldep = f"We have studied {number_of_sentences} sentences and failed on {cpt} in treebank `{filename}`.\n"
    grammar_feature_for_reldep += f"Of those, {property_matches} words match `{reldep}`.\n"
    if property_matches != 0:
        right_type_dict = statifier.format_case_stats(right_type_dict)
        grammar_feature_for_reldep += (f"We get the following distribution of values for `{gf[0]}` "
                                       f"matching `{reldep}`:\n")
        for v in right_type_dict:
            grammar_feature_for_reldep += f"\t{v}: {right_type_dict[v]}\n"

        if right_type_dict.get("Other_Word_Type", 0) != 0:
            grammar_feature_for_reldep += ("The Other Words have the following distribution of grammatical "
                                           "features:\n")
            other_type_dict = statifier.format_property_stats(other_type_dict)
            for v in other_type_dict:
                grammar_feature_for_reldep += f"Feature {v}:\n"
                for a in other_type_dict[v]:
                    grammar_feature_for_reldep += f"\t{a}: {other_type_dict[v][a]}\n"
        with open(f"{out}/Grammar_Matches/{gf[0]}_matching_{reldep}", 'w') as f:
            f.write(grammar_feature_for_reldep)
        reldep_for_grammar_feature += (f"We have studied {number_of_sentences} sentences and failed on {cpt} in "
                                       f"treebank `{filename}`.\n We get the following distribution "
                                       f"of RelDep for words matching `{gf[0]}={gf[1]}`:\n")
        reldep_dict = statifier.format_case_stats(reldep_dict)
        for v in reldep_dict:
            reldep_for_grammar_feature += f"RelDep {v}: {reldep_dict[v]}\n"
        with open(f"{out}/RelDep_Matches/RelDep_matching_{gf[0]}={gf[1]}.txt", "w") as f:
            f.write(reldep_for_grammar_feature)
    if verbose:
        print(grammar_feature_for_reldep)
        if property_matches != 0:
            print(reldep_for_grammar_feature)


if __name__ == "__main__":
    grammar_feature = (args.feature_type, args.value)
    if args.filename != "all":
        empacking(args.filename, args.property, grammar_feature, args.index, args.verbose)
    else:
        treebanks = os.listdir(UDDIR)
        reldep_dict = {}
        number_of_studied_sentences = 0
        cpt = 0
        for treebank in treebanks:
            content = os.listdir(f"{UDDIR}/{treebank}")
            for c in list(filter(lambda t: t[-7:] == ".conllu", content)):
                empacking(f"{treebank}/{c}", args.property, grammar_feature, args.index, args.verbose)

        for treebank in treebanks:
            content = os.listdir(f"{UDDIR}/{treebank}")
            for c in list(filter(lambda t: t[-7:] == ".conllu", content)):
                try:
                    with open(f"{UDDIR}/{treebank}/{c[:-7]}/RelDep_Matches/RelDep_matching_{grammar_feature[0]}={grammar_feature[1]}.txt", 'r') as f:
                        reldeps = f.readlines()
                    for i in range(len(reldeps)):
                        reldeps[i] = reldeps[i].split(" ")
                    number_of_studied_sentences += int(reldeps[0][3])
                    cpt += int(reldeps[0][8])
                    for i in range(2, len(reldeps)):
                        rel = reldeps[i][1][:-1]
                        reldep_dict[rel] = reldep_dict.get(rel, 0) + int(reldeps[i][2])
                except FileNotFoundError:
                    empacking(f"{treebank}/{c}", args.property, grammar_feature, args.index, args.verbose)
                    try :
                        with open(
                                f"{UDDIR}/{treebank}/{c[:-7]}/RelDep_Matches/RelDep_matching_{grammar_feature[0]}={grammar_feature[1]}.txt",
                                'r') as f:
                            reldeps = f.readlines()
                        for i in range(len(reldeps)):
                            reldeps[i] = reldeps[i].split(" ")
                        number_of_studied_sentences += int(reldeps[0][3])
                        cpt += int(reldeps[0][8])
                        for i in range(2, len(reldeps)):
                            rel = reldeps[i][1][:-1]
                            reldep_dict[rel] = reldep_dict.get(rel, 0) + int(reldeps[i][2])
                    except FileNotFoundError:
                        pass
        reldep_for_grammar_feature = (f"We have studied {number_of_studied_sentences} sentences and failed on {cpt} in "
                                       f"all treebanks.\nWe get the following distribution "
                                       f"of RelDep for words matching `{grammar_feature[0]}={grammar_feature[1]}`:\n")
        reldep_dict = statifier.format_case_stats(reldep_dict)
        for v in reldep_dict:
            reldep_for_grammar_feature += f"RelDep {v}: {reldep_dict[v]}\n"
        with open(f"RelDep_Matches/RelDep_matching_{grammar_feature[0]}={grammar_feature[1]}.txt", "w") as f:
            f.write(reldep_for_grammar_feature)
