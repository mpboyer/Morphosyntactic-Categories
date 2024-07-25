import re


forbidden_reldeps = ["det", "conj", "case", "amod"]
allowed_pos = ['NOUN']


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
    cases = set()

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
                    continue

                elif annotations[7] not in forbidden_reldeps:
                    attributes = {
                        "position": word + offset,
                        "grapheme": annotations[1],
                        "lemma": annotations[2],
                        "part_of_speech": annotations[3],
                        "edge_type": annotations[7].split(":")[0]
                    }
                    features = annotations[5]

                    if annotations[6] != "0":
                        attributes["predecessor"] = annotations[6]
                    else:
                        attributes["predecessor"] = str(word + offset)

                    if features == "_":
                        pass
                    elif annotations[3] not in allowed_pos:
                        pass
                    else:
                        features = features.split("|")
                        for feature in features:
                            feature = feature.split("=")
                            if feature[0] == 'Case':
                                attributes[feature[0]] = feature[1]
                                cases.add(feature[1])
                    sentence_dict[attributes["position"]] = attributes

            except IndexError:
                number_of_cpt_sentences += 1
                print(number_of_cpt_sentences)
            word += 1
        trees.append(sentence_dict)

    results = []
    for case in cases:
        reldep_for_grammar_feature = ""
        reldep_for_grammar_feature += (
            f"We have studied {number_of_studied_sentences} sentences and failed on {number_of_cpt_sentences} in "
            f"treebank `{filename}`.\n We get the following distribution "
            f"of RelDep for words matching `Case={case}`:\n")
        res_dict = {}
        for t in trees:
            for w in t:
                if is_prefix(case, t[w].get('Case', "")):
                    res_dict[t[w]["edge_type"]] = res_dict.get(t[w]["edge_type"], 0) + 1

        for v in res_dict:
            reldep_for_grammar_feature += f"RelDep {v}: {res_dict[v]}\n"

        results.append((case, reldep_for_grammar_feature))

    return results


def vectorize_appos(filename):
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
    appos = {}

    for sentence in sentences:
        to_check = []
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
                    continue

                else:
                    attributes = {
                        "position": word + offset,
                        "grapheme": annotations[1],
                        "lemma": annotations[2].replace('/', '|'),
                        "part_of_speech": annotations[3],
                        "edge_type": annotations[7].split(":")[0]
                    }
                    if annotations[6] != "0":
                        attributes["predecessor"] = annotations[6]
                    else:
                        attributes["predecessor"] = str(word + offset)

                    if attributes["part_of_speech"] == 'ADP':
                        a = attributes["lemma"]
                        appos[a] = appos.get(a, {})
                        to_check.append((a, attributes["predecessor"]))

                    sentence_dict[attributes["position"]] = attributes

            except IndexError:
                number_of_cpt_sentences += 1
                print(number_of_cpt_sentences)
            word += 1

        for p, pred in to_check:
            try :
                pred_dict = sentence_dict[int(pred)]
            except ValueError:
                continue
            if pred_dict["edge_type"] not in forbidden_reldeps:
                appos[p][pred_dict["edge_type"]] = appos[p].get(pred_dict["edge_type"], 0) + 1

        trees.append(sentence_dict)

    results = []
    for lemma in appos:
        # print(lemma)
        reldep_for_grammar_feature = ""
        reldep_for_grammar_feature += (
            f"We have studied {number_of_studied_sentences} sentences and failed on {number_of_cpt_sentences} in "
            f"treebank `{filename}`.\n We get the following distribution "
            f"of RelDep for appos matching `Lemma={lemma}`:\n")
        res_dict = {}
        for t in trees:
            for w in t:
                res_dict[t[w]["edge_type"]] = res_dict.get(t[w]["edge_type"], 0) + 1

        for v in res_dict:
            reldep_for_grammar_feature += f"RelDep {v}: {res_dict[v]}\n"

        results.append((lemma, reldep_for_grammar_feature))
    return results


if __name__ == "__main__":
    vectorize_appos("../ud-treebanks-v2.14/UD_Classical_Armenian-CAVaL/xcl_caval-ud-train.conllu")
