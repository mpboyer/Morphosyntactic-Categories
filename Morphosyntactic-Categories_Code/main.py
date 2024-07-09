import os

import pandas
from tqdm import tqdm

import get_case_conllu

MODE = "Pronouns"
UDDIR = "ud-treebanks-v2.14"
SAVE_DIR = f"{MODE}_Case_RelDep_Matches"


def empacker(filename):
    files = filename.split("/")
    pwd = f"{UDDIR}/{files[0]}"
    out = pwd + f"/{files[1][:-7]}/"
    try:
        os.mkdir(out)
    except FileExistsError:
        pass

    try:
        os.mkdir(out + f'/{SAVE_DIR}')
    except FileExistsError:
        pass

    case_vectors = get_case_conllu.vectorize(f"{UDDIR}/{filename}")

    for c, vector in case_vectors:
        with open(f"{out}/{SAVE_DIR}/RelDep_matching_Case={c}.txt", "w") as f:
            f.write(vector)


def process_all_banks():
    treebanks = os.listdir(UDDIR)
    for treebank in (pbar := tqdm(treebanks, colour='#7d1dd3')):
        pbar.set_description(f"Processing {treebank}")
        content = os.listdir(f"{UDDIR}/{treebank}")
        for c in list(filter(lambda t: t[-7:] == ".conllu", content)):
            empacker(f"{treebank}/{c}")
    pbar.set_description("Treebank processing done")


def get_all_cases():
    cases = set()
    treebanks = os.listdir(UDDIR)
    for treebank in treebanks:
        content = os.listdir(f"{UDDIR}/{treebank}")
        for corpus in list(filter(lambda t: t[-7:] == ".conllu", content)):
            present_cases = list(
                    filter(lambda t: t[16:20] == "Case", os.listdir(f"{UDDIR}/{treebank}/{corpus[:-7]}/{SAVE_DIR}"))
            )
            cases |= set([t[21:24] for t in present_cases])
    return cases


def from_vectors_to_csvs():
    cases = get_all_cases()
    for case in cases:
        rel_dep_matching_grammar_feature = f'RelDep_matching_Case={case}'
        try:
            open(f"{SAVE_DIR}/{rel_dep_matching_grammar_feature}.csv", "r")
        except FileNotFoundError:
            print("Tabulating Case")
            value_dicts = []
            fieldnames = ["Treebank", "Number of Sentences", "Failures", "Total"]
            fields = set()

            for c in (pbar := tqdm(
                    list(
                        filter(
                            lambda t: t[1][-7:] == ".conllu",
                            [(treebanks, treebank) for treebanks in os.listdir(UDDIR) for treebank in
                             os.listdir(f"{UDDIR}/{treebanks}")]
                        )
                    )
                    , colour="#7d1dd3"
            )):

                treebanks, treebank = c
                treebank = treebank[:-7]
                pbar.set_description(f"Tabulating {treebanks}/{treebank}")
                try:
                    with open(
                            f"{UDDIR}/{treebanks}/{treebank}/{SAVE_DIR}/{rel_dep_matching_grammar_feature}.txt"
                    ) as f:
                        results = f.readlines()
                    results[0] = results[0].split(" ")

                    vec_coordinates = {
                        "Treebank": treebank,
                        "Number of Sentences": results[0][3],
                        "Failures": results[0][8]}
                    total_values = 0
                    for line in results[2:]:
                        res = line.split(" ")
                        reldep = res[1][:-1]
                        number = res[2]
                        vec_coordinates[reldep] = int(number)
                        total_values += int(number)
                        fields.add(reldep)
                    vec_coordinates["Total"] = total_values
                    value_dicts.append(vec_coordinates)
                except FileNotFoundError:
                    pass

            pbar.set_description("Tabulating Treebanks Done")
            pbar.close()
            pandas.DataFrame(value_dicts, columns=fieldnames + sorted(fields)).to_csv(
                f"{SAVE_DIR}/{rel_dep_matching_grammar_feature}.csv", index=False
            )


if __name__ == "__main__":
    process_all_banks()
    from_vectors_to_csvs()
