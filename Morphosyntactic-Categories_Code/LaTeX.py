import pandas

def figurifier(grammar_feature):
    results = r"\renewcommand{\arraystretch}{1.1}" + "\n"
    proximities = pandas.ExcelFile(f"DuoProximity/{grammar_feature[0]}={grammar_feature[1]}_Proximity.xlsx")
    results += r"\begin{table}[H]" + "\n\t" + r"\centering" + "\n\t" + r"\begin{NiceTabular}{" + r"c" * (
        len(proximities.sheet_names)) + "}\n\t\t"
    results += r"Proximity with: "
    for s in proximities.sheet_names:
        if s != "Sheet":
            concurrent_case = s.split("_")[2]
            results += f"& {concurrent_case} "
    results += r"\\" + "\n"

    value_dict = {
        "Median": {},
        "Mean": {},
        "NLow": {},
        "NHigh": {},
        "First Quartile": {},
        "Third Quartile": {},
    }
    for s in proximities.sheet_names:
        if s != "Sheet":
            ws = proximities.parse(s)
            ws = ws[ws.columns[1:]].to_numpy()[574:, :574]
            value_dict["Median"][s] = round(np.nanmedian(ws), 5)
            value_dict["First Quartile"][s] = round(np.nanquantile(ws, 0.25), 5)
            value_dict["Third Quartile"][s] = round(np.nanquantile(ws, 0.75), 5)
            value_dict["Mean"][s] = round(np.nanmean(ws), 5)
            value_dict["NLow"][s] = round(np.count_nonzero((ws > 0) & (ws < .2)), 5)
            value_dict["NHigh"][s] = round(np.count_nonzero((ws < 1) & (ws > .8)), 5)

    for stat in value_dict:
        results += f"\t\t{stat} "
        for g in value_dict[stat]:
            results += f"& {value_dict[stat][g]} "
        results += r"\\" + "\n"

    results += "\t" + r"\CodeAfter" + "\n\t\t"
    results += r"\begin{tikzpicture}" + "\n\t\t\t"
    results += r"\foreach \i in {1,...," + f"{len(value_dict) + 2}" + r"}" + "\n\t\t\t\t"
    results += r"{\draw[draw=vulm] (1|-\i) -- (" + f"{len(proximities.sheet_names) + 1}|-" + r"\i);}" + "\n\t\t\t"
    results += r"\draw[draw=vulm] (2|-1)--(2|-" + f"{len(value_dict) + 2});"
    results += r"\end{tikzpicture}" + "\n\t"
    results += r"\end{NiceTabular}" + "\n\t"
    results += r"\caption{Proximities for " + f"{grammar_feature[0]}={grammar_feature[1]}" + "}\n"
    results += r"\end{table}"
    with open(f"DuoProximity/{grammar_feature[0]}={grammar_feature[1]}_Proximity.tex", 'w') as f:
        f.write(results)