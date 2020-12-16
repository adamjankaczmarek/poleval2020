import sys
import persisted_dictionary as pd
from collections import defaultdict


def get_results(in_file, out_file, mult):
    rd = pd.Persisted_dict(in_file, False)
    scores = defaultdict(list)
    for k in rd:
        v = rd[k]
        scores[k[0]].append((mult*v['ac'] + sum(v['bert']), k[1]))

    best_instance = dict()
    for x in rd:
        x = x[0]
        best_instance[x] = sorted(scores[x], key=lambda x:x[0])[0][1]

    uniq_utt_id = {k[0] for k in rd}
    best_id = {k: sorted(scores[k], key=lambda x:x[0])[0][1] for k in uniq_utt_id}
    out_utt = ["\t".join([i[0], rd[i]['sen'] + "\n"]) for i in best_id.items()]
    with open(out_file, "w+") as out_utt_file:
        out_utt_file.writelines(out_utt)


if __name__ == "__main__":
    mult = float(sys.argv[1])
    pd_file = sys.argv[2]
    out_file = sys.argv[3]
    get_results(pd_file, out_file, mult)

