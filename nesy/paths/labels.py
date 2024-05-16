import math
import sqlite3
from tqdm import tqdm
from .utils import count_lines
import pandas as pd
import os
import json


def create_wikidata_labels_sqlite(raw_labels):
    """
    This function creates a SQLITE database file containing the labels of wikidata entities. It is used to efficiently
    access to the large wikidata label file while generating labels for knowledge graph paths.

    :param raw_labels: path to raw wikidata labels file
    """
    conn = sqlite3.connect("./data/wikidata/labels.db")
    c = conn.cursor()

    c.execute('''CREATE TABLE labels (wikidata_id TEXT PRIMARY KEY, label TEXT)''')
    with open(raw_labels, 'r') as f:
        for i, line in enumerate(tqdm(f, total=count_lines(raw_labels))):
            if i != 0:
                split_line = line.split("\t")
                c.execute("INSERT INTO labels VALUES (?, ?)", (split_line[1], split_line[3]))

    conn.commit()
    conn.close()


def convert_ids_to_labels(wiki_paths_file, add_wiki_id=False):
    """
    This function takes a tsv file containing paths between wikidata entities and converts the IDs into actual wikidata
    labels. Note each row represents a different path.

    It creates a new tsv file (with the same structure of the given one) containing labels instead of IDs.

    Note that when a label is missing in the database, the wikidata ID is used in place on the label.

    :param wiki_paths_file: path to the tsv file containing wikidata paths
    :param add_wiki_id: whether to include the wikidata IDs associated with the labels in the paths
    """
    df = pd.read_csv(wiki_paths_file, sep="\t")
    conn = sqlite3.connect("./data/wikidata/labels.db")
    c = conn.cursor()

    def get_label(x):
        suffix = ""
        if x.startswith("P") and x.endswith("_"):
            x = x[:-1]
            suffix = " (inverse)"
        try:
            label = c.execute("SELECT label FROM labels WHERE wikidata_id=?", (x,)).fetchone()[0]
        except TypeError:
            return x
        # remove language specification
        if "@" in label:
            label = label.split("@")[0]
        # remove ' in front and tail
        if label.startswith("'") and label.endswith("'"):
            label = label.strip("'")
        label = label + suffix
        return label

    res = df.map(lambda x: (x + " : " + get_label(x)
                            if add_wiki_id else get_label(x)) if isinstance(x, str) or not math.isnan(x) else "")
    res.to_csv(wiki_paths_file[:-4] + "_labelled.tsv", index=False, sep="\t")
    res = res.values
    out = {"standard": {}, "inverse": {}}
    for i, path in enumerate(res):
        out_str = ""
        # out_str += "path %d: " % (i + 1, )
        for item in path:
            if item != "":
                out_str = out_str + item + " ---> "
            else:
                out_str = out_str.rstrip(" ---> ")
                break
        out_str = out_str.rstrip(" ---> ")
        n_hops = int(out_str.count("--->") / 2)

        if "inverse" in out_str:
            if n_hops in out["inverse"]:
                out["inverse"][n_hops].append(out_str)
            else:
                out["inverse"][n_hops] = [out_str]
        else:
            if n_hops in out["standard"]:
                out["standard"][n_hops].append(out_str)
            else:
                out["standard"][n_hops] = [out_str]

    # create JSON file with the paths
    with open(wiki_paths_file[:-4] + "_labelled.json", "w", encoding='utf-8') as outfile:
        json.dump(out, outfile, indent=4)

    # create txt file of the paths
    with open(wiki_paths_file[:-4] + "_labelled.txt", 'w', encoding='utf-8') as f:
        for relation, hops in out.items():
            f.write(relation + "\n\n")
            for hop, paths in hops.items():
                f.write("%s-hop\n\n" % (hop, ))
                for path in paths:
                    f.write(path + "\n")
                f.write("\n")
            f.write("\n\n")

    conn.close()


def generate_all_labels(paths_dir):
    """
    This function iterates the given folder, looks for all paths_all.tsv files, and generate labels for all the found
    paths.

    :param paths_dir: path to directory containing the path files
    """
    for root, dirs, files in os.walk(paths_dir):
        for filename in files:
            if "all." in filename:
                convert_ids_to_labels(os.path.join(root, filename))
