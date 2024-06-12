import csv


def convert_tsv_to_csv(
    input_file_path: str, output_file_path: str, columns_to_retain: list[str]
):
    with open(input_file_path, "r", newline="") as tsvfile:
        with open(output_file_path, "w", newline="") as csvfile:
            tsvreader = csv.DictReader(tsvfile, delimiter="\t")
            csvwriter = csv.DictWriter(csvfile, fieldnames=columns_to_retain)
            csvwriter.writeheader()
            for row in tsvreader:
                csvwriter.writerow({col: row[col] for col in columns_to_retain})


def split_csv_for_neo4j(
    csv_file_file_path: str, discarded_triples_file_path: str
) -> None:
    nodes_set = set()
    useless_props = set()
    relationships_file = "out/relationships.csv"
    with open(csv_file_file_path, "r") as f, open(relationships_file, "w") as out_f:
        next(f)  # Skip header
        out_f.write(":START_ID,property,:END_ID,:TYPE\n")

        for line in f:
            node1, prop, node2 = line.strip().split(",")
            if node1[0] == "P" or node2[0] == "P":
                useless_props.add(prop)
            else:
                nodes_set.add(node1)
                nodes_set.add(node2)
                out_f.write(line.strip() + f",{prop}" + "\n")

    nodes_list = sorted(list(nodes_set))

    nodes_file = "out/nodes.csv"
    with open(nodes_file, "w") as f:
        f.write("qualifier_id:ID\n")
        for node in nodes_list:
            f.write(node + "\n")
    with open(discarded_triples_file_path, "w") as f:
        for prop in useless_props:
            f.write(prop + "\n")
