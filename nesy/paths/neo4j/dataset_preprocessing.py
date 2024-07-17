import csv
import pandas as pd


def convert_tsv_to_csv(
    input_file_path: str, output_file_path: str, columns_to_retain: list[str]
) -> None:
    """
    Converts a TSV file to a CSV file while filtering the columns from the input file.
    Args:
        input_file_path: The path to the input TSV file.
        output_file_path: The path to the output CSV file.
        columns_to_retain: A whitelist of columns to retain.

    Returns:
        None
    """
    # read the tsv file into memory
    with open(input_file_path, "r", newline="") as tsvfile:
        tsvreader = csv.DictReader(tsvfile, delimiter="\t")

        # create/overwrite the output file
        with open(output_file_path, "w", newline="") as csvfile:
            csvwriter = csv.DictWriter(csvfile, fieldnames=columns_to_retain)
            csvwriter.writeheader()
            # write the output row by row
            for row in tsvreader:
                # dunno why the linter is complaining, using a string name to access the column works
                csvwriter.writerow({col: row[col] for col in columns_to_retain})


def split_csv_for_neo4j(
    csv_file_file_path: str,
    nodes_output_file_path: str,
    rels_output_file_path: str,
    discarded_rels_output_file_path: str,
) -> None:
    """
    Given a CSV file containing KG triples, split it into one properties CSV file and one nodes CSV file.
    Whenever triples which use properties as endpoints are encountered, they are discarded and their relation is
    added to the discarded rels file. There are no guarantees of uniqueness of such relationships.
    Args:
        csv_file_file_path: The path to the CSV file containing KG triples.
        nodes_output_file_path: The path of the file where the nodes CSV will be written.
        rels_output_file_path: The path of the file where the relations CSV will be written.
        discarded_rels_output_file_path: The path to the CSV file for saving discarded relations.

    Returns:
        None
    """
    # create sets to store unique nodes and discarded rels
    nodes_set = set()
    discarded_rels = set()
    with open(csv_file_file_path, "r") as input_csv_file, open(
        rels_output_file_path, "w"
    ) as out_f:
        next(input_csv_file)  # Skip the input header

        # write the header to the properties file
        out_f.write(":START_ID,property,:END_ID,:TYPE\n")

        # read the input triples one by one
        for line in input_csv_file:
            node1, prop, node2 = line.strip().split(",")
            # if at least one endpoint uses a property discard it
            if node1[0] == "P" or node2[0] == "P":
                discarded_rels.add(prop)
            else:
                nodes_set.add(node1)
                nodes_set.add(node2)
                out_f.write(line.strip() + f",{prop}" + "\n")

    # create a sorted list of unique nodes, I don't remember why I was sorting it
    nodes_list = sorted(list(nodes_set))

    with open(nodes_output_file_path, "w") as nodes_output_file:
        # write the header to the nodes file
        # I was using the name 'qualified_id' erroneously, it should be replaced
        nodes_output_file.write("qualifier_id:ID\n")

        # write the nodes to the file one at a time
        for node in nodes_list:
            nodes_output_file.write(node + "\n")
    with open(discarded_rels_output_file_path, "w") as discarded_rels_output_file:
        # write the discarded rels one at a time
        for prop in discarded_rels:
            discarded_rels_output_file.write(prop + "\n")


def create_csv_files_neo4j(claims_path: str, labels_path: str) -> None:
    """
    It takes as input the claims and labels tsv files of wikidata and creates two csv files for neo4j. A node file
    containing information regarding entities and relationships, and a relationships file containing the
    wikidata triplets. The file is saved at the same folder.

    :param claims_path: file containing all the claims of the wikidata dump
    :param labels_path: file containing the english labels of the wikidata dump
    """
    # create sets to store unique nodes and discarded rels
    nodes_set = set()
    discarded_rels = set()

    # create labels dictionary
    labels_dict = {}
    with open(labels_path, "r") as labels_file:
        next(labels_file)
        for line in labels_file:
            stripped_line = line.strip().split(",")
            node = stripped_line[0]
            label = ",".join(stripped_line[1:])
            labels_dict[node] = label

    with open(claims_path, "r") as claims_file, open(
        "/".join(claims_path.split("/")[:-1]) + "/relationships.csv", "w"
    ) as out_f:
        next(claims_file)  # Skip the input header

        # write the header to the properties file
        out_f.write(":START_ID,label,:END_ID,:TYPE\n")

        # read the input triples one by one
        for line in claims_file:
            node1, prop, node2 = line.strip().split(",")
            # if at least one endpoint uses a property discard it
            if node1[0] == "P" or node2[0] == "P":
                discarded_rels.add(prop)
            else:
                nodes_set.add(node1)
                nodes_set.add(node2)
                # nodes_set.add(prop)
                if prop in labels_dict:
                    label = labels_dict[prop]
                else:
                    # check if it is an inverse relationship
                    if prop.endswith("_") and prop[:-1] in labels_dict:
                        label = labels_dict[prop[:-1]]
                    else:
                        label = "no_label"
                out_f.write(node1 + "," + label + "," + node2 + "," + prop + "\n")

    # create a sorted list of unique nodes, I don't remember why I was sorting it
    nodes_list = sorted(list(nodes_set))

    with open("/".join(claims_path.split("/")[:-1]) + "/nodes.csv", "w") as nodes_output_file:
        # write the header to the nodes file
        # I was using the name 'qualified_id' erroneously, it should be replaced
        nodes_output_file.write("wikidata_id:ID,label,:LABEL\n")

        # write the nodes to the file one at a time
        for node in nodes_list:
            if node in labels_dict:
                label = labels_dict[node]
            else:
                label = "no_label"
            if node[0] == "P":
                type_ = "relation"
            else:
                type_ = "entity"
            nodes_output_file.write(node + "," + label + "," + type_ + "\n")

    with open("/".join(claims_path.split("/")[:-1]) + "/discarded.csv", "w") as discarded_rels_output_file:
        # write the discarded rels one at a time
        for prop in discarded_rels:
            discarded_rels_output_file.write(prop + "\n")


def create_csv_files_neo4j_no_inverse(claims_path: str, labels_path: str, selected_properties: str = None) -> None:
    """
    Same as previous one but it does not include inverse relationships. Also, it gives the possibility to select
    the properties to include in the relationships.csv file.

    :param claims_path: file containing all the claims of the wikidata dump
    :param labels_path: file containing the english labels of the wikidata dump
    :param selected_properties: path to the csv file containing the properties to include in the relationships file
    """
    if selected_properties is not None:
        selected_properties = pd.read_csv(selected_properties)
        selected_properties = list(selected_properties["wiki_id"])
    # create sets to store unique nodes and discarded rels
    nodes_set = set()
    discarded_rels = set()

    # create labels dictionary
    labels_dict = {}
    with open(labels_path, "r") as labels_file:
        next(labels_file)
        for line in labels_file:
            stripped_line = line.strip().split(",")
            node = stripped_line[0]
            label = ",".join(stripped_line[1:])
            # remove ' and @en from labels
            label = label[1:-4]
            label = label.replace(",", "")
            labels_dict[node] = label

    with open(claims_path, "r") as claims_file, open(
        "/".join(claims_path.split("/")[:-1]) + "/relationships.csv", "w"
    ) as out_f:
        next(claims_file)  # Skip the input header

        # write the header to the properties file
        out_f.write(":START_ID,wikidata_id,label,:END_ID,:TYPE\n")

        # read the input triples one by one
        for line in claims_file:
            node1, prop, node2 = line.strip().split(",")
            # check that the property is not an inverse (ends with _) and it is in the set of selected ones
            if not prop.endswith("_") and prop in selected_properties:
                # if at least one endpoint uses a property discard it
                if node1[0] == "P" or node2[0] == "P":
                    discarded_rels.add(prop)
                else:
                    nodes_set.add(node1)
                    nodes_set.add(node2)
                    # nodes_set.add(prop)
                    if prop in labels_dict:
                        label = labels_dict[prop]
                    else:
                        label = "no_label"
                    out_f.write(node1 + "," + prop + "," + label + "," + node2 + "," + "relation" + "\n")

    # create a sorted list of unique nodes, I don't remember why I was sorting it
    nodes_list = sorted(list(nodes_set))

    with open("/".join(claims_path.split("/")[:-1]) + "/nodes.csv", "w") as nodes_output_file:
        # write the header to the nodes file
        # I was using the name 'qualified_id' erroneously, it should be replaced
        nodes_output_file.write("wikidata_id:ID,label,:LABEL\n")

        # write the nodes to the file one at a time
        for node in nodes_list:
            if node in labels_dict:
                label = labels_dict[node]
            else:
                label = "no_label"
            if node[0] == "P":
                type_ = "relation"
            else:
                type_ = "entity"
            nodes_output_file.write(node + "," + label + "," + type_ + "\n")

    with open("/".join(claims_path.split("/")[:-1]) + "/discarded.csv", "w") as discarded_rels_output_file:
        # write the discarded rels one at a time
        for prop in discarded_rels:
            discarded_rels_output_file.write(prop + "\n")
