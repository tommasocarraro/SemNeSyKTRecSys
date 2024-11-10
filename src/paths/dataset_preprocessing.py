import os.path
from loguru import logger
import pandas as pd
import csv
import json
import re


def process_wikidata_dump(input_file, output_labels_file, output_triples_file):
    with open(input_file, "r", encoding="utf-8") as f_in, open(
        output_labels_file, "w", newline="", encoding="utf-8"
    ) as f_out_labels, open(
        output_triples_file, "w", newline="", encoding="utf-8"
    ) as f_out_triples:

        # CSV writers for labels and triples
        labels_writer = csv.writer(f_out_labels)
        triples_writer = csv.writer(f_out_triples)

        # Write headers for both CSV files
        labels_writer.writerow(["id", "english_label"])
        triples_writer.writerow(["subject", "predicate", "object"])

        # Process the JSON file line-by-line
        for line in f_in:
            # Skip the starting and ending brackets of the JSON array
            if line.startswith("[") or line.startswith("]"):
                continue

            # Remove trailing commas and newlines
            line = line.rstrip(",\n")

            try:
                entity = json.loads(line)  # Parse the JSON line into a dictionary
            except json.JSONDecodeError:
                continue  # Skip any malformed JSON line

            # Extract the entity ID
            entity_id = entity.get("id")

            # Extract and write English label if available
            english_label = entity.get("labels", {}).get("en", {}).get("value")
            if entity_id and english_label:
                labels_writer.writerow([entity_id, english_label])

            # Extract and write triples for datatype 'wikibase-item'
            if "claims" in entity:
                for prop, claims in entity["claims"].items():
                    for claim in claims:
                        mainsnak = claim.get("mainsnak", {})
                        if (
                            mainsnak.get("snaktype") == "value"
                            and mainsnak.get("datatype") == "wikibase-item"
                        ):
                            object_id = (
                                mainsnak.get("datavalue", {}).get("value", {}).get("id")
                            )
                            if object_id:
                                triples_writer.writerow([entity_id, prop, object_id])


def remove_special_characters(input_string: str) -> str:
    return re.sub(r"[^\w\d\s]", "", input_string)


def create_csv_files_neo4j(
    claims_path: str, labels_path: str, selected_properties: str = None
) -> None:
    """
    It takes as input the claims and labels tsv files of wikidata and creates two csv files for neo4j. A node file
    containing information regarding entities and relationships, and a relationships file containing the
    wikidata triplets. The file is saved at the same folder.
    Also, it gives the possibility to select the properties to include in the relationships.csv file.

    :param claims_path: file containing all the claims of the wikidata dump
    :param labels_path: file containing the english labels of the wikidata dump
    :param selected_properties: path to the csv file containing the properties to include in the relationships file
    """
    if selected_properties is not None:
        selected_properties = pd.read_csv(selected_properties)
        selected_properties = list(selected_properties["wiki_id"])

    # create sets to store unique nodes
    nodes_set = set()

    # create labels dictionary
    logger.debug("Reading the labels file into memory")
    labels_dict = {}
    with open(labels_path, "r") as labels_file:
        next(labels_file)
        for line in labels_file:
            stripped_line = line.strip().split(",")
            node = stripped_line[0]
            labels_dict[node] = remove_special_characters(stripped_line[1])

    logger.debug("Reading the claims file into memory")
    with open(claims_path, "r") as claims_file, open(
        os.path.join(os.path.dirname(claims_path), "relationships.csv"), "w"
    ) as rels_file:
        rels_writer = csv.writer(rels_file)
        next(claims_file)  # Skip the input header

        # write the header to the properties file
        rels_writer.writerow([":START_ID", "wikidata_id", "label", ":END_ID", ":TYPE"])

        # read the input triples one by one
        for line in claims_file:
            node1, prop, node2 = line.strip().split(",")
            # check that the property is in the set of selected ones
            if prop in selected_properties:
                # if at least one endpoint is a property discard the triple
                if node1[0] != "P" and node2[0] != "P":
                    nodes_set.add(node1)
                    nodes_set.add(node2)
                    if prop in labels_dict:
                        label = labels_dict[prop]
                    else:
                        label = "no_label"

                    rels_writer.writerow([node1, prop, label, node2, "relation"])

    logger.debug("Writing the nodes file to file system")
    with open(
        os.path.join(os.path.dirname(claims_path), "nodes.csv"), "w"
    ) as nodes_file:
        nodes_writer = csv.writer(nodes_file)
        # write the header to the nodes file
        nodes_writer.writerow(["wikidata_id:ID", "label", ":LABEL"])

        # write the nodes to the file system one at a time
        for node in (item for item in nodes_set):
            if node in labels_dict:
                label = labels_dict[node]
            else:
                label = "no_label"
            if node[0] == "P":  # this can never occur
                type_ = "relation"
            else:
                type_ = "entity"
            nodes_writer.writerow([node, label, type_])
