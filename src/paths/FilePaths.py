from dataclasses import dataclass
from typing import Optional


@dataclass
class FilePaths:
    """
    This class contains the references to the file paths to the files necessary for computing the paths in Neo4j.
    The domain names are used to differentiate between the precomputed paths stored on the Neo4j database, in order
    to subsequently retrieve the paths for the correct source-target pairs.
    The review files are required if cold start and popularity threshold need to be applied
    """

    source_domain_name: str
    mapping_source_domain: str
    reviews_source_domain: Optional[str]

    target_domain_name: str
    mapping_target_domain: str
    reviews_target_domain: Optional[str]
