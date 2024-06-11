from typing import TypedDict, Optional
from nesy.dataset_augmentation.api_querying.utils import ErrorCode


class QueryResults(TypedDict):
    title: str
    person: list[str]
    year: str
    err: Optional[ErrorCode]
    api_name: str
