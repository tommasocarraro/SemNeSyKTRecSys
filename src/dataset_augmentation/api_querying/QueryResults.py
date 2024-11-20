from typing import Optional, TypedDict

from src.dataset_augmentation.api_querying.utils import ErrorCode


class QueryResults(TypedDict):
    title: str
    person: list[str]
    year: str
    err: Optional[ErrorCode]
    api_name: str
