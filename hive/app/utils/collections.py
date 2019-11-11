from typing import List, Any


def safe_remove(lst: List[Any], item: Any) -> List[Any]:
    """
    :param List[Any] lst: list of workers' names that belonged to an hive
    :param item: item to remove from the list
    :returns List[Any] worker_name_list: unmodified or without the specified worker_name
    """
    try:
        lst.remove(item)
        return lst
    except ValueError:
        return lst
