import argparse
from copy import deepcopy
from typing import Dict, Any, List


def add_arguments(
    parser: argparse.ArgumentParser,
    arguments: Dict[str, Dict[str, Any]],
    exclude: List[str] = None,
):
    """Adds arguments from `arguments` to `parser`.

    Args:
        parser: CL argument parser.
        arguments: a dictionary with name of variable
            as key and kwargs as value.
        exclude: list of arguments to exclude.
    """

    exclude = set(exclude or [])
    for k, v in arguments.items():
        if k not in exclude:
            v_argparse = deepcopy(v)
            v_argparse.pop("metadata", None)
            parser.add_argument(f"--{k}", **v_argparse)


def add_metadata(
    metadata: Dict[str, Dict[str, Any]],
    arguments: Dict[str, Dict[str, Any]],
    exclude: List[str] = None,
):
    """Adds metadata from `arguments` to `metadata`.

    Args:
        metadata: dict of metadata per parameter.
        arguments: a dictionary with name of variable
            as key and kwargs as value.
        exclude: list of arguments to exclude.
    """

    exclude = set(exclude or [])
    for k, v in arguments.items():
        if k not in exclude:
            param_metadata = v.get("metadata", None)
            if param_metadata:
                metadata[k] = param_metadata
