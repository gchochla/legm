import argparse
from copy import deepcopy
from typing import Dict, Any, List

import gridparse


def add_arguments(
    parser: argparse.ArgumentParser,
    arguments: Dict[str, Dict[str, Any]],
    exclude: List[str] = None,
    replace_underscores: bool = False,
):
    """Adds arguments from `arguments` to `parser`.

    Args:
        parser: CL argument parser.
        arguments: a dictionary with name of variable
            as key and kwargs as value.
        exclude: list of arguments to exclude.
        replace_underscores: whether to replace underscores
            with dashes in CL argument.
    """

    exclude = set(exclude or [])
    for k, v in arguments.items():
        if k not in exclude:
            v_argparse = deepcopy(v)
            v_argparse.pop("metadata", None)
            if not isinstance(parser, gridparse.GridArgumentParser):
                v_argparse.pop("searchable", None)
            parser.add_argument(
                f"--{k}"
                if not replace_underscores
                else f"--{k.replace('_', '-')}",
                **v_argparse,
            )


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
