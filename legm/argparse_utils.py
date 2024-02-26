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
                v_argparse.pop("splits", None)
            parser.add_argument(
                (
                    f"--{k}"
                    if not replace_underscores
                    else f"--{k.replace('_', '-')}"
                ),
                **v_argparse,
            )


def add_metadata(
    metadata_dest: Dict[str, Dict[str, Any]],
    arguments: Dict[str, Dict[str, Any]],
    exclude: List[str] = None,
):
    """Adds metadata from `arguments` to `metadata_dest` in-place.

    Args:
        metadata_dest: dict of destination metadata per parameter.
        arguments: a dictionary with name of variable
            as key and kwargs as value.
        exclude: list of arguments to exclude.

    Raises:
        AssertionError: if a parameter doesn't have `splits`
        and its parent does.
    """

    exclude = set(exclude or [])
    for k, v in arguments.items():
        param_metadata = v.get("metadata", {})
        if k in exclude or not param_metadata:
            continue

        k_has_splits = bool(v.get("splits", []))
        parent = param_metadata.get("parent", None)
        if parent:
            parent_has_splits = bool(arguments[parent].get("splits", []))
        else:
            parent_has_splits = False

        assert (
            k_has_splits or not parent_has_splits
        ), "Parent of a parameter without splits cannot have splits."

        if not k_has_splits:
            # if no splits, add metadata as is
            metadata_dest[k] = param_metadata
        elif k_has_splits and not parent_has_splits:
            # if splits and no parent splits,
            # add metadata in split parameter
            for split in v["splits"]:
                metadata_dest[
                    gridparse.GridArgumentParser._add_split_in_arg(k, split)
                ] = param_metadata
        else:
            # if splits and parent has splits,
            # add metadata in split parameter with same-split parent
            for split in v["splits"]:
                split_metadata = deepcopy(param_metadata)
                split_metadata["parent"] = (
                    gridparse.GridArgumentParser._add_split_in_arg(
                        parent, split
                    )
                )
                metadata_dest[
                    gridparse.GridArgumentParser._add_split_in_arg(k, split)
                ] = split_metadata
