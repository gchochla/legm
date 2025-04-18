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


def parse_args_and_metadata(
    modules: list[Any],
    args: list[dict[str, Any]],
    subparsers: list[str] = None,
    dest: str = None,
    conditional_modules: list[dict[str, Any]] = None,
    conditional_args: list[dict[str, dict[str, Any]]] = None,
) -> tuple[list[gridparse.Namespace], dict[str, dict[str, Any]]]:
    """Parses arguments and returns metadata.

    Args:
        modules: list of modules with `argparse_args` method.
        args: list of dictionaries with arguments.
        subparsers: list of subparsers.
        dest: destination/argument of subparsers.
        conditional_modules: list of conditional modules, i.e. modules per subparser.
        conditional_args: list of conditional arguments, i.e. arguments per subparser.

    Returns:
        tuple with parsed arguments and metadata.
    """

    parser = gridparse.GridArgumentParser()
    metadata = {}

    if not isinstance(modules, list):
        modules = [modules]
    if not isinstance(args, list):
        args = [args]

    if subparsers:
        sp = parser.add_subparsers(dest=dest)

        if conditional_modules:
            if not isinstance(conditional_modules, list):
                conditional_modules = [conditional_modules]
        if conditional_args:
            if not isinstance(conditional_args, list):
                conditional_args = [conditional_args]

        for selector in subparsers:
            selected_sp = sp.add_parser(selector)
            metadata[selector] = {}

            argparse_args = {}
            if conditional_modules:
                current_modules = [
                    m[selector] for m in conditional_modules
                ] + modules
            else:
                current_modules = modules

            for module in current_modules:
                argparse_args.update(module.argparse_args())

            add_arguments(selected_sp, argparse_args, replace_underscores=True)
            add_metadata(metadata[selector], argparse_args)

            for arg in args:
                add_arguments(selected_sp, arg, replace_underscores=True)
                add_metadata(metadata[selector], arg)

            if conditional_args:
                for arg in conditional_args:
                    add_arguments(
                        selected_sp, arg[selector], replace_underscores=True
                    )
                    add_metadata(metadata[selector], arg[selector])

    else:
        argparse_args = {}
        for module in modules:
            argparse_args.update(module.argparse_args())

        add_arguments(parser, argparse_args, replace_underscores=True)
        add_metadata(metadata, argparse_args)

        for arg in args:
            add_arguments(parser, arg, replace_underscores=True)
            add_metadata(metadata, arg)

    return parser.parse_args(), metadata
