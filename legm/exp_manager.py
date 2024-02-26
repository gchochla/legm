import os
import sys
import yaml
import json
import pickle
import re
import pprint
import logging
import shutil
import warnings
from copy import deepcopy
from numbers import Number
from types import SimpleNamespace
from typing import Any, Dict, Optional, Union, Callable, List, Tuple, Sequence
from collections import defaultdict

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from legm.logging_utils import LoggingMixin


class ExperimentManager(LoggingMixin):
    """Module that handles everything that's necessary to compare
    and aggregate metrics across experiments.

    Attributes:
        _directory: main experiment directory.
        _experiment_category: specific experiment being ran
            (also name of experiment subfolder).
        _description: optional information to differentiate experiment
            with others that use the same hyperparams (e.g. for internal
            code change).
        _alternative_experiment_name: optional alternative name for
            experiment subfolder.
        _param_dict: name and value of parameters.
        _disabled_params: which params to not consider for comparison, etc.
        _name_params: which params, if any, to use to name the experiment.
        _metric_dict: name and values of metrics (across epochs/steps).
        _metric_step_dict: name and steps of metrics.
        _best_metric_dict: name and best value of metrics (based on method
            of selection designated by user or last epoch by default).
        _test_metric_dict: name and test value of metrics.
        _metric_dict_indexed: name and values of metrics (across epochs/steps)
            for different IDs.
        _metric_step_dict_indexed: name and steps of metrics for different IDs.
        _best_metric_dict_indexed: name and best value of metrics (based on
            method of selection designated by user or last epoch by default)
            for different IDs.
        _test_metric_dict_indexed: name and test value of metrics for different
            IDs.
        _parent_param_dict: name of parent parameter for some parameter
            (if any). If parent param is `None` or `False`, child param's
            value is not considered.
        _custom_data: custom data to log (keyed by filename).
        _dummy_active: a value of param that is considered active.
        _experiment_folder: actual folder used to log.
        _writer: tensorboard summary writer.
        _time_metric_names: names assumed to include run time.
        _parent_param_value_dict: active value of parent.
        See `LoggingMixin` for other attributes.
    """

    @staticmethod
    def argparse_args():
        return LoggingMixin.argparse_args() | dict(
            description=dict(
                type=str,
                help="description of experiment",
                metadata=dict(disable_comparison=True),
            ),
            alternative_experiment_name=dict(
                type=str,
                help="alternative name for experiment subfolder",
                metadata=dict(disable_comparison=True),
            ),
        )

    def __init__(
        self,
        experiment_root_directory: str,
        experiment_category: str,
        alternative_experiment_name: Optional[str] = None,
        description: Optional[str] = None,
        logging_level: Union[int, str] = logging.INFO,
    ):
        """Init.

        Args:
            experiment_root_directory: main experiment directory.
            experiment_category: specific category of experiment being ran
                (also name of experiment subfolder).
            alternative_experiment_name: optional alternative name for
                experiment subfolder.
            description: optional information to differentiate
                experiment with others that use the same hyperparams
                (e.g. for internal code change).
        """

        super().__init__()

        # basics
        self._directory = experiment_root_directory
        self._experiment_category = experiment_category
        self._alternative_experiment_name = alternative_experiment_name
        self._description = description or ""
        self._experiment_folder = None
        self._writer = None

        self.logging_level = (
            logging_level
            if isinstance(logging_level, int)
            else getattr(logging, logging_level.upper())
        )

        self.logging_file = None

        # log hyperparams
        self._param_dict = {}
        self._disabled_params = set()
        self._name_params = set()
        # if name is transformed, keep correspondence to original name
        self._name_param_correspondence = {}
        self._parent_param_dict = {}
        self._parent_param_value_dict = defaultdict(dict)

        # log metrics
        self._metric_dict = {}
        self._metric_step_dict = {}
        self._best_metric_dict = {}
        self._test_metric_dict = {}

        # log indexed metrics
        self._metric_dict_indexed = {}
        self._metric_step_dict_indexed = {}
        self._best_metric_dict_indexed = {}
        self._test_metric_dict_indexed = {}

        # log custom data
        self._custom_data = {}

        # log time
        self._time_metric_names = ("time", "time_per_sample")

    def __getattr__(self, name):

        if name == "_experiment_category":
            # only reach this point for previous versions of the code
            # that used `_experiment_name` instead of `_experiment_category`
            warnings.warn("`_experiment_name` is deprecated.")
            return self._experiment_name

        try:
            return self._param_dict[name]
        except KeyError:
            raise AttributeError(f"Param {name} not set")

    def __setstate__(self, d):  # for pickling
        self.__dict__ = d
        self._writer = None

    def __getstate__(self):  # for pickling
        return {k: v for k, v in self.__dict__.items() if k not in ("_writer")}

    def _is_inactive(cls, param, child=None) -> bool:
        """Checks if param is inactive."""
        value = cls._param_dict.get(param, True)

        if child is not None:
            active_value = cls._parent_param_value_dict.get(child, {}).get(
                param, None
            )
            if callable(active_value):
                return not active_value(value)
            elif active_value is not None:
                return value != active_value

        return value is None or not value

    _dummy_active = True

    def __eq__(self, __o: object) -> bool:
        """Checks if `_experiment_category` and active params are the same."""

        def comp(eh1: ExperimentHandler, eh2: ExperimentHandler) -> bool:
            """See if active params in `eh1` are equal to the
            corresponding params of `eh2."""
            for param_name, value in eh1._param_dict.items():
                if param_name not in eh1._disabled_params:
                    parent_name = eh1._parent_param_dict.get(param_name, None)

                    # dont compare values that are inactive
                    # or have inactive parent
                    if not (
                        eh1._is_inactive(param_name)
                        or eh1._is_inactive(parent_name, child=param_name)
                    ):
                        # check if params exists and equal in eh2
                        if (
                            param_name not in eh2._param_dict
                            or value != eh2._param_dict[param_name]
                        ):
                            return False
            return True

        if not isinstance(__o, ExperimentHandler):
            return False

        if self._experiment_category != __o._experiment_category:
            return False

        if not (comp(self, __o) and comp(__o, self)):
            return False

        return True

    @classmethod
    def load_existent(
        cls, experiment_subfolder, description=None
    ) -> "ExperimentHandler":
        """Instantiates handler from existing logs.

        Args:
            experiment_subfolder: specific configuration directory.
            descriptions: optional information to differentiate
                experiment with others that use the same hyperparams
                (e.g. for internal code change).

        Returns:
            Handler.
        """

        pkl_filename = os.path.join(experiment_subfolder, "obj.pkl")
        with open(pkl_filename, "rb") as fp:
            obj = pickle.load(fp)
        obj._description = description if description is not None else ""
        return obj

    def start(self):
        """Initiates logging by creating the folder, storing the object
        and the current parameters, and starting Tensorboard."""

        self._experiment_folder = self._get_experiment_folder(
            pattern_matching=False
        )
        self._writer = SummaryWriter(
            os.path.join(self._experiment_folder, "tensorboard")
        )

        # TODO: if run fails, then results are not logged but params are
        # What happens first time in old folder?
        yml_filename = os.path.join(self._experiment_folder, "params.yml")
        # if params exist, load and augment
        if os.path.exists(yml_filename):
            with open(yml_filename) as fp:
                params = yaml.safe_load(fp)

            yml_metrics_filename = os.path.join(
                self._experiment_folder, "metrics.yml"
            )

            # for backwards compatibility
            if not next(iter(params)).startswith("experiment_"):
                if os.path.exists(yml_metrics_filename):
                    # if metrics exist, duplicate existing params for each previous experiment
                    with open(yml_metrics_filename) as fp:
                        metrics = yaml.safe_load(fp)
                else:
                    # failed experiment, so start from scratch
                    metrics = {}

                params = {
                    f"experiment_{i}": params for i in range(len(metrics))
                }

            # if params were logged without metrics, then remove last params
            if os.path.exists(yml_metrics_filename):
                with open(yml_metrics_filename) as fp:
                    metrics = yaml.safe_load(fp)

                if len(params) > len(metrics):
                    params.pop(f"experiment_{len(params)-1}")

            else:
                params = {}

            params[f"experiment_{len(params)}"] = self._param_dict
        else:
            params = {"experiment_0": self._param_dict}

        with open(yml_filename, "w") as fp:
            yaml.dump(params, fp)

        name_filename = os.path.join(self._experiment_folder, "names.txt")
        with open(name_filename, "w") as fp:
            fp.write(
                f"{self._experiment_param_name()}:\n"
                + "\n".join(sorted(self._name_params))
            )

        self.logging_file = os.path.join(self._experiment_folder, "log.txt")
        open(self.logging_file, "a").close()  # touch file

        self.create_logger(
            logging_file=self.logging_file, logging_level=self.logging_level
        )

        pkl_filename = os.path.join(self._experiment_folder, "obj.pkl")
        with open(pkl_filename, "wb") as fp:
            pickle.dump(self, fp)

    def reset(self):
        """Resets tracked metrics, but otherwise keeps hyperparams the same."""
        # metrics
        self._metric_dict = {}
        self._metric_step_dict = {}
        self._best_metric_dict = {}
        self._test_metric_dict = {}

        # indexed metrics
        self._metric_dict_indexed = {}
        self._metric_step_dict_indexed = {}
        self._best_metric_dict_indexed = {}
        self._test_metric_dict_indexed = {}

        # custom data
        self._custom_data = {}

    def tensorboard_write(
        self, name: str, value: Any, step: Optional[int] = None
    ):
        """Logs to Tensorboard. Preferably use first because it
        raises an error if handler not initiated."""

        assert (
            self._writer is not None
        ), "You need to initiate the handler by calling `.start`."

        if isinstance(value, Number):
            self._writer.add_scalar(name, value, global_step=step)
        elif isinstance(value, str):
            self._writer.add_text(name, value, global_step=step)

    def set_parent(
        self, child: str, parent: str, active_value: Optional[Any] = None
    ):
        """Sets parent variable of `child`. Optionally, define when the
        parent is considered active (with certain value or function of
        parent value).

        Args:
            child: str name of child variable.
            parent: str name of parent variable.
            active_value: optional value of parent variable when active
                or function of parent value returning True when active.
        """
        assert (
            parent in self._param_dict
        ), f"{parent} not in parameters of ExperimentManager"
        assert (
            child in self._param_dict
        ), f"{child} not in parameters of ExperimentManager"

        self._parent_param_dict[child] = parent
        self._parent_param_value_dict[child][parent] = active_value

    def set_param(
        self, name: str, value: Any, parent: Optional[str] = None
    ) -> Any:
        """Sets param `name` to `value` and returns `value`.
        The parent of `name` can be specified with `parent`
        (i.e. if the parent is `False` or `None`, the value
        of this parameter won't be considered).

        Args:
            name: str name of variable.
            value: any value.
            parent: str name of optional parent variable.

        Returns:
            The value of the parameter.
        """

        # avoid setting existing params in _param_dict
        if name in self.__dict__ or name in self.argparse_args():
            return value

        self._param_dict[name] = value
        if parent is not None:
            assert (
                parent in self._param_dict
            ), f"{parent} not in parameters of ExperimentManager"
            self.set_parent(name, parent)
        return value

    def set_namespace_params(
        self,
        arg_params: SimpleNamespace,
        parent: Optional[str] = None,
    ) -> SimpleNamespace:
        """`set_param` for names and values in a namespace.
        Returns the namespace."""
        for name, value in arg_params.__dict__.items():
            self.set_param(name, value, parent)
        return arg_params

    def set_param_metadata(
        self,
        param_metadata: Dict[str, Dict[str, Any]],
        arg_params: SimpleNamespace,
    ):
        """Uses metadata dicts per param to set name variables, disable
        comparison of parameters across experiments, and set parents.

        Args:
            param_metadata: dict indexed by param name, with
                potentially the following keys:
                    name: Optional[Union[bool, Callable]],
                    name_transform: Optional[Callable],
                    disable_comparison: Optional[bool],
                    parent: Optional[str]
                    parent_active: Optional[Union[Any, Callable]]
                    name_priority: Optional[int]
            arg_params: parameters and their values in a namespace.
        """

        param_use_name = {}

        for param, metadata in param_metadata.items():
            parent = metadata.get("parent", None)
            if parent is not None:
                self.set_parent(
                    param, parent, metadata.get("parent_active", None)
                )

            disable_comparison = metadata.get("disable_comparison", False)
            if disable_comparison or (
                parent is not None and self._is_inactive(parent, child=param)
            ):
                # disable from comparison if parent is inactive
                # so it shows for its children when the may have "name" set
                self.disable_param(param)

            # if comparison disabled, no need to check if it's in the name
            else:
                use_in_name = metadata.get("name", False)
                curr_param = param
                # if any ancestor has name disabled, do not include in name
                while use_in_name:  # bool(func) is true
                    if curr_param in self._parent_param_dict:
                        parent = self._parent_param_dict[curr_param]
                        if parent in self._disabled_params or (
                            not param_use_name.get(parent, True)
                        ):
                            use_in_name = False

                        curr_param = parent
                    else:
                        break

                if not isinstance(use_in_name, bool):
                    use_in_name = use_in_name(getattr(arg_params, param))

                param_use_name[param] = use_in_name
                original_param = param

                if use_in_name:
                    name_transform = metadata.get("name_transform", None)
                    priority = max(0, metadata.get("name_priority", 0))

                    value = getattr(arg_params, param)
                    if name_transform is not None:
                        value = name_transform(value)

                    if "train" in param:
                        param = param.replace(
                            "train", "ctrain"
                        )  # before dev, eval

                    if (
                        priority > 0
                        or "train" in param
                        or name_transform is not None
                    ):  # if name has been / to be modified in any way
                        param = "_" * priority + param + "___"
                        self.set_param(param, value)
                        self.disable_param(param)

                    self.name_param(param)
                    self._name_param_correspondence[original_param] = param

    def set_dict_params(
        self, dict_params: Dict[str, Any], parent: Optional[str] = None
    ) -> Dict[str, Any]:
        """`set_param` for names and values in a dict.
        Returns the dict."""
        for name, value in dict_params.items():
            self.set_param(name, value, parent)
        return dict_params

    def set_metric(
        self,
        name: str,
        value: Any,
        test: bool = False,
        step: Optional[int] = None,
        id: Optional[str] = None,
    ) -> Any:
        """Sets metric `name` to `value` and returns `value`.
        If `id` is not `None`, the metric is *indexed*.

        Args:
            name: str name of variable.
            value: any value.
            test: whether this is a test or dev metric
                (default is dev, aka False).
            step: steps trained so far, default is not to log step.
            id: optional id to index metric.

        Returns:
            The value of the metric.
        """

        # for numpy floats, etc that mess up yaml
        if isinstance(value, int):
            value = int(value)
        elif isinstance(value, Number):
            value = float(value)

        if not test:
            if name.startswith("best_"):
                name = "_" + name
            if id is not None:
                self._metric_dict_indexed.setdefault(id, {}).setdefault(
                    name, []
                ).append(value)
                self._metric_step_dict_indexed.setdefault(id, {}).setdefault(
                    name, []
                ).append(step)
            else:
                self._metric_dict.setdefault(name, []).append(value)
                self._metric_step_dict.setdefault(name, []).append(step)
        else:
            if id is not None:
                self._test_metric_dict_indexed.setdefault(id, {})[
                    "test_" + name
                ] = value
            else:
                self._test_metric_dict["test_" + name] = value
        return value

    def set_dict_metrics(
        self,
        metrics_dict: Union[Dict[str, Any], Dict[str, Dict[str, Any]]],
        test: bool = False,
        step: Optional[int] = None,
    ) -> Dict[str, Any]:
        """`set_metric` for names and values in a dict.
        Handles additionally indexed dict (index is first key).
        Returns the input dict."""

        for k, v in metrics_dict.items():
            if isinstance(v, dict):
                for kk, vv in v.items():
                    self.set_metric(kk, vv, test=test, step=step, id=k)
            else:
                self.set_metric(k, v, test=test, step=step)

        return metrics_dict

    def set_custom_data(self, data, basename: Optional[str] = None):
        """Sets custom data to designated file (default yml).

        Args:
            data: data to log.
            basename: optional filename to log to.

        Raises:
            NotImplementedError: if `basename` already exists,
            and `data` is of type whose update is not implemented.

            AssertionError: if `basename` is not a valid format
            (judged by extension).
        """

        if basename is None:
            basename = "custom_data.yml"
        basename.replace("yaml", "yml")

        format = os.path.splitext(basename)[1][1:]
        assert format in ("yml", "txt", "json", "pkl"), "Invalid format"

        filename = self.get_save_filename(basename=basename)

        if filename in self._custom_data:
            existing_data = self._custom_data[filename]["data"]
            if isinstance(existing_data, dict):
                existing_data.update(data)
                data = existing_data
            elif isinstance(existing_data, list):
                existing_data.extend(data)
                data = existing_data
            elif isinstance(existing_data, str):
                data = existing_data + "\n" + data
            elif isinstance(existing_data, Number):
                data = existing_data + data
            else:
                raise NotImplementedError(
                    f"Update to custom data not implemented for type {type(existing_data)}"
                )

        self._custom_data[filename] = dict(data=data, format=format)

    def disable_param(self, name: str):
        """Disables parameter `name` from comparison to other configurations."""
        if name in self.__dict__ or name in self.argparse_args():
            return
        assert (
            name in self._param_dict
        ), f"{name} not in parameters of ExperimentManager"
        self._disabled_params.add(name)

    def disable_params(self, names: List[str]):
        """Disables parameters in `names` from comparison
        to other configurations."""
        for name in names:
            self.disable_param(name)

    def name_param(self, name: str):
        """Sets parameter `name` from usage in naming of experiment."""
        assert (
            name in self._param_dict
        ), f"{name} not in parameters of ExperimentManager"
        self._name_params.add(name)

    def name_params(self, names: List[str]):
        """Sets name parameters."""
        for name in names:
            self.name_param(name)

    def capture_metrics(self, metric_names=None) -> Callable:
        """Decorator that captures return values of functions as metrics.

        If the function returns a list (or equivalent), the `metric_names`
        must be set, in order, to get the name of the metrics. If it is a
        dict, then we assume keys are variable names.

        Args:
            metric_names: if the returned values of the function are not in
                a dict, the names of the metrics.

        Returns:
            A function that records the metrics and then returns them.
        """

        def actual_decorator(fun):
            def wrapper(*args, **kwargs):
                results = fun(*args, **kwargs)
                if not hasattr(results, "__len__"):
                    results = [results]

                if metric_names is None:
                    assert isinstance(results, dict)
                    self.set_dict_metrics(results)
                else:
                    self.set_dict_metrics(
                        {k: v for k, v in zip(metric_names, results)}
                    )
                return results

            return wrapper

        return actual_decorator

    def get_save_filename(self, basename: str = "model.pt") -> str:
        """Returns filename within the experiment folder
        to save object (default: PyTorch model)."""
        config_directory = self._get_experiment_folder(pattern_matching=False)
        model_filename = os.path.join(
            config_directory,
            (self._description + "-" if self._description else "") + basename,
        )
        return model_filename

    @staticmethod
    def __format_string(s):
        if isinstance(s, str):
            return (
                s.replace(os.sep, "--")
                .replace(",", "---")
                .replace("=", "--eq--")
            )
        if hasattr(s, "__len__"):
            return "+".join([ExperimentManager.__format_string(ss) for ss in s])
        return str(s)

    def _experiment_param_name(self) -> str:
        """Returns name of experiment based on name parameters."""

        return ",".join(
            [
                self.__format_string(self._param_dict[param])
                for param in sorted(self._name_params)
            ]
        )

    def _format_experiment_name(self, name: str) -> str:
        for param in self._param_dict:

            value = self._param_dict[
                self._name_param_correspondence.get(param, param)
            ]

            name = name.replace(
                "{" + param + "}", self.__format_string(str(value))
            )
        return name

    def _get_experiment_folder(
        self, pattern_matching: bool = False
    ) -> Union[str, List[str]]:
        """Returns name of directory of experiment (and creates it if
        it doesn't exist).

        Args:
            pattern_matching: whether to return the name of the current
                configuration or match child variables whose parents
                are deactivated.

        Returns:
            The folder name if `pattern_matching==False`, otherwise all
            equivalent folder names (based on parent variable values).
        """

        if self._experiment_folder is not None and not pattern_matching:
            return self._experiment_folder

        def strict__eq__(eh1: ExperimentHandler, eh2: ExperimentHandler):
            def sorted_dict(d):
                if not isinstance(d, dict):
                    return d
                return {k: sorted_dict(d[k]) for k in sorted(d)}

            eh1_dict = sorted_dict(eh1._param_dict)
            eh2_dict = sorted_dict(eh2._param_dict)
            return eh1_dict == eh2_dict

        # setup general subfolder
        experiment_subfolder = os.path.join(
            self._directory, self._experiment_category
        )
        if not os.path.exists(experiment_subfolder):
            os.makedirs(experiment_subfolder)

        exact_match_subfolder = None
        config_subfolders = []
        for subexperiment_subfolder in os.listdir(experiment_subfolder):
            abs_subexperiment_subfolder = os.path.join(
                experiment_subfolder, subexperiment_subfolder
            )

            try:
                obj = ExperimentHandler.load_existent(
                    abs_subexperiment_subfolder
                )
            except EOFError:
                continue

            if strict__eq__(self, obj):
                exact_match_subfolder = abs_subexperiment_subfolder

            elif self == obj:
                config_subfolders.append(abs_subexperiment_subfolder)

        # if an exact match was not found
        if exact_match_subfolder is None:
            # if only another experiment with the same active params exists
            if len(config_subfolders) == 1:
                exact_match_subfolder = config_subfolders[0]
            # otherwise, more than one equivalent exists, build another one
            else:
                exact_match_subfolder = os.path.join(
                    experiment_subfolder,
                    self._experiment_param_name() + "_0",
                )
                while os.path.exists(exact_match_subfolder):
                    split_name = exact_match_subfolder.split("_")
                    name, index = split_name[:-1], split_name[-1]
                    exact_match_subfolder = (
                        "_".join(name) + "_" + str(int(index) + 1)
                    )

        if self._alternative_experiment_name:
            alternative_subfolder = self._format_experiment_name(
                os.path.join(
                    experiment_subfolder, self._alternative_experiment_name
                )
            )

        # if experiment doesn't exist, create it
        if not os.path.exists(exact_match_subfolder):
            if self._alternative_experiment_name:
                # if alternative is provided, create that one
                exact_match_subfolder = alternative_subfolder
            os.makedirs(exact_match_subfolder)
        # if experiment exists and alternative name is provided
        # move it to potential alternative subfolder
        elif self._alternative_experiment_name:
            shutil.move(exact_match_subfolder, alternative_subfolder)
            exact_match_subfolder = alternative_subfolder

        return (
            [exact_match_subfolder] + config_subfolders
            if pattern_matching
            else exact_match_subfolder
        )

    def _parse_experiments_from_configs(
        self, config_directories: List[str]
    ) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, List[int]]]:
        """Parses experiments and steps from all directories in two dictionaries.

        Args:
            config_directories: experiment directories.

        Returns:
            A single dictionary with experiment logs,
            and a single dictionary with the corresponding steps.
        """

        experiments = {}
        indexed_experiments = {}
        steps = {}
        for config_directory in config_directories:
            metrics_filename = os.path.join(config_directory, "metrics.yml")
            indexed_metrics_filename = os.path.join(
                config_directory, "indexed_metrics.yml"
            )
            step_filename = os.path.join(config_directory, "steps.yml")

            assert os.path.exists(step_filename), (
                f"{step_filename} does not exist. "
                "Please run `log_metrics` before `_parse_experiments_from_configs`."
            )

            if os.path.exists(metrics_filename):
                with open(metrics_filename) as fp:
                    config_experiments = yaml.safe_load(fp)
            else:
                config_experiments = {}

            if os.path.exists(indexed_metrics_filename):
                with open(indexed_metrics_filename) as fp:
                    config_experiments_indexed = yaml.safe_load(fp)
            else:
                config_experiments_indexed = {}

            if os.path.exists(step_filename):
                with open(step_filename) as fp:
                    config_steps = yaml.safe_load(fp)
            else:
                config_steps = {}

            for metric in config_steps:
                if len(steps.get(metric, [])) < len(config_steps[metric]):
                    steps[metric] = config_steps[metric]

            experiments.update(
                {
                    f"experiment_{len(experiments)+i}": config_experiments[
                        f"experiment_{i}"
                    ]
                    for i in range(len(config_experiments))
                }
            )

            indexed_experiments.update(
                {
                    f"experiment_{len(indexed_experiments)+i}": config_experiments_indexed[
                        f"experiment_{i}"
                    ]
                    for i in range(len(config_experiments_indexed))
                }
            )

        return experiments, indexed_experiments, steps

    def set_best(self, method: str, **kwargs):
        """Sets `best_metric_dict` based on lists in `metric_dict`.

        Args:
            method: how to determine which step contains the
                "best" metrics. Can be `"early_stopping"`, `"last"`.
            kwargs: must contain `"step"`, `"best_step"` or `"metric"`
                and "higher_better" if `method=="early_stopping"`.
        """

        step = None
        idx = -1  # method == "last"

        if method == "early_stopping":
            assert ("step" in kwargs or "best_step" in kwargs) or (
                "metric" in kwargs and "higher_better" in kwargs
            ), "Must provide [best_]step or used metric for early_stopping"

            if "step" in kwargs:
                step = kwargs["step"]
            elif "best_step" in kwargs:
                step = kwargs["best_step"]
            else:
                metric = kwargs["metric"]
                argopt = (
                    (lambda x: np.argmax(x))
                    if kwargs["higher_better"]
                    else (lambda x: np.argmin(x))
                )

                idx = argopt(self._metric_dict[metric])

        for metric_name in self._metric_dict:
            if metric_name not in self._time_metric_names:
                if step is not None:
                    idx = self._metric_step_dict[metric_name].index(step)
                self._best_metric_dict[f"best_{metric_name}"] = (
                    self._metric_dict[metric_name][idx]
                )

        for _id in self._metric_dict_indexed:
            for metric_name in self._metric_dict_indexed[_id]:
                if metric_name not in self._time_metric_names:
                    if step is not None:
                        idx = self._metric_step_dict_indexed[_id][
                            metric_name
                        ].index(step)
                    self._best_metric_dict_indexed.setdefault(_id, {})[
                        f"best_{metric_name}"
                    ] = self._metric_dict_indexed[_id][metric_name][idx]

    def log_metrics(self):
        """Logs all metrics."""
        config_directory = self._get_experiment_folder(pattern_matching=False)
        metric_filename = os.path.join(config_directory, "metrics.yml")
        indexed_metric_filename = os.path.join(
            config_directory, "indexed_metrics.yml"
        )
        step_filename = os.path.join(config_directory, "steps.yml")

        if os.path.exists(metric_filename):
            with open(metric_filename) as fp:
                experiments = yaml.safe_load(fp)
        else:
            experiments = {}

        if os.path.exists(indexed_metric_filename):
            with open(indexed_metric_filename) as fp:
                indexed_experiments = yaml.safe_load(fp)
        else:
            indexed_experiments = {}

        if os.path.exists(step_filename):
            with open(step_filename) as fp:
                steps = yaml.safe_load(fp)
        else:
            steps = {}

        if not self._best_metric_dict:
            self.set_best("last")

        experiment = deepcopy(self._metric_dict)
        experiment.update(self._best_metric_dict)
        experiment.update(self._test_metric_dict)

        indexed_experiment = deepcopy(self._metric_dict_indexed)
        for _id, value in self._best_metric_dict_indexed.items():
            indexed_experiment[_id].update(value)
        for _id, value in self._test_metric_dict_indexed.items():
            indexed_experiment.setdefault(_id, {}).update(value)

        # write steps
        for metric in self._metric_step_dict:
            if len(steps.get(metric, [])) < len(self._metric_step_dict[metric]):
                steps[metric] = self._metric_step_dict[metric]

        with open(step_filename, "w") as fp:
            yaml.dump(steps, fp)

        # write metrics
        if experiment:
            experiment.update({"description": self._description})

            experiments[f"experiment_{len(experiments)}"] = experiment

            with open(metric_filename, "w") as fp:
                yaml.dump(experiments, fp)

        if indexed_experiment:
            indexed_experiment.update({"description": self._description})

            indexed_experiments[f"experiment_{len(indexed_experiments)}"] = (
                indexed_experiment
            )

            with open(indexed_metric_filename, "w") as fp:
                yaml.dump(indexed_experiments, fp)

        for fn, info in self._custom_data.items():
            exists = os.path.exists(fn)
            format = info["format"]
            data = info["data"]
            if format in ("yml", "json"):
                if exists:
                    with open(fn) as fp:
                        if format == "yml":
                            existing_data = yaml.safe_load(fp)
                        else:
                            existing_data = json.load(fp)

                    data = {
                        **existing_data,
                        f"experiment_{len(existing_data)}": data,
                    }
                else:
                    # create wrapper dictionaries with #experiment in key
                    data = {f"experiment_0": data}

                with open(fn, "w") as fp:
                    if format == "yml":
                        yaml.dump(data, fp)
                    else:
                        json.dump(data, fp)
            elif format == "pkl":
                if exists:
                    with open(fn, "rb") as fp:
                        existing_data = pickle.load(fp)
                    data = {
                        **existing_data,
                        f"experiment_{len(existing_data)}": data,
                    }
                else:
                    data = {f"experiment_0": data}

                with open(fn, "wb") as fp:
                    pickle.dump(data, fp)
            else:
                with open(fn, "a") as fp:
                    if not isinstance(data, str):
                        data = pprint.pformat(data)
                        if exists:
                            data = "\n" + data
                    fp.write(data)

    def aggregate_results(
        self, aggregation: Union[str, Dict[str, str]] = "mean"
    ):
        """Presents aggregated results of current configuration.

        Args:
            aggregation: how to aggregate best metrics across experiments.
                Can be an `str` (`"mean"` which includes std, or `"median`")
                or a dict whose keys are the metrics. The key `"other"` is
                reserved to be used as the default for each metric not specified
                in the the dict, and if that is not specified, it defaults to
                `"mean"`, which is also the general default.
        """

        def aggregate(method, values):
            if method == "mean":
                aggregated_value = (
                    f"{np.mean(values):.4f}+-{np.std(values):.4f}"
                )
            elif method == "median":
                aggregated_value = f"{np.median(values):.4f}"
            elif method == "outlier_mean":
                # preferably removes from best results
                n_outliers = 2 * len(values) // 10
                values = sorted(values)[
                    n_outliers // 2 : len(values) - (n_outliers + 1) // 2
                ]
                aggregated_value = (
                    f"{np.mean(values):.4f}+-{np.std(values):.4f}"
                )

            return aggregated_value

        if isinstance(aggregation, str):
            aggregation = {"other": aggregation}

        default_aggregation = aggregation.setdefault("other", "mean")

        config_directories = self._get_experiment_folder(pattern_matching=True)

        (
            experiments,
            indexed_experiments,
            _,
        ) = self._parse_experiments_from_configs(config_directories)

        best_metrics = {}
        test_metrics = {}
        time_metrics = {}

        # collect metrics
        for experiment in experiments.values():
            if experiment.pop("description") != self._description:
                continue
            for metric, value in experiment.items():
                if metric.startswith("best_") and not isinstance(value, list):
                    best_metrics.setdefault(metric, []).append(value)

                elif metric.startswith("test_") and not isinstance(value, list):
                    test_metrics.setdefault(metric, []).append(value)
                elif metric in self._time_metric_names:
                    time_metrics.setdefault(metric, []).extend(value)

        indexed_best_metrics = defaultdict(dict)
        indexed_test_metrics = defaultdict(dict)
        indexed_time_metrics = defaultdict(dict)

        # collect indexed metrics
        for experiment in indexed_experiments.values():
            if experiment.pop("description") != self._description:
                continue
            for _id, metric_dict in experiment.items():
                for metric, value in metric_dict.items():
                    if metric.startswith("best_") and isinstance(value, Number):
                        indexed_best_metrics[_id].setdefault(metric, []).append(
                            value
                        )

                    elif metric.startswith("test_") and isinstance(
                        value, Number
                    ):
                        indexed_test_metrics[_id].setdefault(metric, []).append(
                            value
                        )
                    elif metric in self._time_metric_names:
                        indexed_time_metrics[_id].setdefault(metric, []).extend(
                            value
                        )

        # aggregate metrics
        aggregated_metrics = {}
        for metric, values in best_metrics.items():
            method = aggregation.get(
                metric[len("best_") :], default_aggregation
            )

            aggregated_metrics[metric] = aggregate(method, values)

        for metric, values in test_metrics.items():
            method = aggregation.get(
                metric[len("test_") :], default_aggregation
            )
            aggregated_metrics[metric] = aggregate(method, values)

        for metric, values in time_metrics.items():
            aggregated_metrics[metric] = aggregate("mean", values)

        # aggregate indexed metrics
        aggregated_indexed_metrics = {}
        for _id, metric_dict in indexed_best_metrics.items():
            for metric, values in metric_dict.items():
                method = aggregation.get(
                    metric[len("best_") :], default_aggregation
                )

                aggregated_indexed_metrics.setdefault(_id, {})[metric] = (
                    aggregate(method, values)
                )

        for _id, metric_dict in indexed_test_metrics.items():
            for metric, values in metric_dict.items():
                method = aggregation.get(
                    metric[len("test_") :], default_aggregation
                )

                aggregated_indexed_metrics.setdefault(_id, {})[metric] = (
                    aggregate(method, values)
                )

        for _id, metric_dict in indexed_time_metrics.items():
            aggregated_indexed_metrics[_id] = {}
            for metric, values in metric_dict.items():
                aggregated_indexed_metrics.setdefault(_id, {})[metric] = (
                    aggregate("mean", values)
                )

        # write results
        experiment_folder = self._get_experiment_folder(pattern_matching=False)
        results_filename = os.path.join(
            experiment_folder, "aggregated_metrics.yml"
        )
        indexed_results_filename = os.path.join(
            experiment_folder, "aggregated_indexed_metrics.yml"
        )

        # aggregated results
        if os.path.exists(results_filename):
            with open(results_filename) as fp:
                results = yaml.safe_load(fp)
        else:
            results = {}

        if aggregated_metrics:
            results[self._description] = aggregated_metrics

        if results:
            with open(results_filename, "w") as fp:
                yaml.dump(results, fp)

        # aggregated indexed results
        if os.path.exists(indexed_results_filename):
            with open(indexed_results_filename) as fp:
                indexed_results = yaml.safe_load(fp)
        else:
            indexed_results = {}

        if aggregated_indexed_metrics:
            indexed_results[self._description] = aggregated_indexed_metrics

        if indexed_results:
            with open(indexed_results_filename, "w") as fp:
                yaml.dump(indexed_results, fp)

    def plot(
        self,
        aggregation: Union[str, Dict[str, str]] = "mean",
        groups: Optional[List[List[str]]] = None,
        exclude: Optional[List[str]] = None,
        exclude_pattern: Optional[str] = None,
    ):
        """Plots progression of metrics of current configuration.

        Args:
            aggregation: how to aggregate best metrics across experiments.
                Can be an `str` (`"mean"` which includes std, or `"median`")
                or a dict whose keys are the metrics. The key `"other"` is
                reserved to be used as the default for each metric not specified
                in the the dict, and if that is not specified, it defaults to
                `"mean"`, which is also the general default.
            groups: how to group metrics in plots. The ones left unspecified will
                be plotted separately from the rest and each other. Note that we
                also retain the order specified in the group.
            exclude: which metrics to exclude.
            exclude_pattern: regex pattern to use to exclude metrics.
        """

        if isinstance(aggregation, str):
            aggregation = {"other": aggregation}

        default_aggregation = aggregation.setdefault("other", "mean")

        config_directories = self._get_experiment_folder(pattern_matching=True)

        experiments, _, steps = self._parse_experiments_from_configs(
            config_directories
        )

        for key in list(experiments):
            experiment = experiments[key]
            description = experiment.pop("description")
            if self._description == description:
                for metric in list(experiment):
                    if (
                        metric.startswith("best_")
                        or metric.startswith("test_")
                        or metric in self._time_metric_names
                    ):
                        experiment.pop(metric)
                    if metric.startswith("_best"):
                        experiment[metric[1:]] = experiment.pop(metric)
                if not experiment:
                    # in case this has nothing left (e.g. a pure test entry)
                    experiments.pop(key)
            else:
                experiments.pop(key)

        if experiments:
            plot_directory = os.path.join(
                self._get_experiment_folder(pattern_matching=False),
                (f"{self._description}-" if self._description else "")
                + "plots",
            )

            if not os.path.exists(plot_directory):
                os.makedirs(plot_directory)

            all_metrics = sorted(list(next(iter(experiments.values()))))
            if groups is None:
                # assume each metric goes into its own plot
                groups = [[metric] for metric in all_metrics]
            else:
                # find if there are metrics not specified in `groups`
                left_out_metrics = sorted(
                    set(all_metrics).difference(
                        [metric for group in groups for metric in group]
                    )
                )

                # add them on their own plot
                groups.extend([[metric] for metric in left_out_metrics])

            for metrics in groups:
                metrics = set(metrics).difference(exclude or set())
                metrics = list(
                    metrics.difference(
                        [
                            metric
                            for metric in metrics
                            if exclude_pattern is not None
                            and re.fullmatch(exclude_pattern, metric)
                        ]
                    )
                )
                if not metrics:
                    continue
                # necessary because of potential early stopping
                max_length = max(
                    # assume all metrics have same length
                    [len(experiments[exp][metrics[0]]) for exp in experiments]
                )

                color = plt.cm.rainbow(np.linspace(0, 1, len(metrics)))

                for i, metric in enumerate(metrics):
                    method = aggregation.get(metric, default_aggregation)

                    group_exists = len(metrics) > 1

                    # get all values across experiments for each step
                    values_per_step = [
                        [
                            experiments[exp][metric][j]
                            for exp in experiments
                            if len(experiments[exp][metric]) > j
                        ]
                        for j in range(max_length)
                    ]

                    # aggregate
                    if method == "mean":
                        ys = np.array(
                            [np.mean(values) for values in values_per_step]
                        )
                        es = np.array(
                            [np.std(values) for values in values_per_step]
                        )
                    elif method == "median":
                        ys = np.array(
                            [np.mean(values) for values in values_per_step]
                        )
                        es = None

                    xs = steps[metric]
                    if any([step is None for step in xs]):
                        xs = range(len(ys))

                    for x, y in zip(xs, ys):
                        self.tensorboard_write(metric, y, x)

                    plt.plot(
                        xs,
                        ys,
                        label=metric if group_exists else None,
                        c=color[i],
                    )
                    if es is not None:
                        plt.fill_between(
                            xs, ys + es, ys - es, facecolor=color[i], alpha=0.2
                        )

                    # if single metric, then y label
                    if not group_exists:
                        plt.ylabel(
                            metric.title().replace("_", " "),
                            rotation=45,
                            labelpad=25,
                        )
                    # else, legend
                    else:
                        plt.legend(
                            bbox_to_anchor=(1.1, 1.05),
                            ncol=1,
                            fancybox=True,
                            shadow=True,
                        )

                # make xticks start from 1 and up to 11 if steps exceed that
                xticks = [1] + list(
                    range(0, max_length + 1, max(1, max_length // 10))
                )[1:]
                xticks = [xs[i - 1] for i in xticks]
                plt.xticks(xticks, xticks, rotation=45)
                plt.xlabel("Steps")
                plt.tight_layout()

                plot_basename = ",".join(metrics) + ".png"
                max_fname_len = os.statvfs("/")[-1]
                if len(plot_basename) > max_fname_len:
                    max_metric_len = (max_fname_len - len(".png")) // len(
                        metrics
                    )
                    plot_basename = (
                        ",".join(
                            # -1 for ","
                            [metric[: max_metric_len - 1] for metric in metrics]
                        )
                        + ".png"
                    )

                plot_filename = os.path.join(plot_directory, plot_basename)
                plt.savefig(plot_filename)
                plt.clf()

    def _eval_metric_name(self, metric: str, test: bool) -> str:
        """Returns the name of the metric in the configuration."""
        if metric in self._time_metric_names:
            return metric
        return ("test" if test else "best") + "_" + metric

    def get_best(
        self,
        eval_metric: str,
        higher_better: bool = True,
        test: bool = False,
        **kwargs,
    ) -> Tuple[Dict[str, Dict[str, Dict[str, str]]], List[str]]:
        """Computes the best experiment based on the given metric.

        Args:
            eval_metric: The metric to use for evaluation.
            higher_better: Whether a higher value is better. Defaults to True.
            test: Whether to use the test set. Defaults to False.
            kwargs: constrain the search space by specifying key-value pairs,
                values are `list`s of `str`s.

        Returns:
            A tuple containing the configuration and the results of all
            experiments in a dictionary, its keys ordered by the evaluation
            metric, and the actual name of the evaluation metric in the
            configurations.

        Raises:
            ValueError: If no experiment matches the constraints.
        """

        cfg_metrics = {}
        for subfolder in os.listdir(
            os.path.join(self._directory, self._experiment_category)
        ):
            try:
                with open(
                    os.path.join(
                        self._directory,
                        self._experiment_category,
                        subfolder,
                        "aggregated_metrics.yml",
                    )
                ) as fp:
                    aggregated_metrics = yaml.load(fp, Loader=yaml.Loader)[
                        self._description or ""
                    ]

                with open(
                    os.path.join(
                        self._directory,
                        self._experiment_category,
                        subfolder,
                        "params.yml",
                    )
                ) as fp:
                    config = yaml.load(fp, Loader=yaml.Loader)

                satisfy_constraints = True
                for k, v in kwargs.items():
                    # assume that the value is a list of strings
                    satisfy_constraints = str(config.get(k, None)) in v
                    if not satisfy_constraints:
                        break

                if satisfy_constraints:
                    cfg_metrics[subfolder] = {
                        "metrics": aggregated_metrics,
                        "config": config,
                    }

            except FileNotFoundError:
                ...

        if not cfg_metrics:
            raise ValueError(
                "No experiments found for the given constraints: {}".format(
                    kwargs
                )
            )

        eval_metric_key = self._eval_metric_name(eval_metric, test)
        order = sorted(
            cfg_metrics,
            key=lambda x: float(
                cfg_metrics[x]["metrics"][eval_metric_key][
                    # this will include a +- for stddev
                    : cfg_metrics[x]["metrics"][eval_metric_key].find("+")
                ]
                if eval_metric_key in cfg_metrics[x]["metrics"]
                else (-2 * higher_better + 1) * float("inf")
            ),
            reverse=higher_better,
        )

        return cfg_metrics, order

    # TODO: fix conflict between x axis and constraints
    def plot_comparison_bars(
        self,
        x_axis: str,
        y_axis: List[str],
        filename: str,
        x_values: Optional[List[str]] = None,
        title: Optional[str] = None,
        y_axis_alternative_names: Optional[List[str]] = None,
        higher_better: bool = True,
        test: bool = False,
        nested_x_axis: Optional[str] = None,
        nested_x_values: Optional[List[str]] = None,
        offset_y_axis_labels: Union[int, Sequence[int]] = 0,
        offset_x_axis_labels: Union[int, Sequence[int]] = 0,
        y_limit: Optional[float] = None,
        **kwargs,
    ):
        """Plots a bar chart comparing the given alternatives.

        Args:
            x_axis: The x-axis label.
            y_axis: The y-axis labels. First will be used to
                determine best configuration.
            filename: The filename to save the plot to.
            higher_better: Whether a higher value is better.
            test: Whether to use the test set results.
            alternatives: The alternatives to compare.
            nested_x_axis: The nested x-axis label.
            nested_x_values: The nested x-axis values.
            offset_y_axis_labels: The number of pixels to offset the
                y-axis labels by (they are placed on x-axis under their
                corresponding bars, assumption is lenght is about one bar
                width).
            offset_x_axis_labels: The number of pixels to offset the
                x-axis labels by (they are placed under the entire section of
                bars corresponding to them, assumption is lenght is about one
                bar width).
            y_limit: The maximum value for the y-axis.
            kwargs: constrain the search space by specifying key-value pairs,
                values are `list`s of `str`s.
        """

        assert nested_x_axis is None or nested_x_values is not None

        # if x_values are not provided, pick all available
        if x_values is None:
            x_values = []
            for subfolder in os.listdir(
                os.path.join(self._directory, self._experiment_category)
            ):
                try:
                    with open(
                        os.path.join(
                            self._directory,
                            self._experiment_category,
                            subfolder,
                            "params.yml",
                        )
                    ) as fp:
                        config = yaml.load(fp, Loader=yaml.Loader)

                    x_values.append(config[x_axis])

                except FileNotFoundError:
                    ...

            x_values = list(map(str, sorted(list(set(x_values)))))

        if y_axis_alternative_names is None:
            y_axis_alternative_names = y_axis

        # make sure offsets match lengths of labels
        if isinstance(offset_y_axis_labels, int):
            offset_y_axis_labels = np.array(
                [offset_y_axis_labels] * len(y_axis)
            )
        else:
            assert len(offset_y_axis_labels) == len(y_axis)
            offset_y_axis_labels = np.array(offset_y_axis_labels)

        if isinstance(offset_x_axis_labels, int):
            offset_x_axis_labels = np.array(
                [offset_x_axis_labels] * len(x_values)
            )
        else:
            assert len(offset_x_axis_labels) == len(x_values)
            offset_x_axis_labels = np.array(offset_x_axis_labels)

        # get the results for each alternative
        results = {}
        for nested_x_value in nested_x_values or [None]:
            for x_value in x_values:
                results.setdefault(nested_x_value, {})
                for nested_y_value in y_axis:
                    results[nested_x_value].setdefault(nested_y_value, {})

                # only first y label used to determine best configuration

                cfg, order = self.get_best(
                    eval_metric=y_axis[0],
                    higher_better=higher_better,
                    test=test,
                    **{
                        x_axis: [x_value],
                        **(
                            {nested_x_axis: [nested_x_value]}
                            if nested_x_axis
                            else {}
                        ),
                        **kwargs,
                    },
                )

                # for that best configuration, grab all the metrics
                for nested_y_value in y_axis:
                    key = self._eval_metric_name(nested_y_value, test)
                    results[nested_x_value][nested_y_value][x_value] = float(
                        cfg[order[0]]["metrics"][key][
                            : cfg[order[0]]["metrics"][key].find("+")
                        ]
                    )

        bar_width = 0.09
        fig, ax = plt.subplots()
        colors = [
            matplotlib.cm.rainbow(i / len(nested_x_values or [None]))
            for i in range(len(nested_x_values or [None]))
        ]

        # keep height range to position y labels on the x axis
        height_range = [float("inf"), -float("inf")]  # min, max

        for k, metric in enumerate(y_axis):
            for j, nested_x_value in enumerate(results):
                heights = []
                for x_value in x_values:
                    val = results[nested_x_value][metric][x_value]
                    if val < height_range[0]:
                        height_range[0] = val
                    elif val > height_range[1]:
                        height_range[1] = val

        height_range = [height_range[0], y_limit or height_range[1]]

        for k, metric in enumerate(y_axis):
            for j, nested_x_value in enumerate(results):
                heights = []
                for x_value in x_values:
                    heights.append(results[nested_x_value][metric][x_value])

                pos = np.arange(len(heights)) * (
                    # 2 bar gap between bar sections + 1 which is the default
                    ((len(results) + 1) * len(y_axis) + 2)
                    * bar_width
                )
                ax.bar(
                    (
                        pos + k * bar_width * (len(results) + 1) + j * bar_width
                    ).tolist(),
                    heights,
                    bar_width,
                    label=metric + str(j),
                    color=colors[j],
                )

                if j == len(results) - 1:
                    for j, po in enumerate(pos):
                        ax.text(
                            po
                            # move as bars as there are nested x values + gap between nested x values
                            + k * bar_width * (len(results) + 1)
                            # move to the middle of current bars
                            # (we dont consider gap part of bars for text purposes)
                            + len(results) * bar_width / 2
                            # adjust because text is not a point (assume about one bar)
                            # + starting point is after first bar
                            - 2 * bar_width - offset_y_axis_labels[k],
                            -0.065
                            * (height_range[1] - min(height_range[0], 0)),
                            (y_axis_alternative_names[k] or metric)
                            .replace("_", " ")
                            .title(),
                            size=6,
                            fontweight="bold",
                            rotation=15,
                        )

        ax.tick_params(axis="x", which="both", bottom=False, top=False, pad=19)
        ax.set_xticks(
            # move to middle of whole current section of nested_x_values + gap for each
            # y_axis value, adjust one bar back because starting after the first bar
            bar_width * (len(y_axis) * (len(results) + 1) / 2 - 1)
            + pos
            - offset_x_axis_labels
        )
        ax.set_xticklabels([x_value.title() for x_value in x_values])

        ax.set_xlabel(x_axis.replace("_", " ").title(), labelpad=6)

        if nested_x_axis:
            ax.legend(
                [
                    f"{nested_x_axis}={nested_x_value}"
                    for nested_x_value in nested_x_values
                ]
            )

        ax.set_title(title or "")
        plt.ylim(top=height_range[1] * 1.1)
        # fig.tight_layout()
        fig.savefig(filename, bbox_inches="tight")


def get_best():
    """Retrieves the best experiment based on the given metric.
    Use --help for more information."""

    if "--help" in sys.argv or "-h" in sys.argv:
        print(
            "Usage: python -m experiment_handler.get_best "
            "<experiment_folder> <eval_metric> <higher_better> <test> "
            "[--<config_key> <config_value> ...]"
        )
        sys.exit(0)

    experiment_folder = os.path.abspath(sys.argv[1])
    eval_metric = sys.argv[2]
    higher_better = bool(sys.argv[3].title() == "True")
    test = bool(sys.argv[4].title() == "True")
    kwargs = {}
    current_arg = None
    for arg in sys.argv[5:]:
        if arg.startswith("--"):
            current_arg = arg[2:]
            kwargs[current_arg] = []
        else:
            kwargs[current_arg].append(arg)

    e = ExperimentHandler(*os.path.split(experiment_folder))
    cfg_metrics, order = e.get_best(
        eval_metric, higher_better=higher_better, test=test, **kwargs
    )

    eval_metric_key = e._eval_metric_name(eval_metric, test=test)

    best_config = cfg_metrics[order[0]]["config"]
    best_value = cfg_metrics[order[0]]["metrics"][eval_metric_key]
    for param in list(best_config):
        if param.endswith("__"):
            best_config.pop(param)

    return (
        f"Best {eval_metric.replace('_', ' ')}: {best_value} in {order[0]}\n"
        + pprint.pformat(best_config)
        .replace("{", " ")
        .replace("}", " ")
        .replace("'", "")
        .replace(",", "")
    )


ExperimentHandler = ExperimentManager  # to be backwards compatible
