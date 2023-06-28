# Class Hyperparameter *Metadata*

If you haven't seen the [argparse-related doc](./argparse.md), please check it out first.

Besides the `argparse` arguments for each hyperparameter, `argparse_args` also holds extra metadata related to the hyperparameter. These are useful to define hyperparameter importance, hierarchy, etc. Note that we consider a hyperparameter inactive by default when its value is `None` or `False`, or it has been deactivated (see how to do that below). The supported metadata fields so far are:

* `parent` (`Optional[str]`): whether the current hyperparameter is dependent on another hyperparameter. If so, and that other hyperparameter is inactive, then the current hyperparameter is also considered inactive.
* `parent_active` (`Optional[Union[Any, Callable]]`): define what the active value for the parent of the hyperparameter is, as opposed to not `None` or `False`. If a function, its input is expected to be the value of the parent, and it should return a bool.
* `disable_comparison` (`Optional[bool]`): whether to use the value of this hyperparameter to compare across experiments, e.g., the device. By default, the value is used for comparison.
* `name` (`Optional[Union[bool, Callable]]`): whether the value of the hyperparameter should appear in the experiments' subfolder name. If a function, input is the value of the hyperparameter, and output shoud be a bool. If subtree of hyperparameter (defined by `parent`) has been deactivated (i.e. value of ancestor is inactive, or its comparison has been disabled), then the name is not included in the subfolder name.
* `name_transform` (`Optional[Callable]`): if provided, value of hyperparameter is passed through the function before being put into the subfolder name, e.g., remove parent directory from filename.
* `name_priority` (`Optional[int]`): increasing priority of hyperparameter in the experiments' subfolder name. Default is 0.

Care should be taken so that a tree of hyperparameters is provided in order from root to leaves. To integrate these into the hyperparameters, simply define an extra field in the `argparse_args` dictionary for the hyperparameter called `"metadata"`, which itself contains a dictionary of the above values.