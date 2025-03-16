# Your experiment tracker and manager

Overall, the `legm.ExperimentManager`, at its current state, is supposed to hold all hyperparameters and hence define the experiment (and be able to differentiate between different experiments), log results and aggregate across different runs for these experiments (in yaml files), and plot the results for easier gauging. Moreover, it logs the data in `Tensorboard`, which enables comparison between different experiments. We use a general folder to hold experiments of the same kind, which within it contains all the different hyperparameter configurations. `Tensoboard` can be launched at the level of the general folder, and therefore can be used to compare across experiments.

## Initialization

To initialize, provide the general experiments folder (the `.gitignore` here ignores `logs` for example), then the specific kind of experiment being run (for example, "LeNet@MNIST"). You can also provide a description for the experiment. This will differentiate the experiment from experiments with the same hyperparameters, which enables you, for instance, to make temporary changes in the code and have those reflected as different experiments. The name of the specific experiment folder is derived from the parameters designated for the name, which can be done with `.name_param(param)`, or with the `name` property of metadata (See [here](./metadata.md)). You can also specify a name for the experiment subfolder with `alternative_experiment_name`, which can be a template to be filled in with the values of the provided parameters.

## Registering Hyperparameters

There are various methods to register hyperparameters. The simplest one is `.set_param(hyperparam_name: str, hyperparem_value: Any, hyperparam_parent: str)`, where the `hyperparam_parent` denotes which other hyperparameter `hyperparam_name` is dependent on (from the registered hyperparameters). See [this](./metadata.md) to understand the utility of setting parents. To use a dictionary or namespace instead of setting individual hyperparameters, use `set_dict_params` and `set_namespace_params`. Note you can set only one parent for the whole structure. The namespace-specific method is useful for `argparse` arguments, among others.

## Registering Metadata of Hyperparameters

More information is required for some hyperparameters. For such uses, like denoting which parameters should not be counted in the comparison between different hyperparemeter configurations (like the number of workers in the dataloader), or which parameters should go on the name of the experiment for easier navigation, see again [this](./metadata.md). To parse such a metadata structure, we use `set_param_metadata(metadata: Dict[str, Dict[str, Any]], param_namespace)`. Otherwise, you can use utility methods like `disable_param`, `name_param`, etc.

## Start using

After registering hyperparameters, they are then accessible as simple attributes, e.g.

```python
exp_manager = ExperimentManager(this_path, that_path)
exp_manager.set_param("learning_rate", 1e-2)
exp_manager.learning_rate
```

To begin logging, we have to use `.start()`. During training, you can register metrics with similar mechanisms to hyperparameters. The simplest method is `set_metric(metric_name: str, metric_value: float, test_or_dev: bool, step: int)`. However, we recommend `set_dict_metrics(metric_dict: dict[str, float], test_or_dev: bool, step: int)`. Additionally, if your metrics are further indexed by some other key (e.g., if you want to log metrics for each example separately), you can provide the metrics within another dictionary, whose keys are the additional index: `set_dict_metrics(indexed_metric_dict: dict[str, dict[str, float]], ...)`. At the end of training, you can use `.set_best()` to denote which step was the best (e.g., derived from early stopping). Otherwise, the last step is picked.

## Log

To log all those results, at the end of training, use `.log_metrics()`, which creates `metrics.yml` that contains all metrics for all steps (if any non-indexed metrics have been logged), `indexed_metrics.yml` for indexed metrics, `.aggregate_results()`, which aggregates the best and test metrics and calculates mean and standard deviation from the `metrics.yml` file (because more runs may be logged in there) into `aggregated_metrics.yml`, and the equivalent in `aggregated_indexed_metrics.yml`, and `.plot()` (that contains optional arguments) to plot the metrics in `plots/`.

## Folder Structure

To create a more intricate folder structure, and control the name of each experiment more precisely, you can use `alternative_experiment_name`, which can be template / formattable string. You can include the name of variables by enclosing them in curly brackets (`{}`), for example: `this-experiment-var1={variable_1}-var2={variable_2}`. Moreover, you can use a path as an alternative name: `path/to/subfolder/this-experiment-var1={variable_1}-var2={variable_2}`, which will results in subfolders in within the folder you are already logging in.
