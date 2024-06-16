from argparse import Namespace
from legm import ExperimentManager

e = ExperimentManager("logs", "debug")
params = Namespace(lr=1e-3, bs=16, f=True)
e.set_namespace_params(params)
metadata = dict(lr=dict(name=True))
e.set_param_metadata(metadata, params)
e.start()
metrics = {"1": {"a": 0.81, "b": 0.81}, "2": {"a": 0.57, "c": 0.49}}
e.set_custom_data(metrics, "lalala.yml")
e.log_metrics()
e.aggregate_results()
