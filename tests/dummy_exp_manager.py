from argparse import Namespace

from legm import ExperimentManager

if __name__ == '__main__':
    exp_manager = ExperimentManager(
        "logs", "Test", "just_testing", logs_precision=3
    )

    params = Namespace(
        epochs=10,
        batch_size=32,
        learning_rate=0.001,
        optimizer="adam",
        loss="mse",
    )

    exp_manager.set_namespace_params(params)

    exp_manager.start()

    exp_manager.set_dict_metrics(
        {
            "accuracy": 0.9234567543,
            "loss": 0.13456787654,
            "val_accuracy": 0.898765434567,
            "val_loss": 0.29876543456789,
        }
    )
    exp_manager.set_dict_metrics(
        {
            "1": {"loss": [0.16574, 0.0997865435678, 0.086785436789]},
            "2": {"loss": [0.2897654346789, 0.19765433, 0.18123456]},
        }
    )

    exp_manager.set_dict_metrics(
        {
            "accuracy": 0.998765,
            "loss": 0.16754,
            "val_accuracy": 0.821345,
            "val_loss": 0.22134,
        }
    )
    exp_manager.set_dict_metrics(
        {
            "1": {"loss": [0.1132456, 0.12345, 0.08765]},
            "2": {"loss": [0.2132435, 0.145656, 0.1435655]},
        }
    )

    exp_manager.log_metrics()
    exp_manager.reset()

    exp_manager.set_dict_metrics(
        {
            "accuracy": 0.876545678,
            "loss": 0.1098765434567,
            "val_accuracy": 0.81234567,
            "val_loss": 0.21234567,
        }
    )
    exp_manager.set_dict_metrics(
        {
            "1": {"loss": [0.1234567, 0.09398765, 0.06590876]},
            "2": {"loss": [0.1434567, 0.15987654, 0.116754367854]},
        }
    )

    exp_manager.set_dict_metrics(
        {
            "accuracy": 0.8234567,
            "loss": 0.17654,
            "val_accuracy": 0.887654,
            "val_loss": 0.28765,
        }
    )
    exp_manager.set_dict_metrics(
        {
            "1": {"loss": [0.19876, 0.19876, 0.15343]},
            "2": {"loss": [0.1098765, 0.1345676, 0.113245]},
        }
    )

    exp_manager.log_metrics()
    exp_manager.aggregate_results()
