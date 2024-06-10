from argparse import Namespace

from legm import ExperimentManager

if __name__ == '__main__':
    exp_manager = ExperimentManager("logs", "Test", "just_testing", logs_precision=3)

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

    exp_manager.log_metrics()
    exp_manager.aggregate_results()


