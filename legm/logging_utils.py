import logging
import warnings
from typing import Optional, Union


class LoggingMixin:
    """Mixin for logging.

    Initializes a logger if `logging_file` is provided
    and provides a `log` method.

    Attributes:
        _logging_mixin_data: dict for the logging mixin.
    """

    @staticmethod
    def argparse_args():
        return dict(
            logging_level=dict(
                type=str,
                default="info",
                choices=["info", "debug", "warning", "error"],
                help="logging level",
                metadata=dict(disable_comparison=True),
            )
        )

    def __init__(
        self,
        logger: Optional[logging.Logger] = None,
        logging_file: Optional[str] = None,
        logging_level: Optional[Union[int, str]] = None,
        logger_name: str = None,
        main_process: bool = True,
        *args,
        **kwargs,
    ):
        """Initializes the logging mixin.

        Args:
            logger: logger to use. Priority over `logging_file`.
            logging_file: logging file.
            logging_level: logging level.
            logger_name: name of the logger.
        """
        self._logging_mixin_data = {"main_process": main_process}

        if logger is not None:
            self._logging_mixin_data["logger"] = logger
        elif logging_file is not None:
            self.create_logger(logging_file, logging_level, logger_name)

        super().__init__(*args, **kwargs)

    def set_main_process(self, main_process: bool):
        """Sets the main process.

        Args:
            main_process: whether the process is the main process.
        """
        self._logging_mixin_data["main_process"] = main_process

    def is_main_process(self) -> bool:
        """Returns whether the process is the main process."""
        return self._logging_mixin_data["main_process"]

    def create_logger(
        self,
        logging_file,
        logging_level: Optional[Union[int, str]] = None,
        name: Optional[str] = None,
    ):
        """Creates a logger.

        Args:
            logging_file: logging file.
            logging_level: logging level.
            name: name of the logger.
        """

        if name is None:
            name = self.__class__.__name__

        logger = logging.getLogger(name)

        for handler in logger.handlers:
            logger.removeHandler(handler)

        if logging_level is None:
            logging_level = logging.WARNING
        elif isinstance(logging_level, str):
            logging_level = getattr(logging, logging_level.upper())
        logger.setLevel(logging_level)

        fh = logging.FileHandler(logging_file)
        fh.setLevel(logging_level)
        formatter = logging.Formatter(
            "%(levelname)s-%(name)s(%(asctime)s)   %(message)s"
        )
        fh.setFormatter(formatter)

        logger.addHandler(fh)

        self._logging_mixin_data["logger"] = logger

    def get_logger(self) -> Optional[logging.Logger]:
        """Returns the logger."""
        return self._logging_mixin_data.get("logger", None)

    def log(self, message: str, level: Optional[Union[int, str]] = None):
        """Logs a message.

        Args:
            message: message to log.
            level: logging level.
        """

        if not self._logging_mixin_data["main_process"]:
            return

        if "logger" not in self._logging_mixin_data:
            warnings.warn("No logger has been initialized.")
        else:
            if level is None:
                level = logging.WARNING
            elif isinstance(level, str):
                level = getattr(logging, level.upper())

            self._logging_mixin_data["logger"].log(level, message)
