import logging


logging_level_list = ["WARNING", "INFO", "DEBUG"]


def setup_logging(verbose: int = 0) -> None:
    """
    Setup carotid's logging facilities.
    Parameters
    ----------
    verbose: bool
        The desired level of verbosity for logging.
        - 0: WARNING
        - 1: INFO
        - 2: DEBUG
    """
    if verbose > 2:
        verbose = 2

    logging_level = logging_level_list[verbose]

    # Define the module level logger.
    logger = logging.getLogger("carotid")
    logger.setLevel(logging_level)
