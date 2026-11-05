"""Exceptions raised by the data layer."""


class DataIntegrityError(Exception):
    """The dataset is missing, malformed, or fails a blocking integrity check."""
