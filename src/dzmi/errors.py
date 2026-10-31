"""Project-specific exceptions with actionable messages."""


class DZMIError(Exception):
    """Base class for all expected project errors."""


class ConfigError(DZMIError):
    """Invalid or missing configuration."""


class DatabaseError(DZMIError):
    """Database connection, DDL or load problem."""


class DataQualityError(DZMIError):
    """One or more blocking data-quality checks failed."""
