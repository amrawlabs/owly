"""Owly exception hierarchy."""


class OwlyError(Exception):
    """Base exception for all Owly runtime failures."""


class ProviderError(OwlyError):
    """Raised on provider boundary failures."""


class CancellationError(OwlyError):
    """Raised when stream execution is cancelled."""


class ConfigurationError(OwlyError):
    """Raised when runtime/provider configuration is invalid."""
