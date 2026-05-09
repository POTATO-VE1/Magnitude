"""Custom exceptions for the Magnitude Python client."""


class MagnitudeError(Exception):
    """Base exception for all Magnitude client errors."""


class MagnitudeConnectionError(MagnitudeError):
    """Failed to connect to the Magnitude server."""


class CollectionNotFoundError(MagnitudeError):
    """The requested collection does not exist."""


class AuthenticationError(MagnitudeError):
    """API key is invalid or missing."""


class ServerError(MagnitudeError):
    """The server returned an unexpected error."""

    def __init__(self, status_code: int, message: str):
        self.status_code = status_code
        super().__init__(f"Server error ({status_code}): {message}")
