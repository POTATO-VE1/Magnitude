package com.magnitude;

public class MagnitudeException extends Exception {
    public MagnitudeException(String message) {
        super(message);
    }

    public MagnitudeException(String message, Throwable cause) {
        super(message, cause);
    }
}

class ConnectionException extends MagnitudeException {
    public ConnectionException(String message, Throwable cause) {
        super(message, cause);
    }
}

class AuthenticationException extends MagnitudeException {
    public AuthenticationException(String message) {
        super(message);
    }
}

class NotFoundException extends MagnitudeException {
    public NotFoundException(String message) {
        super(message);
    }
}
