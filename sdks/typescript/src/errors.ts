export class MagnitudeError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "MagnitudeError";
  }
}

export class MagnitudeConnectionError extends MagnitudeError {
  constructor(message: string) {
    super(message);
    this.name = "MagnitudeConnectionError";
  }
}

export class CollectionNotFoundError extends MagnitudeError {
  constructor(message: string) {
    super(message);
    this.name = "CollectionNotFoundError";
  }
}

export class AuthenticationError extends MagnitudeError {
  constructor(message: string) {
    super(message);
    this.name = "AuthenticationError";
  }
}
