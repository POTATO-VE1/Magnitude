export { MagnitudeClient } from "./client.js";
export type {
  ClientConfig,
  Tenant,
  Database,
  Collection,
  CreateCollectionOptions,
  SearchOptions,
  SearchResult,
  InsertOptions,
} from "./types.js";
export {
  MagnitudeError,
  MagnitudeConnectionError,
  CollectionNotFoundError,
  AuthenticationError,
} from "./errors.js";
