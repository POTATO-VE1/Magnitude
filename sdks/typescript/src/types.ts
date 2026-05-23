export interface ClientConfig {
  baseUrl: string;
  apiKey?: string;
  timeout?: number;
}

export interface Tenant {
  id: string;
  name: string;
  max_databases: number;
  max_collections: number;
  created_at: number;
}

export interface Database {
  id: string;
  tenant_id: string;
  name: string;
  created_at: number;
}

export interface Collection {
  id: string;
  tenant_id: string;
  database_id: string;
  name: string;
  dimension: number;
  metric: string;
  index_type: string;
  created_at: number;
  vector_count: number;
}

export interface CreateCollectionOptions {
  dimension: number;
  metric?: "l2" | "cosine" | "dot" | "manhattan";
  indexType?: "flat" | "ivf" | "hnsw" | "spann";
}

export interface SearchOptions {
  topK?: number;
  nprobe?: number;
  filter?: Record<string, unknown>;
}

export interface SearchResult {
  id: number;
  distance: number;
  score: number;
  metadata?: Record<string, unknown>;
}

export interface InsertOptions {
  metadata?: Record<string, unknown>[];
}

export interface Envelope<T = unknown> {
  data?: T;
  error?: string;
}
