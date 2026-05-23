import type {
  ClientConfig,
  Tenant,
  Database,
  Collection,
  CreateCollectionOptions,
  SearchOptions,
  SearchResult,
  InsertOptions,
  Envelope,
} from "./types.js";
import {
  MagnitudeError,
  MagnitudeConnectionError,
  CollectionNotFoundError,
  AuthenticationError,
} from "./errors.js";

export class MagnitudeClient {
  private baseUrl: string;
  private apiKey?: string;
  private timeout: number;

  constructor(config: ClientConfig | string) {
    if (typeof config === "string") {
      this.baseUrl = config.replace(/\/$/, "");
    } else {
      this.baseUrl = config.baseUrl.replace(/\/$/, "");
      this.apiKey = config.apiKey;
      this.timeout = config.timeout ?? 30000;
    }
  }

  // ── Private helpers ──────────────────────────────────────────────────────

  private headers(): Record<string, string> {
    const h: Record<string, string> = {
      "Content-Type": "application/json",
    };
    if (this.apiKey) {
      h["Authorization"] = `Bearer ${this.apiKey}`;
    }
    return h;
  }

  private async request<T>(
    method: string,
    path: string,
    body?: unknown,
  ): Promise<T> {
    const url = `${this.baseUrl}${path}`;
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), this.timeout);

    try {
      const resp = await fetch(url, {
        method,
        headers: this.headers(),
        body: body ? JSON.stringify(body) : undefined,
        signal: controller.signal,
      });

      const json: Envelope<T> = await resp.json();

      if (!resp.ok) {
        const msg = json.error ?? `HTTP ${resp.status}`;
        if (resp.status === 401) throw new AuthenticationError(msg);
        if (resp.status === 404) throw new CollectionNotFoundError(msg);
        throw new MagnitudeError(msg);
      }

      return json.data as T;
    } catch (err) {
      if (err instanceof MagnitudeError) throw err;
      if (err instanceof DOMException && err.name === "AbortError") {
        throw new MagnitudeConnectionError("Request timed out");
      }
      throw new MagnitudeConnectionError(
        `Connection failed: ${(err as Error).message}`,
      );
    } finally {
      clearTimeout(timeoutId);
    }
  }

  // ── Health ───────────────────────────────────────────────────────────────

  async healthCheck(): Promise<{ status: string }> {
    return this.request("GET", "/v1/health");
  }

  // ── Tenants ──────────────────────────────────────────────────────────────

  async createTenant(
    name: string,
    options?: { maxDatabases?: number; maxCollections?: number },
  ): Promise<Tenant> {
    return this.request("POST", "/api/v2/tenants", {
      name,
      max_databases: options?.maxDatabases ?? 0,
      max_collections: options?.maxCollections ?? 0,
    });
  }

  async listTenants(): Promise<Tenant[]> {
    return this.request("GET", "/api/v2/tenants");
  }

  async getTenant(id: string): Promise<Tenant> {
    return this.request("GET", `/api/v2/tenants/${id}`);
  }

  async deleteTenant(id: string): Promise<void> {
    await this.request("DELETE", `/api/v2/tenants/${id}`);
  }

  // ── Databases ────────────────────────────────────────────────────────────

  async createDatabase(tenantId: string, name: string): Promise<Database> {
    return this.request(
      "POST",
      `/api/v2/tenants/${tenantId}/databases`,
      { name },
    );
  }

  async listDatabases(tenantId: string): Promise<Database[]> {
    return this.request("GET", `/api/v2/tenants/${tenantId}/databases`);
  }

  async deleteDatabase(tenantId: string, databaseId: string): Promise<void> {
    await this.request(
      "DELETE",
      `/api/v2/tenants/${tenantId}/databases/${databaseId}`,
    );
  }

  // ── Collections ──────────────────────────────────────────────────────────

  async createCollection(
    tenantId: string,
    databaseId: string,
    name: string,
    options: CreateCollectionOptions,
  ): Promise<Collection> {
    return this.request(
      "POST",
      `/api/v2/tenants/${tenantId}/databases/${databaseId}/collections`,
      {
        name,
        dimension: options.dimension,
        metric: options.metric ?? "l2",
        index_type: options.indexType ?? "hnsw",
      },
    );
  }

  async listCollections(
    tenantId: string,
    databaseId: string,
  ): Promise<Collection[]> {
    return this.request(
      "GET",
      `/api/v2/tenants/${tenantId}/databases/${databaseId}/collections`,
    );
  }

  async getCollection(
    tenantId: string,
    databaseId: string,
    collectionId: string,
  ): Promise<Collection> {
    return this.request(
      "GET",
      `/api/v2/tenants/${tenantId}/databases/${databaseId}/collections/${collectionId}`,
    );
  }

  async deleteCollection(
    tenantId: string,
    databaseId: string,
    collectionId: string,
  ): Promise<void> {
    await this.request(
      "DELETE",
      `/api/v2/tenants/${tenantId}/databases/${databaseId}/collections/${collectionId}`,
    );
  }

  // ── Vectors ──────────────────────────────────────────────────────────────

  async insert(
    tenantId: string,
    databaseId: string,
    collectionId: string,
    ids: number[],
    vectors: number[][],
    options?: InsertOptions,
  ): Promise<{ inserted: number }> {
    return this.request(
      "POST",
      `/api/v2/tenants/${tenantId}/databases/${databaseId}/collections/${collectionId}/add`,
      { ids, vectors, metadata: options?.metadata },
    );
  }

  async search(
    tenantId: string,
    databaseId: string,
    collectionId: string,
    query: number[],
    options?: SearchOptions,
  ): Promise<SearchResult[]> {
    return this.request(
      "POST",
      `/api/v2/tenants/${tenantId}/databases/${databaseId}/collections/${collectionId}/query`,
      {
        query,
        k: options?.topK ?? 10,
        nprobe: options?.nprobe ?? 0,
        filter: options?.filter,
      },
    );
  }

  async deleteVectors(
    tenantId: string,
    databaseId: string,
    collectionId: string,
    ids: number[],
  ): Promise<void> {
    await this.request(
      "POST",
      `/api/v2/tenants/${tenantId}/databases/${databaseId}/collections/${collectionId}/delete`,
      { ids },
    );
  }

  async hybridSearch(
    tenantId: string,
    databaseId: string,
    collectionId: string,
    query: number[],
    queryText: string,
    options?: SearchOptions,
  ): Promise<SearchResult[]> {
    return this.request(
      "POST",
      `/api/v2/tenants/${tenantId}/databases/${databaseId}/collections/${collectionId}/hybrid`,
      {
        query,
        query_text: queryText,
        k: options?.topK ?? 10,
        filter: options?.filter,
      },
    );
  }
}
