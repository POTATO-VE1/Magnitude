import { MagnitudeClient } from "magnitude-js";

export interface MagnitudeStoreConfig {
  baseUrl: string;
  apiKey?: string;
  tenantId?: string;
  databaseId?: string;
  collectionId: string;
}

export function createMagnitudeStore(config: MagnitudeStoreConfig) {
  const client = new MagnitudeClient({
    baseUrl: config.baseUrl,
    apiKey: config.apiKey,
  });

  const tenantId = config.tenantId ?? "default_tenant";
  const databaseId = config.databaseId ?? "default";

  return {
    async upsert({
      ids,
      vectors,
      metadata,
    }: {
      ids: string[];
      vectors: number[][];
      metadata?: Record<string, unknown>[];
    }): Promise<void> {
      const numericIds = ids.map((id) => hashId(id));
      await client.insert(
        tenantId,
        databaseId,
        config.collectionId,
        numericIds,
        vectors,
        { metadata: metadata ?? [] },
      );
    },

    async query({
      vector,
      topK = 10,
      filter,
    }: {
      vector: number[];
      topK?: number;
      filter?: Record<string, unknown>;
    }): Promise<{
      ids: string[];
      scores: number[];
    }> {
      const results = await client.search(
        tenantId,
        databaseId,
        config.collectionId,
        vector,
        { topK, filter },
      );

      return {
        ids: results.map((r) => String(r.id)),
        scores: results.map((r) => r.score),
      };
    },

    async delete({ ids }: { ids: string[] }): Promise<void> {
      const numericIds = ids.map((id) => hashId(id));
      await client.deleteVectors(
        tenantId,
        databaseId,
        config.collectionId,
        numericIds,
      );
    },
  };
}

function hashId(id: string): number {
  let hash = 0;
  for (let i = 0; i < id.length; i++) {
    const char = id.charCodeAt(i);
    hash = (hash << 5) - hash + char;
    hash |= 0;
  }
  return Math.abs(hash);
}
