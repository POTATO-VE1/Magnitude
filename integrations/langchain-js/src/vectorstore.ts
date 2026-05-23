import { VectorStore } from "@langchain/core/vectorstores";
import { Document } from "@langchain/core/documents";
import { Embeddings } from "@langchain/core/embeddings";
import { MagnitudeClient } from "magnitude-js";

export interface MagnitudeVectorStoreConfig {
  client: MagnitudeClient;
  collectionName: string;
  embeddings: Embeddings;
  tenantId?: string;
  databaseId?: string;
  collectionId?: string;
}

export class MagnitudeVectorStore extends VectorStore {
  private client: MagnitudeClient;
  private collectionName: string;
  private tenantId: string;
  private databaseId: string;
  private _collectionId: string | undefined;

  _vectorstoreType(): string {
    return "magnitude";
  }

  constructor(config: MagnitudeVectorStoreConfig) {
    super(config.embeddings, {});
    this.client = config.client;
    this.collectionName = config.collectionName;
    this.tenantId = config.tenantId ?? "default_tenant";
    this.databaseId = config.databaseId ?? "default";
    this._collectionId = config.collectionId;
  }

  private async getCollectionId(): Promise<string> {
    if (!this._collectionId) {
      const cols = await this.client.listCollections(
        this.tenantId,
        this.databaseId,
      );
      const existing = cols.find((c) => c.name === this.collectionName);
      if (existing) {
        this._collectionId = existing.id;
      } else {
        const dim = (await this.embeddings.embedQuery("test")).length;
        const col = await this.client.createCollection(
          this.tenantId,
          this.databaseId,
          this.collectionName,
          { dimension: dim, metric: "cosine" },
        );
        this._collectionId = col.id;
      }
    }
    return this._collectionId;
  }

  async addVectors(
    vectors: number[][],
    documents: Document[],
    options?: { ids?: string[] },
  ): Promise<void> {
    const collectionId = await this.getCollectionId();
    const ids =
      options?.ids?.map((id) => this.hashId(id)) ??
      documents.map(() => Math.floor(Math.random() * Number.MAX_SAFE_INTEGER));

    const metadata = documents.map((doc) => ({
      ...doc.metadata,
      text: doc.pageContent,
    }));

    await this.client.insert(
      this.tenantId,
      this.databaseId,
      collectionId,
      ids,
      vectors,
      { metadata },
    );
  }

  async addDocuments(
    documents: Document[],
    options?: { ids?: string[] },
  ): Promise<void> {
    const texts = documents.map((doc) => doc.pageContent);
    const vectors = await this.embeddings.embedDocuments(texts);
    await this.addVectors(vectors, documents, options);
  }

  async similaritySearchVectorWithScore(
    query: number[],
    k: number,
    filter?: Record<string, unknown>,
  ): Promise<[Document, number][]> {
    const collectionId = await this.getCollectionId();

    const results = await this.client.search(
      this.tenantId,
      this.databaseId,
      collectionId,
      query,
      { topK: k, filter },
    );

    return results.map((r) => {
      const meta = { ...(r.metadata ?? {}) };
      const text = (meta.text as string) ?? "";
      delete meta.text;
      return [new Document({ pageContent: text, metadata: meta }), r.distance];
    });
  }

  async delete(options: { ids?: string[] }): Promise<void> {
    if (options.ids) {
      const collectionId = await this.getCollectionId();
      await this.client.deleteVectors(
        this.tenantId,
        this.databaseId,
        collectionId,
        options.ids.map((id) => this.hashId(id)),
      );
    }
  }

  static async fromTexts(
    texts: string[],
    metadatas: Record<string, unknown>[],
    embeddings: Embeddings,
    config: Omit<MagnitudeVectorStoreConfig, "embeddings">,
  ): Promise<MagnitudeVectorStore> {
    const store = new MagnitudeVectorStore({ ...config, embeddings });
    const docs = texts.map(
      (text, i) =>
        new Document({
          pageContent: text,
          metadata: metadatas[i] ?? {},
        }),
    );
    await store.addDocuments(docs);
    return store;
  }

  private hashId(id: string): number {
    let hash = 0;
    for (let i = 0; i < id.length; i++) {
      const char = id.charCodeAt(i);
      hash = (hash << 5) - hash + char;
      hash |= 0;
    }
    return Math.abs(hash);
  }
}
