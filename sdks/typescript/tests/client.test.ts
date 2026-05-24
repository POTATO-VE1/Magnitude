import { describe, it, expect, beforeAll, afterAll } from "vitest";
import { MagnitudeClient, MagnitudeError, AuthenticationError } from "../src/index.js";

const BASE_URL = process.env.MAGNITUDE_URL || "http://localhost:8080";

describe("MagnitudeClient", () => {
  let client: MagnitudeClient;
  let tenantId: string;
  let databaseId: string;
  let collectionId: string;

  beforeAll(async () => {
    client = new MagnitudeClient({ baseUrl: BASE_URL });

    // Create tenant
    const tenant = await client.createTenant("test-tenant");
    tenantId = tenant.id;

    // Create database
    const db = await client.createDatabase(tenantId, "test-db");
    databaseId = db.id;

    // Create collection
    const col = await client.createCollection(tenantId, databaseId, "test-col", {
      dimension: 4,
      metric: "l2",
    });
    collectionId = col.id;
  });

  afterAll(async () => {
    // Cleanup
    try {
      await client.deleteCollection(tenantId, databaseId, collectionId);
      await client.deleteDatabase(tenantId, databaseId);
      await client.deleteTenant(tenantId);
    } catch {
      // ignore cleanup errors
    }
  });

  describe("healthCheck", () => {
    it("returns status ok", async () => {
      const result = await client.healthCheck();
      expect(result.status).toBe("ok");
    });
  });

  describe("tenants", () => {
    it("lists tenants", async () => {
      const tenants = await client.listTenants();
      expect(tenants.length).toBeGreaterThan(0);
      expect(tenants.some((t) => t.id === tenantId)).toBe(true);
    });

    it("gets tenant by id", async () => {
      const tenant = await client.getTenant(tenantId);
      expect(tenant.name).toBe("test-tenant");
    });
  });

  describe("databases", () => {
    it("lists databases", async () => {
      const dbs = await client.listDatabases(tenantId);
      expect(dbs.length).toBeGreaterThan(0);
      expect(dbs.some((d) => d.id === databaseId)).toBe(true);
    });
  });

  describe("collections", () => {
    it("lists collections", async () => {
      const cols = await client.listCollections(tenantId, databaseId);
      expect(cols.length).toBeGreaterThan(0);
      expect(cols.some((c) => c.id === collectionId)).toBe(true);
    });

    it("gets collection by id", async () => {
      const col = await client.getCollection(tenantId, databaseId, collectionId);
      expect(col.name).toBe("test-col");
      expect(col.dimension).toBe(4);
    });
  });

  describe("vectors", () => {
    it("inserts vectors", async () => {
      const result = await client.insert(tenantId, databaseId, collectionId, [1, 2], [[1, 0, 0, 0], [0, 1, 0, 0]], {
        metadata: [{ label: "a" }, { label: "b" }],
      });
      expect(result.inserted).toBe(2);
    });

    it("searches vectors", async () => {
      const results = await client.search(tenantId, databaseId, collectionId, [1, 0, 0, 0], { topK: 2 });
      expect(results.length).toBeGreaterThan(0);
      expect(results[0].id).toBe(1);
    });

    it("searches with filter", async () => {
      const results = await client.search(tenantId, databaseId, collectionId, [1, 0, 0, 0], {
        topK: 10,
        filter: { label: "b" },
      });
      expect(results.length).toBeGreaterThan(0);
    });

    it("deletes vectors", async () => {
      await client.deleteVectors(tenantId, databaseId, collectionId, [1]);
      const results = await client.search(tenantId, databaseId, collectionId, [1, 0, 0, 0], { topK: 10 });
      expect(results.every((r) => r.id !== 1)).toBe(true);
    });
  });

  describe("errors", () => {
    it("throws on invalid url", async () => {
      const badClient = new MagnitudeClient({ baseUrl: "http://localhost:99999", timeout: 1000 });
      await expect(badClient.healthCheck()).rejects.toThrow();
    });

    it("throws on not found", async () => {
      await expect(client.getTenant("nonexistent")).rejects.toThrow();
    });
  });
});
