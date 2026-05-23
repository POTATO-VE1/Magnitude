package com.magnitude;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonProperty;
import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;

import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.time.Duration;
import java.util.List;
import java.util.Map;

public class MagnitudeClient {

    private final String baseUrl;
    private final HttpClient httpClient;
    private final ObjectMapper mapper;
    private final String apiKey;

    public MagnitudeClient(String baseUrl) {
        this(baseUrl, null);
    }

    public MagnitudeClient(String baseUrl, String apiKey) {
        this.baseUrl = baseUrl.replaceAll("/$", "");
        this.apiKey = apiKey;
        this.httpClient = HttpClient.newBuilder()
                .connectTimeout(Duration.ofSeconds(10))
                .build();
        this.mapper = new ObjectMapper();
    }

    // ── Data classes ──────────────────────────────────────────────────────

    @JsonIgnoreProperties(ignoreUnknown = true)
    public static class Tenant {
        public String id;
        public String name;
        @JsonProperty("max_databases")
        public int maxDatabases;
        @JsonProperty("max_collections")
        public int maxCollections;
        @JsonProperty("created_at")
        public long createdAt;
    }

    @JsonIgnoreProperties(ignoreUnknown = true)
    public static class Database {
        public String id;
        @JsonProperty("tenant_id")
        public String tenantId;
        public String name;
        @JsonProperty("created_at")
        public long createdAt;
    }

    @JsonIgnoreProperties(ignoreUnknown = true)
    public static class Collection {
        public String id;
        @JsonProperty("tenant_id")
        public String tenantId;
        @JsonProperty("database_id")
        public String databaseId;
        public String name;
        public int dimension;
        public String metric;
        @JsonProperty("index_type")
        public String indexType;
        @JsonProperty("created_at")
        public long createdAt;
        @JsonProperty("vector_count")
        public int vectorCount;
    }

    @JsonIgnoreProperties(ignoreUnknown = true)
    public static class SearchResult {
        public long id;
        public float distance;
        public float score;
        public Map<String, Object> metadata;
    }

    @JsonIgnoreProperties(ignoreUnknown = true)
    public static class Envelope<T> {
        public T data;
        public String error;
    }

    // ── HTTP helpers ──────────────────────────────────────────────────────

    private <T> T request(String method, String path, Object body, TypeReference<?> typeRef)
            throws MagnitudeException {
        try {
            String url = baseUrl + path;
            HttpRequest.Builder builder = HttpRequest.newBuilder()
                    .uri(URI.create(url))
                    .timeout(Duration.ofSeconds(30))
                    .header("Content-Type", "application/json");

            if (apiKey != null) {
                builder.header("Authorization", "Bearer " + apiKey);
            }

            switch (method) {
                case "GET":
                    builder.GET();
                    break;
                case "DELETE":
                    builder.DELETE();
                    break;
                default:
                    String json = body != null ? mapper.writeValueAsString(body) : "{}";
                    builder.method(method, HttpRequest.BodyPublishers.ofString(json));
                    break;
            }

            HttpResponse<String> resp = httpClient.send(builder.build(),
                    HttpResponse.BodyHandlers.ofString());

            Envelope<?> envelope = mapper.readValue(resp.body(),
                    new TypeReference<Envelope<Object>>() {});

            if (resp.statusCode() >= 400) {
                String msg = envelope.error != null ? envelope.error : "HTTP " + resp.statusCode();
                if (resp.statusCode() == 401) throw new AuthenticationException(msg);
                if (resp.statusCode() == 404) throw new NotFoundException(msg);
                throw new MagnitudeException(msg);
            }

            if (envelope.data == null) return null;
            return mapper.convertValue(envelope.data, typeRef);
        } catch (MagnitudeException e) {
            throw e;
        } catch (Exception e) {
            throw new ConnectionException("Request failed: " + e.getMessage(), e);
        }
    }

    // ── Tenants ───────────────────────────────────────────────────────────

    public Tenant createTenant(String name, int maxDatabases, int maxCollections)
            throws MagnitudeException {
        return request("POST", "/api/v2/tenants",
                Map.of("name", name, "max_databases", maxDatabases, "max_collections", maxCollections),
                new TypeReference<Envelope<Tenant>>() {}).data;
    }

    public List<Tenant> listTenants() throws MagnitudeException {
        return request("GET", "/api/v2/tenants", null,
                new TypeReference<Envelope<List<Tenant>>>() {}).data;
    }

    public void deleteTenant(String id) throws MagnitudeException {
        request("DELETE", "/api/v2/tenants/" + id, null,
                new TypeReference<Envelope<Object>>() {});
    }

    // ── Databases ─────────────────────────────────────────────────────────

    public Database createDatabase(String tenantId, String name) throws MagnitudeException {
        return request("POST", "/api/v2/tenants/" + tenantId + "/databases",
                Map.of("name", name),
                new TypeReference<Envelope<Database>>() {}).data;
    }

    public void deleteDatabase(String tenantId, String databaseId) throws MagnitudeException {
        request("DELETE", "/api/v2/tenants/" + tenantId + "/databases/" + databaseId, null,
                new TypeReference<Envelope<Object>>() {});
    }

    // ── Collections ───────────────────────────────────────────────────────

    public Collection createCollection(String tenantId, String databaseId,
            String name, int dimension, String metric) throws MagnitudeException {
        return request("POST",
                "/api/v2/tenants/" + tenantId + "/databases/" + databaseId + "/collections",
                Map.of("name", name, "dimension", dimension, "metric", metric),
                new TypeReference<Envelope<Collection>>() {}).data;
    }

    public List<Collection> listCollections(String tenantId, String databaseId)
            throws MagnitudeException {
        return request("GET",
                "/api/v2/tenants/" + tenantId + "/databases/" + databaseId + "/collections",
                null, new TypeReference<Envelope<List<Collection>>>() {}).data;
    }

    public void deleteCollection(String tenantId, String databaseId, String collectionId)
            throws MagnitudeException {
        request("DELETE",
                "/api/v2/tenants/" + tenantId + "/databases/" + databaseId
                        + "/collections/" + collectionId,
                null, new TypeReference<Envelope<Object>>() {});
    }

    // ── Vectors ───────────────────────────────────────────────────────────

    public int insert(String tenantId, String databaseId, String collectionId,
            long[] ids, float[][] vectors, List<Map<String, Object>> metadata)
            throws MagnitudeException {
        String path = "/api/v2/tenants/" + tenantId + "/databases/" + databaseId
                + "/collections/" + collectionId + "/add";
        Map<String, Object> body = Map.of("ids", ids, "vectors", vectors,
                "metadata", metadata != null ? metadata : List.of());
        Map<String, Integer> result = request("POST", path, body,
                new TypeReference<Envelope<Map<String, Integer>>>() {}).data;
        return result != null ? result.getOrDefault("inserted", 0) : 0;
    }

    public List<SearchResult> search(String tenantId, String databaseId,
            String collectionId, float[] query, int topK, Map<String, Object> filter)
            throws MagnitudeException {
        String path = "/api/v2/tenants/" + tenantId + "/databases/" + databaseId
                + "/collections/" + collectionId + "/query";
        Map<String, Object> body = new java.util.HashMap<>();
        body.put("query", query);
        body.put("k", topK);
        if (filter != null) body.put("filter", filter);
        return request("POST", path, body,
                new TypeReference<Envelope<List<SearchResult>>>() {}).data;
    }

    public void deleteVectors(String tenantId, String databaseId,
            String collectionId, long[] ids) throws MagnitudeException {
        String path = "/api/v2/tenants/" + tenantId + "/databases/" + databaseId
                + "/collections/" + collectionId + "/delete";
        request("POST", path, Map.of("ids", ids),
                new TypeReference<Envelope<Object>>() {});
    }
}
