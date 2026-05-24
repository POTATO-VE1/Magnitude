using System.Net.Http.Json;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace Magnitude.Client;

public class MagnitudeException : Exception { public MagnitudeException(string msg) : base(msg) { } }
public class ConnectionException : MagnitudeException { public ConnectionException(string msg) : base(msg) { } }
public class AuthenticationException : MagnitudeException { public AuthenticationException(string msg) : base(msg) { } }
public class NotFoundException : MagnitudeException { public NotFoundException(string msg) : base(msg) { } }

public class Tenant { public string Id { get; set; } = ""; public string Name { get; set; } = ""; }
public class Database { public string Id { get; set; } = ""; public string TenantId { get; set; } = ""; public string Name { get; set; } = ""; }
public class Collection { public string Id { get; set; } = ""; public string Name { get; set; } = ""; public int Dimension { get; set; } public string Metric { get; set; } = ""; }
public class SearchResult { public ulong Id { get; set; } public float Distance { get; set; } public float Score { get; set; } public Dictionary<string, object>? Metadata { get; set; } }

public class MagnitudeClient : IDisposable
{
    private readonly HttpClient _http;
    private readonly string _baseUrl;
    private readonly string? _apiKey;

    public MagnitudeClient(string baseUrl, string? apiKey = null)
    {
        _baseUrl = baseUrl.TrimEnd('/');
        _apiKey = apiKey;
        _http = new HttpClient { Timeout = TimeSpan.FromSeconds(30) };
    }

    private HttpRequestMessage Req(string method, string path, object? body = null)
    {
        var msg = new HttpRequestMessage(new HttpMethod(method), _baseUrl + path);
        if (_apiKey != null) msg.Headers.Authorization = new("Bearer", _apiKey);
        if (body != null) msg.Content = JsonContent.Create(body);
        return msg;
    }

    private async Task<T> Send<T>(HttpRequestMessage msg)
    {
        var resp = await _http.SendAsync(msg);
        var json = await resp.Content.ReadAsStringAsync();
        var envelope = JsonSerializer.Deserialize<Envelope<T>>(json, new JsonSerializerOptions { PropertyNameCaseInsensitive = true });
        if (!resp.IsSuccessStatusCode)
        {
            var err = envelope?.Error ?? $"HTTP {(int)resp.StatusCode}";
            if (resp.StatusCode == System.Net.HttpStatusCode.Unauthorized) throw new AuthenticationException(err);
            if (resp.StatusCode == System.Net.HttpStatusCode.NotFound) throw new NotFoundException(err);
            throw new MagnitudeException(err);
        }
        if (envelope?.Data == null)
        {
            throw new MagnitudeException("Server returned empty data");
        }
        return envelope.Data;
    }

    public async Task<Tenant> CreateTenantAsync(string name) =>
        await Send<Tenant>(Req("POST", "/api/v2/tenants", new { name }));

    public async Task<List<Tenant>> ListTenantsAsync() =>
        await Send<List<Tenant>>(Req("GET", "/api/v2/tenants"));

    public async Task<Database> CreateDatabaseAsync(string tenantId, string name) =>
        await Send<Database>(Req("POST", $"/api/v2/tenants/{tenantId}/databases", new { name }));

    public async Task<Collection> CreateCollectionAsync(string tenantId, string dbId, string name, int dimension, string metric = "l2") =>
        await Send<Collection>(Req("POST", $"/api/v2/tenants/{tenantId}/databases/{dbId}/collections", new { name, dimension, metric }));

    public async Task<int> InsertAsync(string tenantId, string dbId, string colId, ulong[] ids, float[][] vectors, List<Dictionary<string, object>>? metadata = null)
    {
        var result = await Send<Dictionary<string, int>>(Req("POST", $"/api/v2/tenants/{tenantId}/databases/{dbId}/collections/{colId}/add", new { ids, vectors, metadata }));
        return result.GetValueOrDefault("inserted", 0);
    }

    public async Task<List<SearchResult>> SearchAsync(string tenantId, string dbId, string colId, float[] query, int topK = 10, Dictionary<string, object>? filter = null) =>
        await Send<List<SearchResult>>(Req("POST", $"/api/v2/tenants/{tenantId}/databases/{dbId}/collections/{colId}/query", new { query, k = topK, filter }));

    public async Task DeleteVectorsAsync(string tenantId, string dbId, string colId, ulong[] ids) =>
        await Send<object>(Req("POST", $"/api/v2/tenants/{tenantId}/databases/{dbId}/collections/{colId}/delete", new { ids }));

    public void Dispose() => _http.Dispose();
}

internal class Envelope<T> { public T? Data { get; set; } public string? Error { get; set; } }
