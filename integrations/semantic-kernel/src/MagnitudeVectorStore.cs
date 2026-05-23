using Microsoft.Extensions.VectorData;
using Magnitude.Client;

namespace SemanticKernel.Magnitude;

public class MagnitudeVectorStore : VectorStore
{
    private readonly MagnitudeClient _client;
    private readonly string _tenantId;
    private readonly string _databaseId;

    public MagnitudeVectorStore(MagnitudeClient client, string tenantId = "default_tenant", string databaseId = "default")
    {
        _client = client;
        _tenantId = tenantId;
        _databaseId = databaseId;
    }

    public override VectorStoreCollection<TKey, TRecord> GetCollection<TKey, TRecord>(string name, VectorStoreRecordDefinition? definition = null)
    {
        return (VectorStoreCollection<TKey, TRecord>)(object)new MagnitudeCollection(_client, _tenantId, _databaseId, name, definition);
    }

    public override IAsyncEnumerable<string> ListCollectionNamesAsync(CancellationToken cancellationToken = default)
    {
        return AsyncEnumerable.Empty<string>();
    }
}

internal class MagnitudeCollection : VectorStoreCollection<string, VectorStoreRecord>
{
    private readonly MagnitudeClient _client;
    private readonly string _tenantId;
    private readonly string _databaseId;
    private readonly string _collectionName;
    private string? _collectionId;

    public MagnitudeCollection(MagnitudeClient client, string tenantId, string databaseId, string collectionName, VectorStoreRecordDefinition? definition)
    {
        _client = client;
        _tenantId = tenantId;
        _databaseId = databaseId;
        _collectionName = collectionName;
    }

    public override string Name => _collectionName;

    public override async Task<bool> CollectionExistsAsync(CancellationToken cancellationToken = default)
    {
        try { await GetCollectionIdAsync(); return true; }
        catch { return false; }
    }

    public override Task CreateCollectionAsync(CancellationToken cancellationToken = default) => Task.CompletedTask;

    public override Task CreateCollectionIfNotExistsAsync(CancellationToken cancellationToken = default) => Task.CompletedTask;

    public override Task DeleteCollectionAsync(CancellationToken cancellationToken = default) => Task.CompletedTask;

    public override async Task<string> UpsertAsync(VectorStoreRecord record, CancellationToken cancellationToken = default) => "";

    public override async Task<IReadOnlyList<string>> UpsertAsync(IEnumerable<VectorStoreRecord> records, CancellationToken cancellationToken = default) =>
        Array.Empty<string>();

    public override async Task<VectorStoreRecord?> GetAsync(string key, VectorStoreRecordDefinition? definition = null, CancellationToken cancellationToken = default) => null;

    public override async Task<IReadOnlyList<VectorStoreRecord?>> GetAsync(IEnumerable<string> keys, VectorStoreRecordDefinition? definition = null, CancellationToken cancellationToken = default) =>
        Array.Empty<VectorStoreRecord?>();

    public override async Task DeleteAsync(string key, CancellationToken cancellationToken = default) { }

    public override async Task DeleteAsync(IEnumerable<string> keys, CancellationToken cancellationToken = default) { }

    public override async IAsyncEnumerable<VectorStoreSearchResult<VectorStoreRecord>> SearchAsync(VectorSearchQuery vectorQuery, [System.Runtime.CompilerServices.EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        yield break;
    }

    public override IAsyncEnumerable<T> GetAsync<T>(VectorSearchQuery vectorQuery, CancellationToken cancellationToken = default) where T : class
    {
        return AsyncEnumerable.Empty<T>();
    }

    private async Task<string> GetCollectionIdAsync()
    {
        if (_collectionId != null) return _collectionId;
        var cols = await _client.ListCollectionsAsync(_tenantId, _databaseId);
        foreach (var c in cols)
            if (c.Name == _collectionName) { _collectionId = c.Id; return _collectionId; }
        throw new NotFoundException($"Collection {_collectionName} not found");
    }
}
