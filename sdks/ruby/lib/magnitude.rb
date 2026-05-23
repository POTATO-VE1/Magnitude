require "net/http"
require "json"
require "uri"

module Magnitude
  class Error < StandardError; end
  class ConnectionError < Error; end
  class AuthenticationError < Error; end
  class NotFoundError < Error; end

  class Client
    def initialize(base_url, api_key: nil)
      @base_url = base_url.chomp("/")
      @api_key = api_key
    end

    # ── Tenants ────────────────────────────────────────────────────────

    def create_tenant(name, max_databases: 0, max_collections: 0)
      request(:post, "/api/v2/tenants", {
        name: name,
        max_databases: max_databases,
        max_collections: max_collections,
      })
    end

    def list_tenants
      request(:get, "/api/v2/tenants")
    end

    def delete_tenant(id)
      request(:delete, "/api/v2/tenants/#{id}")
    end

    # ── Databases ──────────────────────────────────────────────────────

    def create_database(tenant_id, name)
      request(:post, "/api/v2/tenants/#{tenant_id}/databases", { name: name })
    end

    def delete_database(tenant_id, database_id)
      request(:delete, "/api/v2/tenants/#{tenant_id}/databases/#{database_id}")
    end

    # ── Collections ────────────────────────────────────────────────────

    def create_collection(tenant_id, database_id, name, dimension:, metric: "l2")
      request(:post, "/api/v2/tenants/#{tenant_id}/databases/#{database_id}/collections", {
        name: name,
        dimension: dimension,
        metric: metric,
      })
    end

    def list_collections(tenant_id, database_id)
      request(:get, "/api/v2/tenants/#{tenant_id}/databases/#{database_id}/collections")
    end

    def delete_collection(tenant_id, database_id, collection_id)
      request(:delete, "/api/v2/tenants/#{tenant_id}/databases/#{database_id}/collections/#{collection_id}")
    end

    # ── Vectors ────────────────────────────────────────────────────────

    def insert(tenant_id, database_id, collection_id, ids:, vectors:, metadata: nil)
      body = { ids: ids, vectors: vectors }
      body[:metadata] = metadata if metadata
      result = request(:post,
        "/api/v2/tenants/#{tenant_id}/databases/#{database_id}/collections/#{collection_id}/add",
        body)
      result["inserted"] || 0
    end

    def search(tenant_id, database_id, collection_id, query:, top_k: 10, filter: nil)
      body = { query: query, k: top_k }
      body[:filter] = filter if filter
      request(:post,
        "/api/v2/tenants/#{tenant_id}/databases/#{database_id}/collections/#{collection_id}/query",
        body)
    end

    def delete_vectors(tenant_id, database_id, collection_id, ids:)
      request(:post,
        "/api/v2/tenants/#{tenant_id}/databases/#{database_id}/collections/#{collection_id}/delete",
        { ids: ids })
    end

    private

    def request(method, path, body = nil)
      uri = URI.parse("#{@base_url}#{path}")
      http = Net::HTTP.new(uri.host, uri.port)
      http.use_ssl = uri.scheme == "https"
      http.read_timeout = 30

      case method
      when :get
        req = Net::HTTP::Get.new(uri.path)
      when :post
        req = Net::HTTP::Post.new(uri.path)
        req.body = JSON.generate(body || {})
      when :delete
        req = Net::HTTP::Delete.new(uri.path)
      end

      req["Content-Type"] = "application/json"
      req["Authorization"] = "Bearer #{@api_key}" if @api_key

      resp = http.request(req)
      json = JSON.parse(resp.body)

      unless resp.is_a?(Net::HTTPSuccess)
        msg = json["error"] || "HTTP #{resp.code}"
        raise AuthenticationError, msg if resp.code == "401"
        raise NotFoundError, msg if resp.code == "404"
        raise Error, msg
      end

      json["data"]
    rescue Net::OpenTimeout, Net::ReadTimeout, Errno::ECONNREFUSED => e
      raise ConnectionError, "Connection failed: #{e.message}"
    end
  end
end
