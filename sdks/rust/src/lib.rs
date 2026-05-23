use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum MagnitudeError {
    #[error("connection error: {0}")]
    Connection(String),
    #[error("authentication error: {0}")]
    Auth(String),
    #[error("not found: {0}")]
    NotFound(String),
    #[error("api error: {0}")]
    Api(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tenant {
    pub id: String,
    pub name: String,
    pub max_databases: i64,
    pub max_collections: i64,
    pub created_at: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Database {
    pub id: String,
    pub tenant_id: String,
    pub name: String,
    pub created_at: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Collection {
    pub id: String,
    pub tenant_id: String,
    pub database_id: String,
    pub name: String,
    pub dimension: i64,
    pub metric: String,
    pub index_type: String,
    pub created_at: i64,
    pub vector_count: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    pub id: u64,
    pub distance: f32,
    pub score: f32,
    #[serde(default)]
    pub metadata: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchOptions {
    #[serde(rename = "topK", skip_serializing_if = "Option::is_none")]
    pub top_k: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub nprobe: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub filter: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InsertOptions {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<Vec<serde_json::Value>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Envelope<T> {
    pub data: Option<T>,
    pub error: Option<String>,
}

pub struct MagnitudeClient {
    base_url: String,
    client: Client,
    api_key: Option<String>,
}

impl MagnitudeClient {
    pub fn new(base_url: &str) -> Self {
        Self {
            base_url: base_url.trim_end_matches('/').to_string(),
            client: Client::new(),
            api_key: None,
        }
    }

    pub fn with_api_key(mut self, key: &str) -> Self {
        self.api_key = Some(key.to_string());
        self
    }

    async fn request<T: serde::de::DeserializeOwned>(
        &self,
        method: reqwest::Method,
        path: &str,
        body: Option<impl Serialize>,
    ) -> Result<T, MagnitudeError> {
        let url = format!("{}{}", self.base_url, path);
        let mut req = self.client.request(method, &url);

        if let Some(key) = &self.api_key {
            req = req.bearer_auth(key);
        }

        if let Some(b) = body {
            req = req.json(&b);
        }

        let resp = req
            .send()
            .await
            .map_err(|e| MagnitudeError::Connection(e.to_string()))?;

        let status = resp.status();
        let json: Envelope<T> = resp
            .json()
            .await
            .map_err(|e| MagnitudeError::Connection(e.to_string()))?;

        if !status.is_success() {
            let msg = json.error.unwrap_or_else(|| format!("HTTP {}", status));
            return match status.as_u16() {
                401 => Err(MagnitudeError::Auth(msg)),
                404 => Err(MagnitudeError::NotFound(msg)),
                _ => Err(MagnitudeError::Api(msg)),
            };
        }

        json.data
            .ok_or_else(|| MagnitudeError::Api("empty response".into()))
    }

    // ── Health ──────────────────────────────────────────────────────────────

    pub async fn health_check(&self) -> Result<serde_json::Value, MagnitudeError> {
        self.request(reqwest::Method::GET, "/v1/health", None::<()>)
            .await
    }

    // ── Tenants ─────────────────────────────────────────────────────────────

    pub async fn create_tenant(
        &self,
        name: &str,
        max_databases: Option<i64>,
        max_collections: Option<i64>,
    ) -> Result<Tenant, MagnitudeError> {
        let body = serde_json::json!({
            "name": name,
            "max_databases": max_databases.unwrap_or(0),
            "max_collections": max_collections.unwrap_or(0),
        });
        self.request(reqwest::Method::POST, "/api/v2/tenants", Some(body))
            .await
    }

    pub async fn list_tenants(&self) -> Result<Vec<Tenant>, MagnitudeError> {
        self.request(reqwest::Method::GET, "/api/v2/tenants", None::<()>)
            .await
    }

    pub async fn get_tenant(&self, id: &str) -> Result<Tenant, MagnitudeError> {
        self.request(
            reqwest::Method::GET,
            &format!("/api/v2/tenants/{}", id),
            None::<()>,
        )
        .await
    }

    pub async fn delete_tenant(&self, id: &str) -> Result<(), MagnitudeError> {
        self.request::<serde_json::Value>(
            reqwest::Method::DELETE,
            &format!("/api/v2/tenants/{}", id),
            None::<()>,
        )
        .await?;
        Ok(())
    }

    // ── Databases ───────────────────────────────────────────────────────────

    pub async fn create_database(
        &self,
        tenant_id: &str,
        name: &str,
    ) -> Result<Database, MagnitudeError> {
        let body = serde_json::json!({ "name": name });
        self.request(
            reqwest::Method::POST,
            &format!("/api/v2/tenants/{}/databases", tenant_id),
            Some(body),
        )
        .await
    }

    pub async fn list_databases(&self, tenant_id: &str) -> Result<Vec<Database>, MagnitudeError> {
        self.request(
            reqwest::Method::GET,
            &format!("/api/v2/tenants/{}/databases", tenant_id),
            None::<()>,
        )
        .await
    }

    pub async fn delete_database(
        &self,
        tenant_id: &str,
        database_id: &str,
    ) -> Result<(), MagnitudeError> {
        self.request::<serde_json::Value>(
            reqwest::Method::DELETE,
            &format!("/api/v2/tenants/{}/databases/{}", tenant_id, database_id),
            None::<()>,
        )
        .await?;
        Ok(())
    }

    // ── Collections ─────────────────────────────────────────────────────────

    pub async fn create_collection(
        &self,
        tenant_id: &str,
        database_id: &str,
        name: &str,
        dimension: i64,
        metric: &str,
    ) -> Result<Collection, MagnitudeError> {
        let body = serde_json::json!({
            "name": name,
            "dimension": dimension,
            "metric": metric,
        });
        self.request(
            reqwest::Method::POST,
            &format!(
                "/api/v2/tenants/{}/databases/{}/collections",
                tenant_id, database_id
            ),
            Some(body),
        )
        .await
    }

    pub async fn list_collections(
        &self,
        tenant_id: &str,
        database_id: &str,
    ) -> Result<Vec<Collection>, MagnitudeError> {
        self.request(
            reqwest::Method::GET,
            &format!(
                "/api/v2/tenants/{}/databases/{}/collections",
                tenant_id, database_id
            ),
            None::<()>,
        )
        .await
    }

    pub async fn delete_collection(
        &self,
        tenant_id: &str,
        database_id: &str,
        collection_id: &str,
    ) -> Result<(), MagnitudeError> {
        self.request::<serde_json::Value>(
            reqwest::Method::DELETE,
            &format!(
                "/api/v2/tenants/{}/databases/{}/collections/{}",
                tenant_id, database_id, collection_id
            ),
            None::<()>,
        )
        .await?;
        Ok(())
    }

    // ── Vectors ─────────────────────────────────────────────────────────────

    pub async fn insert(
        &self,
        tenant_id: &str,
        database_id: &str,
        collection_id: &str,
        ids: &[u64],
        vectors: &[Vec<f32>],
        options: Option<InsertOptions>,
    ) -> Result<usize, MagnitudeError> {
        let body = serde_json::json!({
            "ids": ids,
            "vectors": vectors,
            "metadata": options.and_then(|o| o.metadata),
        });
        let result: HashMap<String, usize> = self
            .request(
                reqwest::Method::POST,
                &format!(
                    "/api/v2/tenants/{}/databases/{}/collections/{}/add",
                    tenant_id, database_id, collection_id
                ),
                Some(body),
            )
            .await?;
        Ok(result.get("inserted").copied().unwrap_or(0))
    }

    pub async fn search(
        &self,
        tenant_id: &str,
        database_id: &str,
        collection_id: &str,
        query: &[f32],
        options: Option<SearchOptions>,
    ) -> Result<Vec<SearchResult>, MagnitudeError> {
        let opts = options.unwrap_or(SearchOptions {
            top_k: Some(10),
            nprobe: None,
            filter: None,
        });
        let body = serde_json::json!({
            "query": query,
            "k": opts.top_k.unwrap_or(10),
            "nprobe": opts.nprobe.unwrap_or(0),
            "filter": opts.filter,
        });
        self.request(
            reqwest::Method::POST,
            &format!(
                "/api/v2/tenants/{}/databases/{}/collections/{}/query",
                tenant_id, database_id, collection_id
            ),
            Some(body),
        )
        .await
    }

    pub async fn delete_vectors(
        &self,
        tenant_id: &str,
        database_id: &str,
        collection_id: &str,
        ids: &[u64],
    ) -> Result<(), MagnitudeError> {
        let body = serde_json::json!({ "ids": ids });
        self.request::<serde_json::Value>(
            reqwest::Method::POST,
            &format!(
                "/api/v2/tenants/{}/databases/{}/collections/{}/delete",
                tenant_id, database_id, collection_id
            ),
            Some(body),
        )
        .await?;
        Ok(())
    }
}
