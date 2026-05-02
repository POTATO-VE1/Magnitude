import requests
import json
import urllib3
from typing import List, Dict, Any

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

class VectorDBClient:
    def __init__(self, host: str = "https://127.0.0.1:8443", admin_host: str = "http://127.0.0.1:9090"):
        self.host = host
        self.admin_host = admin_host
        self.tenant_id = None
        self.db_id = None
        self.api_key = None

    def get_or_create_tenant(self, name: str) -> str:
        # Check if exists
        res = requests.get(f"{self.admin_host}/api/v2/tenants", verify=False)
        if res.status_code == 200:
            for t in (res.json().get("data") or []):
                if t.get("name") == name:
                    return t["id"]
                    
        # Otherwise create
        res = requests.post(f"{self.admin_host}/api/v2/tenants", json={
            "name": name,
            "max_databases": 10,
            "max_collections": 100
        }, verify=False)
        if res.status_code == 201:
            return res.json()["data"]["id"]
        raise Exception(f"Tenant creation failed: {res.text}")

    def get_or_create_database(self, tenant_id: str, name: str) -> str:
        res = requests.get(f"{self.admin_host}/api/v2/tenants/{tenant_id}/databases", verify=False)
        if res.status_code == 200:
            for db in (res.json().get("data") or []):
                if db.get("name") == name:
                    return db["id"]
                    
        res = requests.post(f"{self.admin_host}/api/v2/tenants/{tenant_id}/databases", json={
            "name": name
        }, verify=False)
        if res.status_code == 201:
            return res.json()["data"]["id"]
        raise Exception(f"Database creation failed: {res.text}")

    def get_or_create_collection(self, tenant_id: str, db_id: str, name: str, dimension: int, metric: str = "cosine", index_type: str = "hnsw") -> str:
        res = requests.get(f"{self.host}/api/v2/tenants/{tenant_id}/databases/{db_id}/collections", verify=False)
        if res.status_code == 200:
            for c in (res.json().get("data") or []):
                if c.get("name") == name:
                    return c["id"]
                    
        res = requests.post(f"{self.host}/api/v2/tenants/{tenant_id}/databases/{db_id}/collections", json={
            "name": name,
            "dimension": dimension,
            "metric": metric,
            "index_type": index_type
        }, verify=False)
        if res.status_code == 201:
            return res.json()["data"]["id"]
        raise Exception(f"Collection creation failed: {res.text}")

    def insert_vectors(self, tenant_id: str, db_id: str, col_id: str, ids: List[int], vectors: List[List[float]], metadata: List[Dict[str, Any]] = None):
        payload = {
            "ids": ids,
            "vectors": vectors
        }
        if metadata:
            payload["metadata"] = metadata
            
        res = requests.post(f"{self.host}/api/v2/tenants/{tenant_id}/databases/{db_id}/collections/{col_id}/add", json=payload, verify=False)
        if res.status_code not in (200, 201):
            raise Exception(f"Failed to insert vectors: {res.text}")
        return res.json()

    def search_vectors(self, tenant_id: str, db_id: str, col_id: str, query_embedding: List[float], top_k: int = 10) -> List[Dict]:
        payload = {
            "query": query_embedding,
            "k": top_k
        }
        res = requests.post(f"{self.host}/api/v2/tenants/{tenant_id}/databases/{db_id}/collections/{col_id}/query", json=payload, verify=False)
        if res.status_code != 200:
            raise Exception(f"Failed to search vectors: {res.text}")
        return res.json()["data"]

    def hybrid_search(self, tenant_id: str, db_id: str, col_id: str, query_embedding: List[float], query_text: str, top_k: int = 10) -> List[Dict]:
        payload = {
            "query_embedding": query_embedding,
            "query_text": query_text,
            "top_k": top_k
        }
        res = requests.post(f"{self.host}/api/v2/tenants/{tenant_id}/databases/{db_id}/collections/{col_id}/hybrid", json=payload, verify=False)
        if res.status_code != 200:
            raise Exception(f"Failed to hybrid search: {res.text}")
        return res.json()["data"]
