"""
HydraDB HTTP client wrapper.
"""

import requests
from typing import Optional, Dict, Any, List


class HydraDBClient:
    """HTTP client for HydraDB query API."""
    
    def __init__(
        self,
        url: str = "http://localhost:18443",
        auth_token: str = "local-dev-auth-token-32-characters-long",
        namespace: str = "default",
        cell_id: str = "cell-0"
    ):
        self.url = url.rstrip("/")
        self.auth_token = auth_token
        self.namespace = namespace
        self.cell_id = cell_id
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {auth_token}",
            "X-Graph-Namespace": namespace,
            "Content-Type": "application/json"
        })
    
    def query(
        self,
        query: str,
        parameters: Optional[Dict[str, Any]] = None,
        bookmark: Optional[str] = None,
        timeout_ms: Optional[int] = None,
        page_size: Optional[int] = None,
        consistency: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Execute a Cypher query.
        
        Args:
            query: Cypher query string
            parameters: Query parameters
            bookmark: Causal read position
            timeout_ms: Query timeout in milliseconds
            page_size: Number of rows per page
            consistency: "causal" or "strong"
        
        Returns:
            Dict with keys: query_id, columns, rows, read_epoch, next_cursor, bookmark
        """
        body = {
            "cell_id": self.cell_id,
            "query": query
        }
        
        if parameters:
            body["parameters"] = parameters
        if bookmark:
            body["bookmark"] = bookmark
        if timeout_ms:
            body["timeout_ms"] = timeout_ms
        if page_size:
            body["page_size"] = page_size
        if consistency:
            body["consistency"] = consistency
        
        response = self.session.post(
            f"{self.url}/v1/graphs/default/query",
            json=body
        )
        response.raise_for_status()
        return response.json()
    
    def execute(self, query: str, parameters: Optional[Dict[str, Any]] = None) -> List[Dict]:
        """
        Execute a query and return rows as list of dicts.
        
        Args:
            query: Cypher query string
            parameters: Query parameters
        
        Returns:
            List of row dicts with column names as keys
        """
        result = self.query(query, parameters)
        columns = result.get("columns", [])
        rows = result.get("rows", [])
        
        result_rows = []
        for row in rows:
            row_dict = {}
            for col, val in zip(columns, row):
                row_dict[col] = self._parse_value(val)
            result_rows.append(row_dict)
        
        return result_rows
    
    def _parse_value(self, val: Dict) -> Any:
        """Parse a HydraDB value into Python type."""
        if not val:
            return None
        
        val_type = val.get("type")
        val_value = val.get("value")
        
        if val_type == "null" or val is None:
            return None
        elif val_type == "vertex_id":
            return val_value
        elif val_type == "integer":
            return val_value
        elif val_type == "signed_integer":
            return val_value
        elif val_type == "float":
            return val_value
        elif val_type == "boolean":
            return val_value
        elif val_type == "string":
            return val_value
        elif val_type == "list":
            return [self._parse_value(v) for v in val_value]
        elif val_type == "path":
            return self._parse_path(val_value)
        else:
            return val_value
    
    def _parse_path(self, path_data: Dict) -> Dict:
        """Parse a path result."""
        return {
            "nodes": [self._parse_value(n) for n in path_data.get("nodes", [])],
            "relationships": [self._parse_value(r) for r in path_data.get("relationships", [])]
        }
    
    def health_check(self) -> bool:
        """Check if HydraDB is healthy."""
        try:
            response = requests.get(f"{self.url}/healthz", timeout=5)
            return response.status_code == 200
        except Exception:
            return False
    
    def ready_check(self) -> bool:
        """Check if HydraDB is ready (using admin port)."""
        try:
            # Admin port is typically 19091
            admin_url = self.url.replace(":18443", ":19091")
            response = requests.get(f"{admin_url}/readyz", timeout=5)
            return response.status_code == 200
        except Exception:
            return False
