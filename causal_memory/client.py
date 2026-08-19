"""
HydraDB HTTP client wrapper.

Matches the live query-engine contract:
- Rows are [[{"type": ..., "value": ...}, ...], ...].
- Value types seen in practice: vertex_id, integer, signed_integer, float,
  boolean, string, null, list, path, vertex_property.
- Path values carry nodes/relationships whose properties are wrapped as
  {"String": ..., "Integer": ..., "Float": ...}.
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
        cell_id: str = "cell-0",
        admin_url: Optional[str] = None,
        default_timeout_ms: int = 25_000,
    ):
        self.url = url.rstrip("/")
        self.auth_token = auth_token
        self.namespace = namespace
        self.cell_id = cell_id
        self.admin_url = admin_url or self.url
        # Engine admission control rejects client_query_runtime_ms > 30000.
        # Use a generous-but-valid deadline so graph-wide deletes and path
        # traversals on grown graphs do not hit the default short deadline.
        self.default_timeout_ms = default_timeout_ms
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {auth_token}",
            "X-Graph-Namespace": namespace,
            "Content-Type": "application/json",
        })

    def query(
        self,
        query: str,
        parameters: Optional[Dict[str, Any]] = None,
        bookmark: Optional[str] = None,
        timeout_ms: Optional[int] = None,
        page_size: Optional[int] = None,
        consistency: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Execute a Cypher query.

        Returns:
            Dict with keys: query_id, columns, rows, read_epoch, next_cursor, bookmark
        """
        body: Dict[str, Any] = {
            "cell_id": self.cell_id,
            "query": query,
        }

        if parameters:
            body["parameters"] = parameters
        if bookmark:
            body["bookmark"] = bookmark
        if timeout_ms:
            body["timeout_ms"] = timeout_ms
        elif self.default_timeout_ms:
            body["timeout_ms"] = self.default_timeout_ms
        if page_size:
            body["page_size"] = page_size
        if consistency:
            body["consistency"] = consistency

        response = self.session.post(
            f"{self.url}/v1/graphs/default/query",
            json=body,
        )
        response.raise_for_status()
        return response.json()

    def execute(self, query: str, parameters: Optional[Dict[str, Any]] = None, timeout_ms: Optional[int] = None) -> List[Dict]:
        """
        Execute a query and return rows as list of dicts.
        """
        result = self.query(query, parameters, timeout_ms=timeout_ms)
        columns = result.get("columns", [])
        rows = result.get("rows", [])

        result_rows = []
        for row in rows:
            row_dict = {}
            for col, val in zip(columns, row):
                row_dict[col] = self._parse_value(val)
            result_rows.append(row_dict)

        return result_rows

    def _parse_value(self, val: Any) -> Any:
        """Parse a HydraDB value into a Python type."""
        if val is None:
            return None

        if isinstance(val, dict) and "type" in val:
            val_type = val.get("type")
            val_value = val.get("value")

            if val_type == "null":
                return None
            elif val_type == "vertex_id":
                return val_value
            elif val_type in ("integer", "signed_integer"):
                return int(val_value)
            elif val_type == "float":
                return float(val_value)
            elif val_type == "boolean":
                return bool(val_value)
            elif val_type == "string":
                return val_value
            elif val_type == "list":
                return [self._parse_value(v) for v in (val_value or [])]
            elif val_type == "path":
                return self._parse_path(val_value or {})
            elif val_type == "vertex":
                return self._parse_path_node(val_value or {})
            elif val_type == "vertex_property":
                return self._parse_property_map(val_value)
            else:
                return val_value

        if isinstance(val, list):
            return [self._parse_value(v) for v in val]
        return val

    def _parse_property_map(self, value: Any) -> Dict[str, Any]:
        """Properties come as {"String": ..., "Integer": ..., "Float": ...}."""
        if isinstance(value, dict):
            for key in ("String", "Integer", "Float", "Boolean", "List"):
                if key in value:
                    inner = value[key]
                    if key == "Integer":
                        return int(inner)
                    if key == "Float":
                        return float(inner)
                    if key in ("String", "Boolean"):
                        return inner
                    if key == "List":
                        return [self._parse_property_map(v) for v in (inner or [])]
        return value

    def _parse_path_node(self, node: Dict) -> Dict:
        """A node in a path: {id, labels, properties}."""
        props = node.get("properties", {})
        parsed_props = {}
        for k, v in props.items():
            parsed_props[k] = self._parse_property_map(v)
        return {
            "id": node.get("id"),
            "labels": node.get("labels", []),
            "properties": parsed_props,
        }

    def _parse_path(self, path_data: Dict) -> Dict:
        """Parse a path result into dicts with nested property maps unwrapped."""
        nodes = [_parse_path_node_safe(n) for n in path_data.get("nodes", [])]
        rels = []
        for r in path_data.get("relationships", []):
            rprops = {}
            for k, v in (r.get("properties") or {}).items():
                rprops[k] = _parse_property_map_safe(v)
            rels.append({
                "id": r.get("id"),
                "type": r.get("edge_type"),
                "src": r.get("src"),
                "dst": r.get("dst"),
                "properties": rprops,
            })
        return {"nodes": nodes, "relationships": rels}

    def health_check(self) -> bool:
        """Check if HydraDB is healthy."""
        try:
            response = requests.get(f"{self.admin_url.replace(':8443', ':18443')}/healthz", timeout=5)
            if response.status_code == 200:
                return True
            response = requests.get(f"{self.admin_url}/healthz", timeout=5)
            return response.status_code == 200
        except Exception:
            return False

    def ready_check(self) -> bool:
        """Check if HydraDB is ready (admin port /readyz)."""
        try:
            response = requests.get(f"{self.admin_url}/readyz", timeout=5)
            return response.status_code == 200
        except Exception:
            return False


def _parse_property_map_safe(value: Any) -> Any:
    """Standalone copy for module-level reuse."""
    if isinstance(value, dict):
        for key in ("String", "Integer", "Float", "Boolean", "List"):
            if key in value:
                inner = value[key]
                if key == "Integer":
                    return int(inner)
                if key == "Float":
                    return float(inner)
                if key in ("String", "Boolean"):
                    return inner
                if key == "List":
                    return [_parse_property_map_safe(v) for v in (inner or [])]
    return value


def _parse_path_node_safe(node: Dict) -> Dict:
    props = {}
    for k, v in (node.get("properties") or {}).items():
        props[k] = _parse_property_map_safe(v)
    return {
        "id": node.get("id"),
        "labels": node.get("labels", []),
        "properties": props,
    }