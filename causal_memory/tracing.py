"""
Per-query tracing for the causal-memory retrieval pipeline.

Every retrieval tool call records a timeline of stages (resolve target ->
traverse graph -> narrate) with per-stage elapsed ms. Traces are appended to
a JSONL file so the graph viewer (a separate process) can stream them live
in real time via /api/activity.
"""

import json
import os
import threading
import time

_TRACE_PATH = os.environ.get("HYDRADNA_TRACE_PATH", "/tmp/hydradna-trace.jsonl")
_lock = threading.Lock()


class Stage:
    __slots__ = ("name", "start_ms", "end_ms", "detail")

    def __init__(self, name: str, start_ms: float, detail: str = ""):
        self.name = name
        self.start_ms = start_ms
        self.end_ms = start_ms
        self.detail = detail

    def finish(self) -> None:
        self.end_ms = time.monotonic() * 1000

    def elapsed(self) -> float:
        return round(self.end_ms - self.start_ms, 1)


class Trace:
    def __init__(self, tool: str, query: str = ""):
        self.id = f"{int(time.time() * 1000)}-{os.getpid()}"
        self.tool = tool
        self.query = query
        self.start = time.monotonic() * 1000
        self.end = self.start
        self.stages: list[Stage] = []
        self.result = ""

    def stage(self, name: str, detail: str = "") -> Stage:
        s = Stage(name, time.monotonic() * 1000, detail)
        self.stages.append(s)
        return s

    def finish(self, result: str = "") -> None:
        self.end = time.monotonic() * 1000
        self.result = result

    def total_ms(self) -> float:
        return round(self.end - self.start, 1)

    def to_json(self) -> dict:
        return {
            "id": self.id,
            "tool": self.tool,
            "query": self.query,
            "start": int(self.start),
            "total_ms": self.total_ms(),
            "stages": [
                {"name": s.name, "ms": s.elapsed(), "detail": s.detail}
                for s in self.stages
            ],
            "result": self.result[:300],
        }

    def emit(self) -> None:
        self.finish()
        try:
            with _lock:
                with open(_TRACE_PATH, "a") as f:
                    f.write(json.dumps(self.to_json()) + "\n")
        except OSError:
            pass


def recent_traces(limit: int = 50) -> list[dict]:
    """Read the most recent traces from the JSONL log (newest first)."""
    traces = []
    try:
        with open(_TRACE_PATH) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        traces.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
    except OSError:
        return []
    return traces[-limit:][::-1]