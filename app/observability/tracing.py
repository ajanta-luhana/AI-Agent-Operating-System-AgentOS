import functools, time
from langgraph.errors import GraphBubbleUp
from app.llm.client import USAGE
from app.observability.events import emit


def traced(name: str):
    """Wrap a graph node: records start/end/error, latency and token usage."""
    def deco(fn):
        @functools.wraps(fn)
        def wrapper(state, *a, **kw):
            run_id = state.get("run_id")
            emit(run_id, name, "start")
            t0, u0 = time.time(), dict(USAGE)
            try:
                out = fn(state, *a, **kw)
            except GraphBubbleUp:               # interrupt() is control flow, not an error
                raise
            except Exception as e:
                emit(run_id, name, "error", {"error": repr(e)}, int((time.time() - t0) * 1000))
                raise
            emit(run_id, name, "end",
                 {"prompt_tokens": USAGE["prompt"] - u0["prompt"],
                  "completion_tokens": USAGE["completion"] - u0["completion"]},
                 int((time.time() - t0) * 1000))
            return out
        return wrapper
    return deco
