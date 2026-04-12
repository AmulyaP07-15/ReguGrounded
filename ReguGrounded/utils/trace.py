# utils/trace.py
#
# Request-scoped trace ID using Python's contextvars.
#
# A single ContextVar holds the current trace ID for the executing thread/task.
# Set it once at the top of query_interface.run(); every module that calls
# current_trace_id() will see the same value for the lifetime of that request,
# with zero changes to function signatures.
#
# The TraceIdFilter attaches the ID to every log record so every log line
# automatically shows which request it belongs to.

import contextvars
import logging
import uuid

_trace_id_var: contextvars.ContextVar[str] = contextvars.ContextVar(
    "trace_id", default="no-trace"
)


def set_trace_id(trace_id: str | None = None) -> str:
    """
    Set the trace ID for the current context.
    Generates a short UUID if none supplied.
    Returns the ID that was set.
    """
    tid = trace_id or uuid.uuid4().hex[:12]
    _trace_id_var.set(tid)
    return tid


def current_trace_id() -> str:
    """Return the trace ID for the current request context."""
    return _trace_id_var.get()


class TraceIdFilter(logging.Filter):
    """
    Logging filter that injects `trace_id` into every LogRecord.
    Add to any handler or logger to get trace IDs in log output.
    """
    def filter(self, record: logging.LogRecord) -> bool:
        record.trace_id = _trace_id_var.get()
        return True
