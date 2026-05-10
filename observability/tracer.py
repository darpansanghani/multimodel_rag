from opentelemetry import trace
from contextlib import contextmanager

def get_tracer():
    return trace.get_tracer("rag-pipeline")

@contextmanager
def span(name, attributes=None):
    with get_tracer().start_as_current_span(name) as s:
        if attributes:
            for k, v in attributes.items():
                s.set_attribute(k, str(v))
        yield s
