"""Our MCP server: the `run_python` sandbox, the artifact store, and the metric log.

`sandbox.py` and `store.py` are the implementations; `src/ds_agents/tools/local.py` binds them
to the `Tools` Protocol in-process and the protocol layer will bind the same objects over MCP.
Nothing here may import `ds_agents` into the sandbox worker -- see `_worker.py`.
"""
