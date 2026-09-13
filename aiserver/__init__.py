"""CaseSorter AI Server package.

Everything beyond the original OpenAI-compatible inference endpoints lives
here: the SQLite-backed model registry, training-image storage, the
out-of-process ConvNeXt trainer, evaluation, model ZIP import/export,
community login/download, the remote-client API, and the web UI.
"""

__version__ = "0.2.0"
