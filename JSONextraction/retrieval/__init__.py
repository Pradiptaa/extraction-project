"""Retrieval layer: embedding view, Chroma load, retrievers, the regression gate,
and the optional chat layer. See retrieval/README.md.

Chroma telemetry is switched off here, at package import, rather than in
`config.load_settings`. chromadb 0.5.23 defaults `anonymized_telemetry` to True
and reads it from the environment when a client is created, so anything that
opened a client without loading `retrieval/.env` first — every unit test, a
one-off script, `reindex` before settings were read — ran with it on. It
must be set before the first client exists: Chroma refuses a second client on
the same path with different settings, so it cannot be passed per client
without every caller (tests included) agreeing.

An explicit ANONYMIZED_TELEMETRY in the environment still wins. An EMPTY one is
treated as unset: chromadb parses it as a bool and refuses "" at client
creation, so a blank `ANONYMIZED_TELEMETRY=` line would break every command.

The logger is silenced because the installed posthog version does not match
chromadb's call signature: every event raises inside chromadb and is logged as
"Failed to send telemetry event ..." even with telemetry disabled (the disabled
flag is checked after the failing call). Nothing is sent either way; the noise
only buries real log output.
"""
import logging
import os

if not os.environ.get("ANONYMIZED_TELEMETRY", "").strip():
    os.environ["ANONYMIZED_TELEMETRY"] = "False"
logging.getLogger("chromadb.telemetry.product.posthog").setLevel(logging.CRITICAL)
