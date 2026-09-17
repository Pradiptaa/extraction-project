import logging
import os

if not os.environ.get("ANONYMIZED_TELEMETRY", "").strip():
    os.environ["ANONYMIZED_TELEMETRY"] = "False"
logging.getLogger("chromadb.telemetry.product.posthog").setLevel(logging.CRITICAL)
