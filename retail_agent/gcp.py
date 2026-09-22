"""Which Google Cloud project BigQuery jobs and Vertex AI calls go to.

One answer for both, so the warehouse and the model can never bill two
different projects:

  1. GOOGLE_CLOUD_PROJECT, if set — an override, not a requirement;
  2. otherwise the project Application Default Credentials resolve to, which is
     the one `gcloud config set project` chose;
  3. otherwise the quota project that `gcloud auth application-default login`
     wrote into the credentials file.

Step 3 matters because google.auth finds the configured project by running the
gcloud binary. Where gcloud is not on PATH — a shell opened before the SDK was
installed or moved, a CI runner — step 2 finds nothing, even though the
credentials themselves name a project.
"""

from __future__ import annotations

import os


def resolve_project(explicit: str | None = None) -> str | None:
    """The project to use, or None when nothing names one."""
    if explicit:
        return explicit
    if configured := os.getenv("GOOGLE_CLOUD_PROJECT"):
        return configured
    try:
        import google.auth

        credentials, project = google.auth.default()
    except Exception:  # noqa: BLE001 - "no credentials" is reported by the caller
        return None
    return project or getattr(credentials, "quota_project_id", None) or None
