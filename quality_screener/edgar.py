"""HTTP client for sec.gov (companyfacts, submissions, ticker map).

Every request sends ``config.USER_AGENT``, is rate-limited to ``config.MAX_REQUESTS_PER_SEC``,
retries with backoff on 429/5xx, and is cached under ``config.RAW_DIR`` so re-runs never
re-download. Not implemented yet.
"""
