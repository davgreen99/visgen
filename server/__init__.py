"""Backend package: audio analysis, the trained CNN and dataset tooling."""

# Route TLS through the OS trust store - works around antivirus HTTPS interception
try:
    import truststore

    truststore.inject_into_ssl()
except Exception:
    pass
