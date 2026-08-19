"""Backend package: audio analysis, the trained CNN and dataset tooling."""

# Route TLS through the OS trust store - antivirus HTTPS interception (see DEPLOY.md)
try:
    import truststore

    truststore.inject_into_ssl()
except Exception:
    pass
