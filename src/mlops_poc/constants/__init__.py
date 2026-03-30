"""Project-wide constants."""

FEATURE_COLUMNS = [
    "cpu_usage_pct",
    "memory_usage_pct",
    "pod_restart_count",
    "request_latency_p99_ms",
    "error_rate_pct",
    "network_bytes_in",
    "network_bytes_out",
    "disk_io_read_mbps",
    "disk_io_write_mbps",
    "pod_pending_count",
    "node_not_ready_count",
]


class Severity:
    OK = "OK"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"
