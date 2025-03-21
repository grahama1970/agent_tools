# Monitoring Implementation for QA Generation

This document describes the monitoring system implemented for the QA generation pipeline. It covers the metrics collection, alerting, and integration with other components.

## Overview

The monitoring system provides comprehensive visibility into the QA generation pipeline's performance, health, and resource utilization. It offers:

1. Metrics collection for performance monitoring
2. Alerting for error conditions and performance issues
3. Integration with cache metrics
4. Detailed timing measurements for operations
5. Success/failure tracking

## Components

### 1. Metrics Collection System

The monitoring system provides two implementation options:

- **Prometheus Integration**: When `prometheus_client` is available, metrics are stored in Prometheus format, supporting standard monitoring ecosystems.
- **In-Memory Fallback**: If Prometheus is unavailable, an in-memory store tracks metrics, ensuring monitoring works in all environments.

#### Key Metrics Tracked

| Metric Name | Type | Description |
|-------------|------|-------------|
| `qa_generation_time` | Histogram | Time taken to generate QA pairs, by section type |
| `qa_pairs_generated` | Counter | Number of QA pairs generated, by section type |
| `qa_errors_total` | Counter | Total number of errors in QA generation, by error type |
| `qa_cache_hit` | Counter | Number of cache hits |
| `qa_cache_miss` | Counter | Number of cache misses |
| `qa_api_calls` | Counter | Number of API calls made, by model and status |
| `qa_processing_duration` | Histogram | Total processing time for extraction |
| `qa_worker_utilization` | Gauge | Worker pool utilization percentage |

### 2. Alerting System

The monitoring system includes an alerting mechanism that checks metrics against defined thresholds:

- **Error Threshold**: Alerts when error count exceeds 5 (configurable)
- **Processing Time Threshold**: Alerts when processing time exceeds 30 seconds (configurable)
- **Cache Hit Rate Threshold**: Alerts when cache hit rate falls below 20% (configurable)

Alerts are:
- Logged with appropriate severity levels
- Written to an `alerts.jsonl` file
- Ready for integration with external alerting systems (Slack, email, etc.)

### 3. Context Manager for Timing

The `MetricsContextManager` provides a clean way to measure execution time:

```python
# Example usage
from agent_tools.dualipa.qa.monitoring import MetricsContextManager, QA_PROCESSING_DURATION

with MetricsContextManager(QA_PROCESSING_DURATION):
    # Code to be timed
    await process_complex_operation()
```

The context manager automatically:
- Records start and end time
- Calculates elapsed time
- Updates appropriate metric
- Records errors if exceptions occur

## Integration Points

### 1. Processor Integration

The monitoring system integrates with `processor.py`:

- Optional monitoring via `enable_monitoring` parameter
- Full process timing with context manager
- Worker utilization metrics
- Section type tracking
- Error handling with monitoring

### 2. Cache Integration

Monitoring pulls statistics from the cache system:

- Cache hit/miss counts
- Hit rate calculation
- Alert on low hit rates

### 3. Batch Processing Integration

Monitors effectiveness of worker pools:

- Worker utilization tracking
- Processing times by section type
- Error rate monitoring

## Usage Examples

### 1. Basic Metrics Recording

```python
from agent_tools.dualipa.qa.monitoring import record_metric, QA_PAIRS_GENERATED

# Record a metric with labels
record_metric(QA_PAIRS_GENERATED, 5, {"section_type": "text"})
```

### 2. Getting Current Metrics

```python
from agent_tools.dualipa.qa.monitoring import get_processing_metrics

# Get all current metrics
metrics = get_processing_metrics()
print(f"Generated {metrics['pairs_generated']} QA pairs")
print(f"Cache hit rate: {metrics['cache_hit_rate']:.2f}")
```

### 3. Alerting

```python
from agent_tools.dualipa.qa.monitoring import check_alert_conditions, send_alert

# Check all alert conditions
check_alert_conditions()

# Send a specific alert
send_alert("Custom alert message", severity="warning")
```

### 4. Full Pipeline with Monitoring

```python
from agent_tools.dualipa.qa.processor import process_extraction_json
from agent_tools.dualipa.qa.models.config import QAGenerationConfig

# Process with monitoring enabled
response = await process_extraction_json(
    input_data=input_json,
    config=QAGenerationConfig(),
    enable_monitoring=True
)

# Metrics are included in response
print(response.generation_metadata["metrics"])
```

## Customization

### 1. Alert Thresholds

The default alert thresholds can be adjusted in `monitoring.py`:

```python
# Alert thresholds
ALERT_ERROR_THRESHOLD = 5  # Trigger alert after 5 errors
ALERT_PROCESSING_TIME_THRESHOLD = 30.0  # 30 seconds
ALERT_CACHE_HIT_RATE_THRESHOLD = 0.2  # 20%
```

### 2. Custom Metrics

New metrics can be added by updating the constants and Prometheus metrics in `monitoring.py`.

## Testing

The monitoring system includes comprehensive tests:

1. `test_process_extraction_json_full_real`: End-to-end test of full pipeline with monitoring
2. `test_process_metadata_with_metrics`: Test of metrics collection and retrieval
3. `test_error_alerting`: Test of alert triggering based on thresholds
4. `test_cache_metrics_integration`: Test of integration with cache metrics

## Best Practices

1. **Always Initialize**: Ensure metrics are initialized before use with `initialize_metrics()`
2. **Use Context Managers**: Use `MetricsContextManager` for timing operations
3. **Check Alerts Regularly**: Call `check_alert_conditions()` after significant operations
4. **Label Metrics**: Use appropriate labels (e.g., section_type, error_type) for detailed analysis
5. **Integrate Cache Stats**: Call `integrate_cache_metrics()` to get the latest cache performance data

## Future Enhancements

1. Add support for more alerting channels (Slack, email, etc.)
2. Implement real-time monitoring dashboard
3. Add more granular metrics for LLM-specific operations
4. Support metric aggregation across multiple processes
5. Add historical metrics tracking and trend analysis