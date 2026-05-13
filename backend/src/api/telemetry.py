import logging
import os
import time
import boto3
import watchtower
from datetime import datetime

cloudwatch = boto3.client('cloudwatch', region_name=os.getenv("AWS_REGION", "us-east-1"))
NAMESPACE = "BrandGuardian/CompliancePipeline"



def setup_cloudwatch_logging(logger_name: str) -> logging.Logger:

    """
    Attach CloudWatch handler to any logger.
    """
    logger = logging.getLogger(logger_name)

    cw_handler = watchtower.CloudWatchLogHandler(
        log_group="/brand-guardian/compliance-pipeline",
        stream_name=f"{logger_name}-{datetime.utcnow().strftime('%Y-%m-%d')}",
        boto3_client=boto3.client('logs', region_name=os.getenv("AWS_REGION", "us-east-1"))
    )
    cw_handler.setLevel(logging.INFO)
    logger.addHandler(cw_handler)

    return logger



def put_metric(metric_name: str, value: float, unit: str = "Count", dimensions: dict = {}):
    """
    Push a custom metric to CloudWatch.
    """
    try:
        cloudwatch.put_metric_data(
            Namespace=NAMESPACE,
            MetricData=[{
                'MetricName': metric_name,
                'Value': value,
                'Unit': unit,
                'Dimensions': [
                    {'Name': k, 'Value': v}
                    for k, v in dimensions.items()
                ]
            }]
        )
    except Exception as e:
        logging.getLogger("cloudwatch").warning(f"Failed to push metric {metric_name}: {e}")


class MetricTimer:
    """
    Context manager to measure and log execution time.
    """
    def __init__(self, metric_name: str, dimensions: dict = {}):
        self.metric_name = metric_name
        self.dimensions = dimensions

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        elapsed_ms = (time.time() - self.start) * 1000
        put_metric(self.metric_name, elapsed_ms, unit="Milliseconds", dimensions=self.dimensions)


def log_audit_event(video_id: str, status: str, violations: int, duration_ms: float):
    """
    Log a full audit event as structured metrics.
    """
    put_metric("AuditCompleted", 1, dimensions={"VideoId": video_id})
    put_metric("AuditStatus", 1, dimensions={"Status": status})
    put_metric("ViolationsDetected", violations, dimensions={"VideoId": video_id})
    put_metric("AuditDurationMs", duration_ms, unit="Milliseconds")