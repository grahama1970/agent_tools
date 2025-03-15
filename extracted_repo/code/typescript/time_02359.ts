/**
 * Convert milliseconds to a Duration string that Cloud Logging expects
 * Format: "1.500s" -> { seconds: 1, nanos: 500000000 }
 */
export function formatDuration(durationMs: number): {
  seconds: number;
  nanos: number;
} {
  const seconds = Math.floor(durationMs / 1000);
  const nanos = (durationMs % 1000) * 1_000_000; // Convert remaining ms to nanoseconds
  return { seconds, nanos };
}

/**
 * Calculate latency between start time and now
 */
export function calculateLatency(startTime: number): {
  seconds: number;
  nanos: number;
} {
  const latencyMs = Date.now() - startTime;
  return formatDuration(latencyMs);
}

/**
 * Format latency for display purposes (not for Cloud Logging)
 */
export function formatLatencyForDisplay(startTime: number): string {
  const latencyMs = Date.now() - startTime;
  return `${(latencyMs / 1000).toFixed(3)}s`;
}
