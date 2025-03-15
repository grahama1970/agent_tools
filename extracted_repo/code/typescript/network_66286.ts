import type { Context } from 'hono';

/**
 * Get client IP address
 * Handles various headers in Cloud Run environment
 */
export function getClientIP(c: Context): string {
  return (
    c.req.header('x-forwarded-for')?.split(',')[0] ||
    c.req.header('x-real-ip') ||
    c.req.header('cf-connecting-ip') ||
    '0.0.0.0'
  );
}
