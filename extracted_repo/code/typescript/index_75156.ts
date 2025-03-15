import { serve } from '@hono/node-server';
import { Hono } from 'hono';
import { bodyLimit } from 'hono/body-limit';
import { compress } from 'hono/compress';
import { cors } from 'hono/cors';
import { timeout } from 'hono/timeout';
import { FILE_SIZE_LIMITS } from './constants.js';
import { processZipFile } from './processZipFile.js';
import { processRemoteRepo } from './remoteRepo.js';
import type { PackResult } from './types.js';
import { handlePackError } from './utils/errorHandler.js';
import { cloudLogger, createErrorResponse, logError, logInfo } from './utils/logger.js';
import { getProcessConcurrency } from './utils/processConcurrency.js';
import { calculateLatency, formatLatencyForDisplay } from './utils/time.js';

// Log server metrics on startup
logInfo('Server starting', {
  metrics: {
    processConcurrency: getProcessConcurrency(),
  },
});

const app = new Hono();

// Setup custom logger
app.use('*', cloudLogger());

// Configure CORS
app.use(
  '/*',
  cors({
    origin: ['http://localhost:5173', 'https://repomix.com', 'https://api.repomix.com'],
    allowMethods: ['GET', 'POST', 'OPTIONS'],
    allowHeaders: ['Content-Type'],
    maxAge: 86400,
    credentials: true,
  }),
);

// Enable compression
app.use(compress());

// Set timeout for API routes
app.use('/api', timeout(30000));

// Health check endpoint
app.get('/health', (c) => c.text('OK'));

// Main packing endpoint
app.post(
  '/api/pack',
  bodyLimit({
    maxSize: FILE_SIZE_LIMITS.MAX_REQUEST_SIZE,
    onError: (c) => {
      const requestId = c.get('requestId');
      const response = createErrorResponse('File size too large', requestId);
      return c.json(response, 413);
    },
  }),
  async (c) => {
    try {
      const formData = await c.req.formData();
      const requestId = c.get('requestId');

      // Get form data
      const format = formData.get('format') as 'xml' | 'markdown' | 'plain';
      const options = JSON.parse(formData.get('options') as string);
      const file = formData.get('file') as File | null;
      const url = formData.get('url') as string | null;

      // Validate input
      if (!file && !url) {
        return c.json(createErrorResponse('Either repository URL or file is required', requestId), 400);
      }

      if (!['xml', 'markdown', 'plain'].includes(format)) {
        return c.json(createErrorResponse('Invalid format specified', requestId), 400);
      }

      // Get client IP
      const clientIp =
        c.req.header('x-forwarded-for')?.split(',')[0] ||
        c.req.header('x-real-ip') ||
        c.req.header('cf-connecting-ip') ||
        '0.0.0.0';

      const startTime = Date.now();

      // Process file or repository
      let result: PackResult;
      if (file) {
        result = await processZipFile(file, format, options, clientIp);
      } else {
        if (!url) {
          return c.json(createErrorResponse('Repository URL is required', requestId), 400);
        }
        result = await processRemoteRepo(url, format, options, clientIp);
      }

      // Log operation result
      logInfo('Pack operation completed', {
        requestId,
        format,
        repository: result.metadata.repository,
        duration: formatLatencyForDisplay(startTime),
        inputType: file ? 'file' : url ? 'url' : 'unknown',
        metrics: {
          totalFiles: result.metadata.summary?.totalFiles,
          totalCharacters: result.metadata.summary?.totalCharacters,
          totalTokens: result.metadata.summary?.totalTokens,
        },
      });

      return c.json(result);
    } catch (error) {
      // Handle errors
      logError('Pack operation failed', error instanceof Error ? error : new Error('Unknown error'), {
        requestId: c.get('requestId'),
      });

      const appError = handlePackError(error);
      return c.json(createErrorResponse(appError.message, c.get('requestId')), appError.statusCode);
    }
  },
);

// Start server
const port = process.env.PORT ? Number.parseInt(process.env.PORT, 10) : 3000;
logInfo(`Server starting on port ${port}`);

serve({
  fetch: app.fetch,
  port,
});

// Export app for testing
export default app;
