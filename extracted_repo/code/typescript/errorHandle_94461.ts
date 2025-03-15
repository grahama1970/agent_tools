import { z } from 'zod';
import { REPOMIX_DISCORD_URL, REPOMIX_ISSUES_URL } from './constants.js';
import { logger, repomixLogLevels } from './logger.js';

export class RepomixError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'RepomixError';
  }
}

export class RepomixConfigValidationError extends RepomixError {
  constructor(message: string) {
    super(message);
    this.name = 'RepomixConfigValidationError';
  }
}

export const handleError = (error: unknown): void => {
  logger.log('');

  if (error instanceof RepomixError) {
    logger.error(`✖ ${error.message}`);
    // If expected error, show stack trace for debugging
    logger.debug('Stack trace:', error.stack);
  } else if (error instanceof Error) {
    logger.error(`✖ Unexpected error: ${error.message}`);
    // If unexpected error, show stack trace by default
    logger.note('Stack trace:', error.stack);

    if (logger.getLogLevel() < repomixLogLevels.DEBUG) {
      logger.log('');
      logger.note('For detailed debug information, use the --verbose flag');
    }
  } else {
    // Unknown errors
    logger.error('✖ An unknown error occurred');

    if (logger.getLogLevel() < repomixLogLevels.DEBUG) {
      logger.note('For detailed debug information, use the --verbose flag');
    }
  }

  // Community support information
  logger.log('');
  logger.info('Need help?');
  logger.info(`• File an issue on GitHub: ${REPOMIX_ISSUES_URL}`);
  logger.info(`• Join our Discord community: ${REPOMIX_DISCORD_URL}`);
};

export const rethrowValidationErrorIfZodError = (error: unknown, message: string): void => {
  if (error instanceof z.ZodError) {
    const zodErrorText = error.errors.map((err) => `[${err.path.join('.')}] ${err.message}`).join('\n  ');
    throw new RepomixConfigValidationError(
      `${message}\n\n  ${zodErrorText}\n\n  Please check the config file and try again.`,
    );
  }
};
