import clipboard from 'clipboardy';
import type { RepomixConfigMerged } from '../../config/configSchema.js';
import { logger } from '../../shared/logger.js';
import type { RepomixProgressCallback } from '../../shared/types.js';

// Additionally copy to clipboard if flag is raised
export const copyToClipboardIfEnabled = async (
  output: string,
  progressCallback: RepomixProgressCallback,
  config: RepomixConfigMerged,
): Promise<undefined> => {
  if (config.output.copyToClipboard) {
    progressCallback('Copying to clipboard...');
    logger.trace('Copying output to clipboard');
    await clipboard.write(output);
  }
};
