import type { RepomixConfigMerged } from '../../config/configSchema.js';
import { logger } from '../../shared/logger.js';
import type { RepomixProgressCallback } from '../../shared/types.js';
import type { RawFile } from '../file/fileTypes.js';
import { filterOutUntrustedFiles } from './filterOutUntrustedFiles.js';
import { type SuspiciousFileResult, runSecurityCheck } from './securityCheck.js';

// marks which files are suspicious and which are safe
export const validateFileSafety = async (
  rawFiles: RawFile[],
  progressCallback: RepomixProgressCallback,
  config: RepomixConfigMerged,
  deps = {
    runSecurityCheck,
    filterOutUntrustedFiles,
  },
) => {
  let suspiciousFilesResults: SuspiciousFileResult[] = [];

  if (config.security.enableSecurityCheck) {
    progressCallback('Running security check...');
    suspiciousFilesResults = await deps.runSecurityCheck(rawFiles, progressCallback);
  }

  const safeRawFiles = deps.filterOutUntrustedFiles(rawFiles, suspiciousFilesResults);
  const safeFilePaths = safeRawFiles.map((file) => file.path);
  logger.trace('Safe files count:', safeRawFiles.length);

  return {
    safeRawFiles,
    safeFilePaths,
    suspiciousFilesResults,
  };
};
