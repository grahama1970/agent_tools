import type { RawFile } from '../file/fileTypes.js';
import type { SuspiciousFileResult } from './securityCheck.js';

export const filterOutUntrustedFiles = (
  rawFiles: RawFile[],
  suspiciousFilesResults: SuspiciousFileResult[],
): RawFile[] =>
  rawFiles.filter((rawFile) => !suspiciousFilesResults.some((result) => result.filePath === rawFile.path));
