import type { RepomixConfigMerged } from '../../config/configSchema.js';
import type { ProcessedFile } from '../file/fileTypes.js';

export interface OutputGeneratorContext {
  generationDate: string;
  treeString: string;
  processedFiles: ProcessedFile[];
  config: RepomixConfigMerged;
  instruction: string;
}

export interface RenderContext {
  readonly generationHeader: string;
  readonly summaryPurpose: string;
  readonly summaryFileFormat: string;
  readonly summaryUsageGuidelines: string;
  readonly summaryNotes: string;
  readonly headerText: string | undefined;
  readonly instruction: string;
  readonly treeString: string;
  readonly processedFiles: ReadonlyArray<ProcessedFile>;
  readonly fileSummaryEnabled: boolean;
  readonly directoryStructureEnabled: boolean;
  readonly escapeFileContent: boolean;
  readonly markdownCodeBlockDelimiter: string;
}
