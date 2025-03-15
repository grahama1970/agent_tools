import type { Ace } from 'ace-builds';
import type { PackResult } from '../api/client';
import { analyticsUtils } from './analytics';

/**
 * Format timestamp to locale string
 */
export function formatTimestamp(timestamp: string): string {
  return new Date(timestamp).toLocaleString();
}

/**
 * Handle clipboard copy with analytics tracking
 */
export async function copyToClipboard(content: string, format: string): Promise<boolean> {
  try {
    await navigator.clipboard.writeText(content);
    analyticsUtils.trackCopyOutput(format);
    return true;
  } catch (err) {
    console.error('Failed to copy:', err);
    return false;
  }
}

/**
 * Convert repository name to format suitable for filename
 */
function formatRepositoryName(repository: string): string {
  // Extract owner and repo from GitHub URL format or use as is
  const match = repository.match(/(?:https:\/\/github\.com\/)?([^/]+)\/([^/]+)(?:\.git)?$/);
  if (match) {
    const [, owner, repo] = match;
    return `${owner}-${repo}`;
  }
  // For non-GitHub repositories or local files, clean up the name
  return repository.replace(/[\/\\]/g, '-').replace(/\.git$/, '');
}

/**
 * Handle file download with analytics tracking
 */
export function downloadResult(content: string, format: string, result: PackResult): void {
  const blob = new Blob([content], { type: 'text/plain' });
  const url = window.URL.createObjectURL(blob);
  const a = document.createElement('a');
  const extension = format === 'markdown' ? 'md' : format === 'xml' ? 'xml' : 'txt';

  const repoName = formatRepositoryName(result.metadata.repository);
  a.href = url;
  a.download = `repomix-output-${repoName}.${extension}`;
  document.body.appendChild(a);
  a.click();

  analyticsUtils.trackDownloadOutput(format);
  window.URL.revokeObjectURL(url);
  document.body.removeChild(a);
}

/**
 * Get Ace editor options
 */
export function getEditorOptions(): Partial<Ace.EditorOptions> {
  return {
    readOnly: true,
    wrap: false,
    showPrintMargin: false,
    fontSize: '13px',
    useWorker: false,
    highlightActiveLine: false,
  };
}
