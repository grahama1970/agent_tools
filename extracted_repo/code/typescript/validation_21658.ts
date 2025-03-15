/**
 * Validates a GitHub repository URL or shorthand format
 * TODO: Share this validation logic with repomix core (src/cli/actions/remoteAction.ts)
 */
export function isValidRemoteValue(remoteValue: string): boolean {
  // Check the short form of the GitHub URL. e.g. yamadashy/repomix
  const namePattern = '[a-zA-Z0-9](?:[a-zA-Z0-9._-]*[a-zA-Z0-9])?';
  const shortFormRegex = new RegExp(`^${namePattern}/${namePattern}$`);
  if (shortFormRegex.test(remoteValue)) {
    return true;
  }

  // Check the direct form of the GitHub URL. e.g.  https://github.com/yamadashy/repomix or https://gist.github.com/yamadashy/1234567890abcdef
  try {
    new URL(remoteValue);
    return true;
  } catch (error) {
    return false;
  }
}
