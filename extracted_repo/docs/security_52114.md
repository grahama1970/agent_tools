# Security

## Security Check Feature

Repomix uses [Secretlint](https://github.com/secretlint/secretlint) to detect sensitive information in your files:
- API keys
- Access tokens
- Credentials
- Private keys
- Environment variables

## Configuration

Security checks are enabled by default.

Disable via CLI:
```bash
repomix --no-security-check
```

Or in `repomix.config.json`:
```json
{
  "security": {
    "enableSecurityCheck": false
  }
}
```

## Security Measures

1. **Binary File Exclusion**: Binary files are not included in output
2. **Git-Aware**: Respects `.gitignore` patterns
3. **Automated Detection**: Scans for common security issues:
    - AWS credentials
    - Database connection strings
    - Authentication tokens
    - Private keys

## When Security Check Finds Issues

Example output:
```bash
🔍 Security Check:
──────────────────
2 suspicious file(s) detected and excluded:
1. config/credentials.json
   - Found AWS access key
2. .env.local
   - Found database password
```

## Best Practices

1. Always review output before sharing
2. Use `.repomixignore` for sensitive paths
3. Keep security checks enabled
4. Remove sensitive files from repository

## Reporting Security Issues

Found a security vulnerability? Please:
1. Do not open a public issue
2. Email: koukun0120@gmail.com
3. Or use [GitHub Security Advisories](https://github.com/yamadashy/repomix/security/advisories/new)
