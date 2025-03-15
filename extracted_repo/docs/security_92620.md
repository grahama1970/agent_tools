# Sicherheit

## Sicherheitsprüfungsfunktion

Repomix verwendet [Secretlint](https://github.com/secretlint/secretlint) zur Erkennung sensibler Informationen in Ihren Dateien:
- API-Schlüssel
- Zugangstoken
- Anmeldedaten
- Private Schlüssel
- Umgebungsvariablen

## Konfiguration

Sicherheitsprüfungen sind standardmäßig aktiviert.

Deaktivierung über CLI:
```bash
repomix --no-security-check
```

Oder in `repomix.config.json`:
```json
{
  "security": {
    "enableSecurityCheck": false
  }
}
```

## Sicherheitsmaßnahmen

1. **Ausschluss von Binärdateien**: Binärdateien werden nicht in die Ausgabe aufgenommen
2. **Git-bewusst**: Berücksichtigt `.gitignore`-Muster
3. **Automatische Erkennung**: Sucht nach häufigen Sicherheitsproblemen:
    - AWS-Anmeldedaten
    - Datenbankverbindungszeichenfolgen
    - Authentifizierungstoken
    - Private Schlüssel

## Wenn die Sicherheitsprüfung Probleme findet

Beispielausgabe:
```bash
🔍 Sicherheitsprüfung:
──────────────────
2 verdächtige Datei(en) erkannt und ausgeschlossen:
1. config/credentials.json
   - AWS-Zugriffsschlüssel gefunden
2. .env.local
   - Datenbank-Passwort gefunden
```

## Best Practices

1. Überprüfen Sie die Ausgabe immer vor dem Teilen
2. Verwenden Sie `.repomixignore` für sensible Pfade
3. Lassen Sie Sicherheitsprüfungen aktiviert
4. Entfernen Sie sensible Dateien aus dem Repository

## Melden von Sicherheitsproblemen

Haben Sie eine Sicherheitslücke gefunden? Bitte:
1. Öffnen Sie kein öffentliches Issue
2. E-Mail: koukun0120@gmail.com
3. Oder nutzen Sie [GitHub Security Advisories](https://github.com/yamadashy/repomix/security/advisories/new) 
