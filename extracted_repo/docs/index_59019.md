# Erste Schritte mit Repomix

Repomix ist ein Tool, das Ihr gesamtes Repository in eine einzige, KI-freundliche Datei verpackt. Es wurde entwickelt, um Ihren Codebase an große Sprachmodelle (LLMs) wie ChatGPT, DeepSeek, Perplexity, Gemini, Gemma, Llama, Grok und mehr zu übergeben.

## Schnellstart

Führen Sie diesen Befehl in Ihrem Projektverzeichnis aus:

```bash
npx repomix
```

Das war's! Sie finden eine `repomix-output.txt` Datei, die Ihr gesamtes Repository in einem KI-freundlichen Format enthält.

Sie können diese Datei dann mit einem Prompt wie diesem an einen KI-Assistenten senden:

```
Diese Datei enthält alle Dateien im Repository in einer Datei zusammengefasst.
Ich möchte den Code refaktorieren, bitte überprüfen Sie ihn zuerst.
```

Die KI wird Ihren gesamten Codebase analysieren und umfassende Einblicke geben:

![Repomix File Usage 1](/images/docs/repomix-file-usage-1.png)

Bei der Diskussion spezifischer Änderungen kann die KI bei der Code-Generierung helfen. Mit Funktionen wie Claudes Artifacts können Sie sogar mehrere voneinander abhängige Dateien erhalten:

![Repomix File Usage 2](/images/docs/repomix-file-usage-2.png)

Viel Spaß beim Programmieren! 🚀

## Kernfunktionen

- **KI-optimierte Ausgabe**: Formatiert Ihren Codebase für einfache KI-Verarbeitung
- **Token-Zählung**: Verfolgt Token-Nutzung für LLM-Kontextgrenzen
- **Git-bewusst**: Berücksichtigt Ihre `.gitignore`-Dateien und `.git/info/exclude`-Dateien
- **Sicherheitsorientiert**: Erkennt sensible Informationen
- **Mehrere Ausgabeformate**: Wählen Sie zwischen Klartext, XML oder Markdown

## Was kommt als Nächstes?

- [Installationsanleitung](installation.md): Verschiedene Möglichkeiten, Repomix zu installieren
- [Verwendungsleitfaden](usage.md): Lernen Sie grundlegende und erweiterte Funktionen kennen
- [Konfiguration](configuration.md): Passen Sie Repomix an Ihre Bedürfnisse an
- [Sicherheitsfunktionen](security.md): Erfahren Sie mehr über Sicherheitsprüfungen

## Community

Treten Sie unserer [Discord-Community](https://discord.gg/wNYzTwZFku) bei für:
- Hilfe mit Repomix
- Teilen Sie Ihre Erfahrungen
- Vorschlagen neuer Funktionen
- Verbindung mit anderen Benutzern

## Unterstützung

Haben Sie einen Fehler gefunden oder brauchen Sie Hilfe?
- [Öffnen Sie ein Issue auf GitHub](https://github.com/yamadashy/repomix/issues)
- Treten Sie unserem Discord-Server bei
- Lesen Sie die [Dokumentation](https://repomix.com)
