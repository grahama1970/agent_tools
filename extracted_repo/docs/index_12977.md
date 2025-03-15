---
layout: home
title: Repomix
titleTemplate: Packen Sie Ihren Codebase in KI-freundliche Formate
aside: false
editLink: false

features:
  - icon: 🤖
    title: KI-Optimiert
    details: Formatiert Ihren Codebase so, dass er für KI leicht zu verstehen und zu verarbeiten ist.

  - icon: ⚙️
    title: Git-Bewusst
    details: Berücksichtigt automatisch Ihre .gitignore-Dateien.

  - icon: 🛡️
    title: Sicherheitsorientiert
    details: Integriert Secretlint für robuste Sicherheitsprüfungen zur Erkennung und Verhinderung der Aufnahme sensibler Informationen.

  - icon: 📊
    title: Token-Zählung
    details: Bietet Token-Zählungen für jede Datei und das gesamte Repository, nützlich für LLM-Kontextgrenzen.

---

<div class="cli-section">

## Schnellstart

Sobald Sie mit Repomix eine gepackte Datei (`repomix-output.txt`) erstellt haben, können Sie diese mit einer Aufforderung wie dieser an einen KI-Assistenten senden:

```
Diese Datei enthält alle Dateien im Repository in einer Datei zusammengefasst.
Ich möchte den Code refaktorieren, bitte überprüfen Sie ihn zuerst.
```

Die KI wird Ihren gesamten Codebase analysieren und umfassende Einblicke geben:

![Repomix File Usage 1](/images/docs/repomix-file-usage-1.png)

Bei der Diskussion spezifischer Änderungen kann die KI bei der Code-Generierung helfen. Mit Funktionen wie Claudes Artifacts können Sie sogar mehrere voneinander abhängige Dateien erhalten:

![Repomix File Usage 2](/images/docs/repomix-file-usage-2.png)

Viel Spaß beim Programmieren! 🚀

## Leitfaden für fortgeschrittene Benutzer

Für fortgeschrittene Benutzer, die mehr Kontrolle benötigen, bietet Repomix umfangreiche Anpassungsmöglichkeiten über seine CLI-Schnittstelle.

### Schnellstart

Sie können Repomix sofort in Ihrem Projektverzeichnis ohne Installation ausprobieren:

```bash
npx repomix
```

Oder installieren Sie es global für wiederholte Verwendung:

```bash
# Installation mit npm
npm install -g repomix

# Alternativ mit yarn
yarn global add repomix

# Alternativ mit Homebrew (macOS/Linux)
brew install repomix

# Dann in einem beliebigen Projektverzeichnis ausführen
repomix
```

Das war's! Repomix generiert eine `repomix-output.txt` Datei in Ihrem aktuellen Verzeichnis, die Ihr gesamtes Repository in einem KI-freundlichen Format enthält.

### Verwendung

Um Ihr gesamtes Repository zu packen:

```bash
repomix
```

Um ein bestimmtes Verzeichnis zu packen:

```bash
repomix path/to/directory
```

Um bestimmte Dateien oder Verzeichnisse mit [Glob-Mustern](https://github.com/mrmlnc/fast-glob?tab=readme-ov-file#pattern-syntax) zu packen:

```bash
repomix --include "src/**/*.ts,**/*.md"
```

Um bestimmte Dateien oder Verzeichnisse auszuschließen:

```bash
repomix --ignore "**/*.log,tmp/"
```

Um ein Remote-Repository zu packen:
```bash
# Kurzform verwenden
npx repomix --remote yamadashy/repomix

# Vollständige URL verwenden (unterstützt Branches und spezifische Pfade)
npx repomix --remote https://github.com/yamadashy/repomix
npx repomix --remote https://github.com/yamadashy/repomix/tree/main

# Commit-URL verwenden
npx repomix --remote https://github.com/yamadashy/repomix/commit/836abcd7335137228ad77feb28655d85712680f1
```

Um eine neue Konfigurationsdatei (`repomix.config.json`) zu initialisieren:

```bash
repomix --init
```

Sobald Sie die gepackte Datei erstellt haben, können Sie sie mit generativen KI-Tools wie Claude, ChatGPT und Gemini verwenden.

#### Docker-Verwendung

Sie können Repomix auch mit Docker ausführen 🐳  
Dies ist nützlich, wenn Sie Repomix in einer isolierten Umgebung ausführen oder Container bevorzugen.

Grundlegende Verwendung (aktuelles Verzeichnis):

```bash
docker run -v .:/app -it --rm ghcr.io/yamadashy/repomix
```

Um ein bestimmtes Verzeichnis zu packen:
```bash
docker run -v .:/app -it --rm ghcr.io/yamadashy/repomix path/to/directory
```

Ein Remote-Repository verarbeiten und in ein `output`-Verzeichnis ausgeben:

```bash
docker run -v ./output:/app -it --rm ghcr.io/yamadashy/repomix --remote https://github.com/yamadashy/repomix
```

### Ausgabeformate

Wählen Sie Ihr bevorzugtes Ausgabeformat:

```bash
# XML-Format (Standard)
repomix --style xml

# Markdown-Format
repomix --style markdown

# Klartext-Format
repomix --style plain
```

### Anpassung

Erstellen Sie eine `repomix.config.json` für dauerhafte Einstellungen:

```json
{
  "output": {
    "style": "markdown",
    "filePath": "custom-output.md",
    "removeComments": true,
    "showLineNumbers": true,
    "topFilesLength": 10
  },
  "ignore": {
    "customPatterns": ["*.test.ts", "docs/**"]
  }
}
```

### Weitere Beispiele
::: tip
💡 Besuchen Sie unser [GitHub-Repository](https://github.com/yamadashy/repomix) für vollständige Dokumentation und weitere Beispiele!
:::

</div> 
