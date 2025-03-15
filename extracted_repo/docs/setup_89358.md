# Entwicklungsumgebung einrichten

## Voraussetzungen

- Node.js ≥ 18.0.0
- Git
- npm
- pnpm (empfohlen)

## Lokale Entwicklung

### Repository klonen

```bash
git clone https://github.com/yamadashy/repomix.git
cd repomix
```

### Abhängigkeiten installieren

Mit pnpm (empfohlen):
```bash
pnpm install
```

Mit npm:
```bash
npm install
```

### Entwicklungsserver starten

```bash
# CLI ausführen
pnpm run repomix

# Entwicklungsserver für die Website starten
pnpm run website:dev
```

## Docker-Entwicklung

```bash
# Image erstellen
docker build -t repomix .

# Container ausführen
docker run -v ./:/app -it --rm repomix
```

## Projektstruktur

```text
.
├── src/                # Quellcode
│   ├── cli/           # CLI-Implementierung
│   ├── config/        # Konfigurationshandling
│   ├── core/          # Kernfunktionalität
│   └── shared/        # Gemeinsame Hilfsprogramme
├── tests/             # Testdateien
├── website/           # Dokumentationswebsite
└── package.json       # Projektabhängigkeiten
```

## Tests

```bash
# Alle Tests ausführen
pnpm run test

# Tests mit Abdeckungsbericht
pnpm run test:coverage

# Bestimmte Tests ausführen
pnpm run test -- tests/cli
```

## Code-Qualität

```bash
# Linting ausführen
pnpm run lint-biome
pnpm run lint-ts
pnpm run lint-secretlint

# Linting mit automatischer Korrektur
pnpm run lint:fix

# Typenprüfung
pnpm run typecheck
```

## Dokumentation

Die Dokumentation befindet sich im Verzeichnis `website/`. Um die Dokumentationswebsite lokal zu entwickeln:

```bash
# Entwicklungsserver starten
pnpm run website:dev

# Produktions-Build erstellen
pnpm run website:build
```

## Release-Prozess

1. Version aktualisieren
```bash
pnpm version patch  # oder minor/major
```

2. Tests und Build ausführen
```bash
pnpm run test:coverage
pnpm run build
```

3. Veröffentlichen
```bash
pnpm publish
```

## Fehlerbehebung

### Häufige Probleme

1. **Node.js Version**
   - Stellen Sie sicher, dass Sie Node.js ≥ 18.0.0 verwenden
   - Überprüfen Sie mit `node --version`

2. **Abhängigkeitsprobleme**
   - Löschen Sie `node_modules` und den Lock-File
   - Führen Sie `pnpm install` erneut aus

3. **Build-Fehler**
   - Führen Sie `pnpm run clean` aus
   - Bauen Sie das Projekt neu mit `pnpm run build`

### Support

Bei Problemen:
- Öffnen Sie ein [GitHub Issue](https://github.com/yamadashy/repomix/issues)
- Treten Sie unserem [Discord-Server](https://discord.gg/wNYzTwZFku) bei 
