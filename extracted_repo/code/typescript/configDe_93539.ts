import { type DefaultTheme, defineConfig } from 'vitepress';

export const configDe = defineConfig({
  lang: 'de',
  description: 'Paketieren Sie Ihre Codebasis in KI-freundliche Formate',
  themeConfig: {
    nav: [
      { text: 'Anleitung', link: '/de/guide/' },
      { text: 'Discord beitreten', link: 'https://discord.gg/wNYzTwZFku' },
    ],
    sidebar: {
      '/de/guide/': [
        {
          text: 'Anleitung',
          items: [
            { text: 'Erste Schritte', link: '/de/guide/' },
            { text: 'Installation', link: '/de/guide/installation' },
            { text: 'Grundlegende Verwendung', link: '/de/guide/usage' },
            { text: 'Prompt-Beispiele', link: '/de/guide/prompt-examples' },
            { text: 'Ausgabeformate', link: '/de/guide/output' },
            { text: 'Kommandozeilenoptionen', link: '/de/guide/command-line-options' },
            { text: 'Remote-Repository-Verarbeitung', link: '/de/guide/remote-repository-processing' },
            { text: 'Konfiguration', link: '/de/guide/configuration' },
            { text: 'Benutzerdefinierte Anweisungen', link: '/de/guide/custom-instructions' },
            { text: 'Kommentare entfernen', link: '/de/guide/comment-removal' },
            { text: 'Code-Komprimierung', link: '/de/guide/code-compress' },
            { text: 'Sicherheit', link: '/de/guide/security' },
            { text: 'MCP-Server', link: '/de/guide/mcp-server' },
            {
              text: 'Tipps & Tricks',
              items: [{ text: 'Best Practices', link: '/de/guide/tips/best-practices' }],
            },
            {
              text: 'Entwicklung',
              items: [
                { text: 'Mitwirken', link: '/de/guide/development/' },
                { text: 'Einrichtung', link: '/de/guide/development/setup' },
              ],
            },
          ],
        },
      ],
    },
  },
});

export const configDeSearch: DefaultTheme.LocalSearchOptions['locales'] = {
  de: {
    translations: {
      button: {
        buttonText: 'Suchen',
        buttonAriaLabel: 'Suchen',
      },
      modal: {
        noResultsText: 'Keine Ergebnisse gefunden',
        resetButtonTitle: 'Suche zurücksetzen',
        backButtonTitle: 'Zurück',
        displayDetails: 'Details anzeigen',
        footer: {
          selectText: 'Auswählen',
          navigateText: 'Navigieren',
          closeText: 'Schließen',
        },
      },
    },
  },
};
