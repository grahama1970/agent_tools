import { type DefaultTheme, defineConfig } from 'vitepress';

export const configEs = defineConfig({
  lang: 'es',
  description: 'Empaqueta tu código en formatos amigables para IA',
  themeConfig: {
    nav: [
      { text: 'Guía', link: '/es/guide/' },
      { text: 'Unirse a Discord', link: 'https://discord.gg/wNYzTwZFku' },
    ],
    sidebar: {
      '/es/guide/': [
        {
          text: 'Guía',
          items: [
            { text: 'Comenzar', link: '/es/guide/' },
            { text: 'Instalación', link: '/es/guide/installation' },
            { text: 'Uso Básico', link: '/es/guide/usage' },
            { text: 'Ejemplos de Prompts', link: '/es/guide/prompt-examples' },
            { text: 'Formatos de Salida', link: '/es/guide/output' },
            { text: 'Opciones de Línea de Comandos', link: '/es/guide/command-line-options' },
            { text: 'Procesamiento de Repositorios Remotos', link: '/es/guide/remote-repository-processing' },
            { text: 'Configuración', link: '/es/guide/configuration' },
            { text: 'Instrucciones Personalizadas', link: '/es/guide/custom-instructions' },
            { text: 'Eliminación de Comentarios', link: '/es/guide/comment-removal' },
            { text: 'Compresión de Código', link: '/es/guide/code-compress' },
            { text: 'Seguridad', link: '/es/guide/security' },
            { text: 'Servidor MCP', link: '/es/guide/mcp-server' },
            {
              text: 'Consejos y Trucos',
              items: [{ text: 'Mejores Prácticas', link: '/es/guide/tips/best-practices' }],
            },
            {
              text: 'Desarrollo',
              items: [
                { text: 'Contribuir', link: '/es/guide/development/' },
                { text: 'Configuración', link: '/es/guide/development/setup' },
              ],
            },
          ],
        },
      ],
    },
  },
});

export const configEsSearch: DefaultTheme.LocalSearchOptions['locales'] = {
  es: {
    translations: {
      button: {
        buttonText: 'Buscar',
        buttonAriaLabel: 'Buscar',
      },
      modal: {
        noResultsText: 'No se encontraron resultados',
        resetButtonTitle: 'Reiniciar búsqueda',
        backButtonTitle: 'Atrás',
        displayDetails: 'Mostrar detalles',
        footer: {
          selectText: 'Seleccionar',
          navigateText: 'Navegar',
          closeText: 'Cerrar',
        },
      },
    },
  },
};
