import { type DefaultTheme, defineConfig } from 'vitepress';

export const configPtBr = defineConfig({
  lang: 'pt-BR',
  description: 'Empacote sua base de código em formatos amigáveis para IA',
  themeConfig: {
    nav: [
      { text: 'Guia', link: '/pt-br/guide/' },
      { text: 'Entrar no Discord', link: 'https://discord.gg/wNYzTwZFku' },
    ],
    sidebar: {
      '/pt-br/guide/': [
        {
          text: 'Guia',
          items: [
            { text: 'Começando', link: '/pt-br/guide/' },
            { text: 'Instalação', link: '/pt-br/guide/installation' },
            { text: 'Uso Básico', link: '/pt-br/guide/usage' },
            { text: 'Exemplos de Prompt', link: '/pt-br/guide/prompt-examples' },
            { text: 'Formatos de Saída', link: '/pt-br/guide/output' },
            { text: 'Opções de Linha de Comando', link: '/pt-br/guide/command-line-options' },
            { text: 'Processamento de Repositório Remoto', link: '/pt-br/guide/remote-repository-processing' },
            { text: 'Configuração', link: '/pt-br/guide/configuration' },
            { text: 'Instruções Personalizadas', link: '/pt-br/guide/custom-instructions' },
            { text: 'Remoção de Comentários', link: '/pt-br/guide/comment-removal' },
            { text: 'Compressão de Código', link: '/pt-br/guide/code-compress' },
            { text: 'Segurança', link: '/pt-br/guide/security' },
            { text: 'Servidor MCP', link: '/pt-br/guide/mcp-server' },
            {
              text: 'Dicas e Truques',
              items: [{ text: 'Melhores Práticas', link: '/pt-br/guide/tips/best-practices' }],
            },
            {
              text: 'Desenvolvimento',
              items: [
                { text: 'Contribuindo', link: '/pt-br/guide/development/' },
                { text: 'Configuração', link: '/pt-br/guide/development/setup' },
              ],
            },
          ],
        },
      ],
    },
  },
});

export const configPtBrSearch: DefaultTheme.LocalSearchOptions['locales'] = {
  'pt-BR': {
    translations: {
      button: {
        buttonText: 'Pesquisar',
        buttonAriaLabel: 'Pesquisar',
      },
      modal: {
        noResultsText: 'Nenhum resultado encontrado',
        resetButtonTitle: 'Limpar pesquisa',
        backButtonTitle: 'Voltar',
        displayDetails: 'Mostrar detalhes',
        footer: {
          selectText: 'Selecionar',
          navigateText: 'Navegar',
          closeText: 'Fechar',
        },
      },
    },
  },
};
