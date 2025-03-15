import { type DefaultTheme, defineConfig } from 'vitepress';

export const configJa = defineConfig({
  lang: 'ja',
  description: 'コードベースをAIフレンドリーな形式にパッケージング',
  themeConfig: {
    nav: [
      // guide
      { text: '使い方', link: '/ja/guide/' },
      { text: 'Discordに参加', link: 'https://discord.gg/wNYzTwZFku' },
    ],
    sidebar: {
      '/ja/guide/': [
        {
          text: '使い方',
          items: [
            { text: 'はじめに', link: '/ja/guide/' },
            { text: 'インストール', link: '/ja/guide/installation' },
            { text: '基本的な使い方', link: '/ja/guide/usage' },
            { text: 'プロンプト例', link: '/ja/guide/prompt-examples' },
            { text: '出力フォーマット', link: '/ja/guide/output' },
            { text: 'コマンドラインオプション', link: '/ja/guide/command-line-options' },
            { text: 'リモートリポジトリの処理', link: '/ja/guide/remote-repository-processing' },
            { text: '設定', link: '/ja/guide/configuration' },
            { text: 'カスタム指示', link: '/ja/guide/custom-instructions' },
            { text: 'コメントの削除', link: '/ja/guide/comment-removal' },
            { text: 'コード圧縮', link: '/ja/guide/code-compress' },
            { text: 'セキュリティ', link: '/ja/guide/security' },
            { text: 'MCPサーバー', link: '/ja/guide/mcp-server' },
            {
              text: 'ヒント＆テクニック',
              items: [{ text: 'ベストプラクティス', link: '/ja/guide/tips/best-practices' }],
            },
            {
              text: '開発',
              items: [
                { text: '開発への貢献', link: '/ja/guide/development/' },
                { text: '環境構築', link: '/ja/guide/development/setup' },
              ],
            },
          ],
        },
      ],
    },
  },
});

export const configJaSearch: DefaultTheme.LocalSearchOptions['locales'] = {
  ja: {
    translations: {
      button: {
        buttonText: '検索',
        buttonAriaLabel: '検索',
      },
      modal: {
        noResultsText: '検索結果がありません',
        resetButtonTitle: '検索をリセット',
        backButtonTitle: '前に戻る',
        displayDetails: '詳細を表示',
        footer: {
          selectText: '選択',
          navigateText: '移動',
          closeText: '閉じる',
        },
      },
    },
  },
};
