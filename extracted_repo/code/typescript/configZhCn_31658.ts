import { type DefaultTheme, defineConfig } from 'vitepress';

export const configZhCn = defineConfig({
  lang: 'zh-CN',
  description: '将代码库打包成AI友好的格式',
  themeConfig: {
    nav: [
      { text: '指南', link: '/zh-cn/guide/' },
      { text: '加入 Discord', link: 'https://discord.gg/wNYzTwZFku' },
    ],
    sidebar: {
      '/zh-cn/guide/': [
        {
          text: '指南',
          items: [
            { text: '开始使用', link: '/zh-cn/guide/' },
            { text: '安装', link: '/zh-cn/guide/installation' },
            { text: '基本用法', link: '/zh-cn/guide/usage' },
            { text: '提示示例', link: '/zh-cn/guide/prompt-examples' },
            { text: '输出格式', link: '/zh-cn/guide/output' },
            { text: '命令行选项', link: '/zh-cn/guide/command-line-options' },
            { text: '远程仓库处理', link: '/zh-cn/guide/remote-repository-processing' },
            { text: '配置', link: '/zh-cn/guide/configuration' },
            { text: '自定义指令', link: '/zh-cn/guide/custom-instructions' },
            { text: '注释移除', link: '/zh-cn/guide/comment-removal' },
            { text: '代码压缩', link: '/zh-cn/guide/code-compress' },
            { text: '安全性', link: '/zh-cn/guide/security' },
            { text: 'MCP 服务器', link: '/zh-cn/guide/mcp-server' },
            {
              text: '技巧与窍门',
              items: [{ text: '最佳实践', link: '/zh-cn/guide/tips/best-practices' }],
            },
            {
              text: '开发',
              items: [
                { text: '贡献指南', link: '/zh-cn/guide/development/' },
                { text: '环境配置', link: '/zh-cn/guide/development/setup' },
              ],
            },
          ],
        },
      ],
    },
  },
});

export const configZhCnSearch: DefaultTheme.LocalSearchOptions['locales'] = {
  'zh-CN': {
    translations: {
      button: {
        buttonText: '搜索',
        buttonAriaLabel: '搜索',
      },
      modal: {
        noResultsText: '未找到相关结果',
        resetButtonTitle: '清除搜索',
        backButtonTitle: '返回',
        displayDetails: '显示详情',
        footer: {
          selectText: '选择',
          navigateText: '切换',
          closeText: '关闭',
        },
      },
    },
  },
};
