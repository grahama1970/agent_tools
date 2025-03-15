import { type DefaultTheme, defineConfig } from 'vitepress';

export const configKo = defineConfig({
  lang: 'ko',
  description: '코드베이스를 AI 친화적인 형식으로 패키징',
  themeConfig: {
    nav: [
      { text: '가이드', link: '/ko/guide/' },
      { text: 'Discord 참여', link: 'https://discord.gg/wNYzTwZFku' },
    ],
    sidebar: {
      '/ko/guide/': [
        {
          text: '가이드',
          items: [
            { text: '시작하기', link: '/ko/guide/' },
            { text: '설치', link: '/ko/guide/installation' },
            { text: '기본 사용법', link: '/ko/guide/usage' },
            { text: '프롬프트 예제', link: '/ko/guide/prompt-examples' },
            { text: '출력 형식', link: '/ko/guide/output' },
            { text: '명령줄 옵션', link: '/ko/guide/command-line-options' },
            { text: '원격 저장소 처리', link: '/ko/guide/remote-repository-processing' },
            { text: '설정', link: '/ko/guide/configuration' },
            { text: '사용자 정의 지침', link: '/ko/guide/custom-instructions' },
            { text: '주석 제거', link: '/ko/guide/comment-removal' },
            { text: '코드 압축', link: '/ko/guide/code-compress' },
            { text: '보안', link: '/ko/guide/security' },
            { text: 'MCP 서버', link: '/ko/guide/mcp-server' },
            {
              text: '팁과 요령',
              items: [{ text: '모범 사례', link: '/ko/guide/tips/best-practices' }],
            },
            {
              text: '개발',
              items: [
                { text: '기여하기', link: '/ko/guide/development/' },
                { text: '환경 설정', link: '/ko/guide/development/setup' },
              ],
            },
          ],
        },
      ],
    },
  },
});

export const configKoSearch: DefaultTheme.LocalSearchOptions['locales'] = {
  ko: {
    translations: {
      button: {
        buttonText: '검색',
        buttonAriaLabel: '검색',
      },
      modal: {
        noResultsText: '검색 결과가 없습니다',
        resetButtonTitle: '검색 초기화',
        backButtonTitle: '뒤로',
        displayDetails: '상세 정보 표시',
        footer: {
          selectText: '선택',
          navigateText: '이동',
          closeText: '닫기',
        },
      },
    },
  },
};
