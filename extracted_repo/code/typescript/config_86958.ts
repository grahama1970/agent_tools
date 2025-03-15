import { defineConfig } from 'vitepress';
import { configDe } from './config/configDe';
import { configEnUs } from './config/configEnUs';
import { configEs } from './config/configEs';
import { configJa } from './config/configJa';
import { configKo } from './config/configKo';
import { configPtBr } from './config/configPtBr';
import { configShard } from './config/configShard';
import { configZhCn } from './config/configZhCn';

export default defineConfig({
  ...configShard,
  locales: {
    root: { label: 'English', ...configEnUs },
    'zh-cn': { label: '简体中文', ...configZhCn },
    ja: { label: '日本語', ...configJa },
    es: { label: 'Español', ...configEs },
    'pt-br': { label: 'Português', ...configPtBr },
    ko: { label: '한국어', ...configKo },
    de: { label: 'Deutsch', ...configDe },
  },
});
