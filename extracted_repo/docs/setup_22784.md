# 開発環境のセットアップ

## 前提条件

- Node.js ≥ 18.0.0
- Git
- npm

## ローカル開発

```bash
# リポジトリのクローン
git clone https://github.com/yamadashy/repomix.git
cd repomix

# 依存関係のインストール
npm install

# CLIの実行
npm run repomix
```

## Docker開発

```bash
# イメージのビルド
docker build -t repomix .

# コンテナの実行
docker run -v ./:/app -it --rm repomix
```

## プロジェクト構造

```
src/
├── cli/          # CLI実装
├── config/       # 設定の処理
├── core/         # コア機能
└── shared/       # 共有ユーティリティ
```

## テスト

```bash
# テストの実行
npm run test

# テストカバレッジ
npm run test-coverage

# リント
npm run lint-biome
npm run lint-ts
npm run lint-secretlint
```

## リリースプロセス

1. バージョンの更新
```bash
npm version patch  # または minor/major
```

2. テストとビルドの実行
```bash
npm run test-coverage
npm run build
```

3. 公開
```bash
npm publish
```
