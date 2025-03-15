# インストール

## npx を使用する方法 (インストール不要)

```bash
npx repomix
```

## グローバルインストール

### npm
```bash
npm install -g repomix
```

### Yarn
```bash
yarn global add repomix
```

### Homebrew（macOS/Linux）
```bash
brew install repomix
```

## Dockerを使用する方法

以下のコマンドで Docker イメージをプルして実行できます。

```bash
# カレントディレクトリを処理
docker run -v .:/app -it --rm ghcr.io/yamadashy/repomix

# 特定のディレクトリを処理
docker run -v .:/app -it --rm ghcr.io/yamadashy/repomix path/to/directory

# リモートリポジトリを処理
docker run -v ./output:/app -it --rm ghcr.io/yamadashy/repomix --remote yamadashy/repomix
```

## VSCode 拡張機能

VSCodeでRepomixを直接実行できるコミュニティメンテナンスの[Repomix Runner](https://marketplace.visualstudio.com/items?itemName=DorianMassoulier.repomix-runner)拡張機能があります。

機能:
- クリック数回でフォルダをパック
- ファイルまたはコンテンツモードでのコピーが可能
- 出力ファイルの自動クリーンアップ
- repomix.config.jsonと連携

[VSCode マーケットプレイス](https://marketplace.visualstudio.com/items?itemName=DorianMassoulier.repomix-runner)からインストールできます。

## システム要件

- Node.js: 18.0.0 以上
- Git: リモートリポジトリを処理する場合はインストールしてください

## インストールの確認

インストール後、以下のコマンドで Repomix が正常に動作することを確認できます。

```bash
repomix --version
repomix --help
```
