# 开发环境搭建

## 前提条件

- Node.js ≥ 18.0.0
- Git
- npm
- pnpm（推荐）

## 本地开发

### 克隆仓库

```bash
git clone https://github.com/yamadashy/repomix.git
cd repomix
```

### 安装依赖

使用 pnpm（推荐）：
```bash
pnpm install
```

使用 npm：
```bash
npm install
```

### 启动开发服务器

```bash
# 运行 CLI
pnpm run repomix

# 启动文档网站开发服务器
pnpm run website:dev
```

## Docker 开发环境

```bash
# 构建镜像
docker build -t repomix .

# 运行容器
docker run -v ./:/app -it --rm repomix
```

## 项目结构

```text
.
├── src/                # 源代码
│   ├── cli/           # CLI 实现
│   ├── config/        # 配置处理
│   ├── core/          # 核心功能
│   └── shared/        # 共享工具
├── tests/             # 测试文件
├── website/           # 文档网站
└── package.json       # 项目依赖
```

## 测试

```bash
# 运行所有测试
pnpm run test

# 生成测试覆盖率报告
pnpm run test:coverage

# 运行特定测试
pnpm run test -- tests/cli
```

## 代码质量

```bash
# 运行代码检查
pnpm run lint

# 自动修复代码问题
pnpm run lint:fix

# 类型检查
pnpm run typecheck
```

## 文档

文档位于 `website/` 目录。要在本地开发文档网站：

```bash
# 启动开发服务器
pnpm run website:dev

# 构建生产版本
pnpm run website:build
```

## 故障排除

### 常见问题

1. **Node.js 版本**
   - 确保使用 Node.js ≥ 18.0.0
   - 使用 `node --version` 检查版本

2. **依赖问题**
   - 删除 `node_modules` 和锁定文件
   - 重新运行 `pnpm install`

3. **构建错误**
   - 运行 `pnpm run clean`
   - 重新构建项目 `pnpm run build`

### 支持

如果遇到问题：
- 创建 [GitHub Issue](https://github.com/yamadashy/repomix/issues)
- 加入我们的 [Discord 服务器](https://discord.gg/wNYzTwZFku)
