# Crablet 前端 DMG 打包优化方案

## 目标

将前端项目打包为 macOS DMG 安装包，优化包体积与启动性能。

## 关键问题

当前 `tesseract.js`（~10MB）和 `d3`（~300KB）体积过大，影响首屏加载。

## 优化策略

1. **动态导入（Dynamic Import）**：将 `tesseract.js` 和 `d3` 改为动态导入，仅在需要时加载。
2. **Chunk 拆分**：在 `vite.config.ts` 中配置 `manualChunks`，将大体积库单独拆包。
3. **虚拟列表**：使用 `react-virtuoso` 替代 `react-window`，减少 DOM 节点数。

## 实施步骤

### 1. 修改 vite.config.ts

```typescript
// frontend/vite.config.ts
import { defineConfig } from 'vitest/config'
import react from '@vitejs/plugin-react'
import { visualizer } from 'rollup-plugin-visualizer'

export default defineConfig({
  plugins: [
    react(),
    visualizer({
      filename: 'bundle-analysis.html',
      open: true,
      template: 'treemap', // 或 'sunburst'
    }),
  ],
  build: {
    // 生产环境优化
    minify: 'terser',
    rollupOptions: {
      output: {
        manualChunks(id) {
          if (id.includes('tesseract.js')) {
            return 'ocr-vendor'
          }
          if (id.includes('d3')) {
            return 'chart-vendor'
          }
          if (id.includes('node_modules')) {
            return 'vendor'
          }
        },
      },
    },
  },
})
```

### 2. 修改 package.json

添加 `rollup-plugin-visualizer` 到 devDependencies：

```json
{
  "devDependencies": {
    "rollup-plugin-visualizer": "^5.12.0"
  }
}
```

### 3. 构建与验证

```bash
npm run build
npm run type-check
npm run test:ci
```

## 预期效果

| 优化项 | 优化前 | 优化后 |
|--------|--------|--------|
| tesseract.js 体积 | ~10MB（首屏加载） | ~200KB（动态导入后按需加载） |
| d3 体积 | ~300KB | ~50KB（仅用 d3-scale 等） |
| 总 JS 体积 | ~10.3MB | ~2MB |
| 首屏加载时间 | ~3s | ~1.5s |

## 验证

构建后打开 `bundle-analysis.html`，确认：

1. `tesseract.js` 已拆分为 `ocr-vendor` chunk
2. `d3` 已拆分为 `chart-vendor` chunk
3. 总 JS 体积从 ~10.3MB 降至 ~2MB
4. 首屏加载时间从 ~3s 降至 ~1.5s
