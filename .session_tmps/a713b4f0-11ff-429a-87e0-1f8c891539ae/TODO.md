# SparkLM 优化 TODO

## 已完成
- [x] 全量测试回归验证（616 passed, 1 pre-existing, 1 skipped）
- [x] VLM Forward 优化：提取采样逻辑为独立函数
- [x] VLM Generate 优化：简化 RoPE 缓存获取
- [x] Labels 处理优化：统一 combined_mask/vision_labels 构建
- [x] pyproject.toml + 安装验证 + README 更新

## 进行中
- [ ] ruff/mypy 门禁 + wheel 构建
- [ ] CI 集成
