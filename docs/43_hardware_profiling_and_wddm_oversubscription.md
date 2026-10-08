# 硬件性能压测与 WDDM 显存超订机制 (Hardware Profiling & WDDM Over-subscription)

> **核心定律**：在消费级/工作站 GPU (如 RTX 4080 Super 32GB) 上部署多模态大模型，推理吞吐 (Throughput) 并非随 Batch Size 单调递增。必须通过实测寻找吞吐拐点（Sweet Spot），严防 WDDM 显存分页惩罚。

## 1. WDDM 显存超订与“反向劣化”机制
**（论文 Supplementary Material 推荐表述）**
在 Windows WDDM 驱动模式下，当 CUDA 申请的显存逼近物理显存上限时（例如 Qwen2.5-VL-7B 在 bs=16 时占用超 32.3GB），PyTorch 的分配器可能不会直接抛出 OOM 异常，而是触发 WDDM 的**显存超订 (over-subscription)**。此时，部分工作集（Working Set）会被驱动分页到系统物理内存中。当 GPU 运算时，需要通过 PCIe 访问被分页的数据，导致有效显存带宽从物理显存的 700–1000 GB/s 骤降至 PCIe 4.0 x16 的约 25–30 GB/s。
这会引发严重的**性能反向劣化**。例如，Qwen2.5-VL-7B 在 bs=8 时吞吐极佳 (1.50 s/img)，但强行拉高至 bs=16 时，尽管未 OOM，吞吐却劣变为 1.89 s/img (劣化 26%)。**因此，最优 batch size 应由实测吞吐曲线的物理拐点决定，而非单纯取显存允许的最大值。**

## 2. 评测模型矩阵 (Model Matrix) 规划与预期余量
文章涉及庞大的对比矩阵，需要为每个模型精准定位专属的“甜点”并发配置。

### 2.1 4B 级别模型
* **包含模型**：`MedGemma-4B-IT`, `medgemma-1.5-4b-it`
* **规模特征**：单样本 KV 显存开销小（约 0.34 GB/样本）。
* **甜点配置 (实测)**：`--batch-size 64 --max-new-tokens 128`
* **性能指标**：0.50 s/img，显存峰值约 22.3 GB。
* **预期**：对于 1.5 版本，由于参数量相似，可以直接复用 `bs=64` 作为甜点基线。

### 2.2 7B / 8B 级别模型
* **包含模型**：`Qwen2.5-VL-7B-Instruct`, `Qwen3-VL-8B-Instruct`, `InternVL3_5-8B`, `llava-med-v1.5-mistral-7b`, `CheXagent-8b`
* **规模特征**：权重约 14-16 GB (bfloat16)，单样本 KV 开销大（约 1.5 - 2 GB/样本）。
* **甜点配置 (实测/预期)**：`--batch-size 8 --max-new-tokens 128`
* **性能指标**：Qwen2.5-VL 实测 1.50 s/img，显存峰值 15.8 GB。
* **警告**：**严禁使用 bs=16 跑全量**。测试证明 16 会触发 WDDM 分页导致劣化且极易 OOM。同级别的 8B 模型一律以 bs=8 起步压测。

### 2.3 3B 组合视觉模型
* **包含模型**：`CheXagent-2-3b` + `XraySigLIP__vit-l-16-siglip-384__webl`
* **规模特征**：LLM 较小 (3B)，但需额外挂载视觉编码器 SigLIP。
* **预期策略**：总参数约 3B+，显存占用估计介于 4B 和 7B 之间。可以从 `bs=32` 起步探测拐点。

### 2.4 30B 巨兽模型
* **包含模型**：`Qwen3-VL-30B-A3B-Instruct`
* **规模特征**：权重巨大。即使是 4-bit 量化，权重也要吃掉约 15 GB；如果是 bfloat16，纯权重将达到 60GB 直接爆显存。
* **预期策略**：**必须开启 4-bit 量化 (`--load-in-4bit`)**。由于 4-bit 推理的算子瓶颈和 30B 的 KV cache，预计必须使用 `bs=1` 或极小并发，不可盲目扩大 batch，防止系统内存被超订拖死。

## 3. 标准化压测流程 (Benchmarking SOP)
接手新模型时的标准操作：
1. **统一生成上限**：必须固定 `--max-new-tokens 128`，释放无谓的 KV 预留。
2. **保守起步**：根据参数规模推算安全下限（如 7B 用 bs=8，4B 用 bs=32）。
3. **探测劣化拐点**：跑前 100 张图 (`--limit 100`)，观察 `sec_per_img` 和 显存峰值。如果速度不升反降，立刻退回上一档。
4. **稳定执行**：锁定甜点配置后，再去掉 limit 跑全量，避免半路 OOM 沉没成本。

### 2.5 CheXagent-8b 实测更新
* **甜点配置**：在修复了底层 `generate_batch` 后，作为 8B 模型，起步依然推荐 `--batch-size 8`。`bs=8` 串行回退模式下已占用 16.6 GB，开启真并行后预估将吃掉约 20GB+，因此切忌尝试 `bs=16`。


### 2.6 MedGemma-1.5-4B-IT 实测
* **环境**: 兼容原有 .venv-vl (gemma3 架构)
* **甜点配置**: --batch-size 64 (约 0.60 s/img, 显存峰值 22.1 GB)。与初代 MedGemma-4B-IT 的性能特征完全一致，显存余量充足。

### 2.7 Qwen3-VL-8B-Instruct 实测
* **环境**: 需现代版 transformers 5.x (.venv-modern)，且底层架构变更。
* **甜点配置**: --batch-size 8 (约 5.45 s/img, 显存峰值 27.2 GB)。由于参数量增加和架构改动，bs=8 已吃掉 27.2 GB，逼近物理显存上限，切忌尝试 bs=16，否则必将触发 WDDM over-subscription 甚至直接 OOM。

