# RUN_ORDER — 先跑哪个，再跑哪个

> 照这个顺序敲命令即可。全程在 `~/Desktop/SCOPE/` 下，离线、不联网。
> 三步串成一条链：**decode → estimate → evaluate**。中间产物都落在 `out/`。

---

## 0. 一次性准备

```bash
cd ~/Desktop/SCOPE
pip install -r requirements.txt
```

（可选）先自检数据能不能读：

```bash
python3 data.py --n 3
```

看到打印出句子 + gold 实体即正常。

---

## 第 1 步 — decode.py（唯一较慢的一步，要模型）

对每句采样解码 K 次（GPU 上一次前向出 K 个样本）。**设备自动选择：CUDA(4070) > MPS > CPU。**

**先用小配置冒烟跑通**：

```bash
python3 decode.py --n 12 --K 6
```

### 在你的 4070 上（推荐配置）

4070 比 Mac 快很多，可以上更强的模型 + 更大规模，这也能压住"解码噪声制造假 singleton"的问题：

```bash
# 3B（fp16，直接上卡，~7GB 显存，稳）
export SCOPE_MODEL=Qwen/Qwen2.5-3B-Instruct
python3 decode.py --real --n 200 --K 12

# 7B（12GB 4070 需 4-bit，需 pip install bitsandbytes）
export SCOPE_MODEL=Qwen/Qwen2.5-7B-Instruct
python3 decode.py --real --n 200 --K 12 --load_4bit
```

- 产出：`out/decodes.json`
- `--real`：存在 OntoSpan gold 时用真数据；无则回退自带样本。
- `--load_4bit`：仅 4070 跑 7B 时开。

---

## 第 2 步 — estimate.py（秒级，纯 Python，无需模型）

从 K 个解码算 Good-Turing 缺失质量 + 自覆盖 + 认证/弃权：

```bash
python3 estimate.py
```

- 读：`out/decodes.json` → 产出：`out/estimates.json`
- 屏幕会打印：certified / abstain 各多少、certified 的平均自覆盖。

---

## 第 3 步 — evaluate.py（秒级，纯 Python，出最终结论）

用 gold 验证 + 主动标注分诊：

```bash
python3 evaluate.py
```

- 读：`out/estimates.json` → 产出：`out/results.json`
- 屏幕会打印三块：
  - **A** 自覆盖 vs 真召回的差距（诚实边界）
  - **B** 弃权是否合理
  - **C** 主动标注分诊：复核前 30% 高风险句捞回多少真实漏抽 vs 随机

---

## 一键跑完（三步连跑）

```bash
cd ~/Desktop/SCOPE && python3 decode.py --n 12 --K 6 && python3 estimate.py && python3 evaluate.py
```

---

## 常见问题

- **改句子数/解码次数**：只在第 1 步 `--n`（句子数）、`--K`（每句解码次数）调。改完必须**重跑第 1→2→3 步**（下游读上游产物）。
- **想用真实大数据**：`decode.py --real`，其余不变。默认真数据路径在 `config.REAL_GOLD`，也可 `export SCOPE_GOLD=/路径/xx.json` 覆盖。
- **换模型**：`export SCOPE_MODEL=Qwen/Qwen2.5-7B-Instruct`（更强、更慢，MPS 上吃内存）。
- **只想重算不重解码**：第 1 步产物在 `out/decodes.json`，改了 estimate/evaluate 逻辑只需重跑第 2、3 步，**不用**重跑第 1 步。
- **弃权很多**：说明该模型解码多样性低（塌缩），属预期；上更强模型或增大 `--K`。
