# 当前状态

本文件记录当前已经确认过的一次 `Miyabi` 现场信息，方便后续人和 AI 不必从零开始判断。

注意：

- 这是一份历史快照，不是实时状态
- 真正开跑前仍然必须重新执行 `scripts/check_miyabi_readiness.sh`

## 已确认存在的目录

根据最近一次终端快照，当前 `Miyabi` 家目录下至少有：

- `~/webshop_on_miyabi_ai_ready`
- `~/webshop_data`
- `~/tmp`

## 已确认存在的数据文件

根据最近一次终端快照，`~/webshop_data` 下至少有：

- `indexes_1k`
- `items_human_ins.json`
- `items_ins_v2_1000.json`
- `items_ins_v2.json`
- `items_shuffle_1000.json`
- `items_shuffle.json`

## 当前最重要的风险

从这次终端快照里，**没有看到全量 `indexes/` 目录**，只看到了：

- `indexes_1k`

这意味着当前状态更像是：

- **1K smoke 数据基本齐**
- **全量正式训练数据未必齐**

所以在没有进一步确认前，不能直接假设现在已经能跑正式全量训练。

## 需要补充确认的项

下一次登录 `Miyabi` 后，最应该先执行的是：

```bash
ls ~/webshop_data
ls ~/webshop_data/indexes | head
ls ~/webshop_data/indexes_1k | head
```

如果 `ls ~/webshop_data/indexes` 报错，就说明：

- 当前仍然只适合 smoke
- 不适合直接开全量正式训练

## 推荐判断标准

### 只跑 smoke test 需要：

- `items_shuffle_1000.json`
- `items_ins_v2_1000.json`
- `items_human_ins.json`
- `indexes_1k`

### 跑正式全量训练需要：

- `items_shuffle.json`
- `items_ins_v2.json`
- `items_human_ins.json`
- `indexes`

## 当前最合理的下一步

1. 先跑 `scripts/check_miyabi_readiness.sh`
2. 明确判断当前是“仅 smoke-ready”还是“full-train-ready”
3. 如果仅 smoke-ready，就先不要提交正式训练作业
