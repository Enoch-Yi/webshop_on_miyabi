# WebShop Assets

这个目录只保留仓库结构，不直接提交大体积数据。

`Miyabi` 上推荐两种使用方式：

1. 直接使用外部数据目录：
   - `WEBSHOP_DATA_DIR=~/webshop_data`
   - `WEBSHOP_SEARCH_DIR=~/webshop_data`
2. 用 `scripts/setup_webshop_assets.py` 把外部数据软链接到仓库内

最少 smoke 需要：

- `items_shuffle_1000.json`
- `items_ins_v2_1000.json`
- `items_human_ins.json`
- `indexes_1k`

全量正式训练还需要：

- `items_shuffle.json`
- `items_ins_v2.json`
- `indexes`
