#!/usr/bin/env bash
# ==============================================
# 用法：bash run_all.sh
# 依赖：已激活的 Python 环境、主脚本 main.py
# ==============================================

# 全局可调参数 ------------------------------------------------
PY=python38            # 或 python3
MAIN=run_multi_gan.py         # 你的入口文件
EPOCHS=10000
BATCH=64
DEVICES="[0]"        # 多卡如 "[0,1]"
NOTES="批量训练 8 个品种的处理后的数据"
# -------------------------------------------------------------

# 品种 -> 对应 CSV 文件名（可自行修改）
declare -A SYMBOLS=(
  ["300股指"]="300股指"
  ["大豆"]="大豆"
  ["螺纹钢"]="螺纹钢"
  ["黄金"]="黄金"
  ["铜"]="铜"
  ["原油"]="原油"
  ["纸浆"]="纸浆"
  ["白糖"]="白糖"
)

GEN_LIST=(gru lstm transformer)

for NAME in "${!SYMBOLS[@]}"; do
  CSV="database/process_${SYMBOLS[$NAME]}.csv"
  OUT_BASE="out_put/multi/${NAME}"
  CKPT_BASE="ckpt/${NAME}"

  for GEN in "${GEN_LIST[@]}"; do
    OUT="${OUT_BASE}/${GEN}"
    CKPT="${CKPT_BASE}/${GEN}"
    echo "===== 训练 $NAME | 生成器: $GEN ====="
    $PY "$MAIN" \
        --notes "$NOTES" \
        --data_path "$CSV" \
        --output_dir "$OUT" \
        --ckpt_dir "$CKPT" \
        --num_epochs "$EPOCHS" \
        --batch_size "$BATCH" \
        --device "$DEVICES" \
        --distill False \
        --cross_finetune False \
        -n 1 \
        -gens "$GEN" \
        --mode train
  done
done
