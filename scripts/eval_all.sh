#!/bin/bash
# 运行所有评估的脚本

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=========================================="
echo "GAM Framework - 运行所有评估"
echo "=========================================="

# 检查必要的环境变量
if [ -z "$OPENAI_API_KEY" ]; then
    echo "警告: OPENAI_API_KEY 环境变量未设置"
    echo "如果使用 OpenAI Generator，请先设置: export OPENAI_API_KEY=your_key"
fi

echo ""
echo "请确保已准备好以下数据集："
echo "  - data/hotpotqa.json"
echo "  - narrativeqa (HuggingFace dataset)"
echo "  - data/locomo.json"
echo "  - data/ruler.jsonl"
echo ""

read -p "是否继续? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "已取消"
    exit 0
fi

# 运行 HotpotQA
echo ""
echo "=========================================="
echo "1/4 - 评估 HotpotQA"
echo "=========================================="
bash "$SCRIPT_DIR/eval_hotpotqa.sh" --max-samples 50

# 运行 NarrativeQA
echo ""
echo "=========================================="
echo "2/4 - 评估 NarrativeQA"
echo "=========================================="
bash "$SCRIPT_DIR/eval_narrativeqa.sh" --max-samples 50

# 运行 LoCoMo
echo ""
echo "=========================================="
echo "3/4 - 评估 LoCoMo"
echo "=========================================="
bash "$SCRIPT_DIR/eval_locomo.sh" --max-samples 50

# 运行 RULER
echo ""
echo "=========================================="
echo "4/4 - 评估 RULER"
echo "=========================================="
bash "$SCRIPT_DIR/eval_ruler.sh" --max-samples 50

echo ""
echo "=========================================="
echo "所有评估完成！"
echo "=========================================="
echo "结果已保存到 outputs/ 目录"

