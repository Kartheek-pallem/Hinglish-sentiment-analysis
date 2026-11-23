#!/usr/bin/env bash
# Runs the whole project. About 10-12 minutes on a free Colab T4 GPU.
#   bash run_all.sh          everything
#   bash run_all.sh --quick  skip the transformer (no GPU needed, ~6 min on CPU)
set -e
cd "$(dirname "$0")"

echo "== 1/6 data";            python src/data.py
echo "== 2/6 classical";       python src/classical.py
echo "== 3/6 base-paper check"; python src/vader_study.py
echo "== 4/6 deep models";     python src/train_deep.py
if [ "$1" != "--quick" ]; then
  echo "== 5/6 HingBERT";      python src/train_transformer.py
else
  echo "== 5/6 HingBERT skipped (--quick)"
fi
echo "== 6/6 evaluation";      python src/evaluate.py
echo; echo "Done. Open results/SUMMARY.md and results/figures/"
