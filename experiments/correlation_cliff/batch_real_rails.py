# experiments/correlation_cliff/batch_real_rails.sh
#!/bin/bash
set -e

RAILS=("openai" "perspective" "llamaguard")
DATASETS=("realtoxicityprompts" "hh-rlhf")

for DATASET in "${DATASETS[@]}"; do
  for RAIL_A in "${RAILS[@]}"; do
    for RAIL_B in "${RAILS[@]}"; do
      if [ "$RAIL_A" != "$RAIL_B" ]; then
        echo "Running: $RAIL_A vs $RAIL_B on $DATASET"
        python run_real_guardrails.py \
          --rail_A $RAIL_A \
          --rail_B $RAIL_B \
          --dataset $DATASET \
          --out_dir artifacts/real_rails/${DATASET}/${RAIL_A}_vs_${RAIL_B}
      fi
    done
  done
done