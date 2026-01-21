#!/bin/bash
set -e

echo "=========================================="
echo "Primary Care AI Futures - Full Pipeline"
echo "=========================================="

# Create output directories
mkdir -p ../results/figures
mkdir -p ../results/tables
mkdir -p ../results/validation

# Set Python path
export PYTHONPATH="${PYTHONPATH}:../code/src"

echo ""
echo "Step 1/5: Data Preparation"
echo "-------------------------------------------"
if [ -f "../data/processed/rssm_meps_prepared.csv" ]; then
    echo "✓ Processed data already exists"
else
    echo "Processing MEPS data..."
    python ../code/src/rssm_data_loader.py
fi

echo ""
echo "Step 2/5: Model Training"
echo "-------------------------------------------"
if [ -f "../results/rssm_best_model.pt" ]; then
    echo "✓ Trained model already exists"
else
    echo "Training RSSM world model..."
    python ../code/src/rssm_training.py
fi

echo ""
echo "Step 3/5: Temporal Validation"
echo "-------------------------------------------"
echo "Running validation on 2021-2025 data..."
jupyter nbconvert --to notebook --execute \
    ../code/notebooks/03_temporal_validation.ipynb \
    --output-dir=../results/validation/

echo ""
echo "Step 4/5: Cost-Effectiveness Analysis"
echo "-------------------------------------------"
echo "Running CEA..."
jupyter nbconvert --to notebook --execute \
    ../code/notebooks/04_cost_effectiveness.ipynb \
    --output-dir=../results/

echo ""
echo "Step 5/5: Generate Figures"
echo "-------------------------------------------"
echo "Generating publication figures..."
jupyter nbconvert --to notebook --execute \
    ../code/notebooks/01_comprehensive_trends.ipynb \
    --output-dir=../results/

jupyter nbconvert --to notebook --execute \
    ../code/notebooks/02_rssm_science_sim.ipynb \
    --output-dir=../results/

echo ""
echo "=========================================="
echo "✓ Pipeline Complete!"
echo "=========================================="
echo ""
echo "Outputs generated:"
echo "  - Figures: ../results/figures/"
echo "  - Tables: ../results/tables/"
echo "  - Validation: ../results/validation/"
echo ""
echo "Compare results to ../code/docs/expected_outputs.json"
echo "to verify correct reproduction."
