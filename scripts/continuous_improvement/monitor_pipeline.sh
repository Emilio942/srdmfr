#!/bin/bash
# Continuous Improvement Pipeline for SRDMFR
# Runs periodic optimization and monitoring

echo "Starting SRDMFR Continuous Improvement Pipeline"
echo "Time: $(date)"

# Check dataset growth
echo "Current dataset status:"
find data/raw/medium_dataset_v1 -name "*.h5" | wc -l

# Monitor training progress  
echo "Monitoring hyperparameter tuning..."
if pgrep -f "hyperparameter_tuning.py" > /dev/null; then
    echo "Hyperparameter tuning is running"
else
    echo "No hyperparameter tuning running"
fi

# Monitor data generation
echo "Monitoring data generation..."
if pgrep -f "generate_medium_dataset.py" > /dev/null; then
    echo "Data generation is running"
    ps aux | grep "generate_medium_dataset.py" | grep -v grep | awk '{print "PID: " $2 ", CPU: " $3 "%, MEM: " $4 "%"}'
else
    echo "No data generation running"
fi

echo "Pipeline check complete at $(date)"
