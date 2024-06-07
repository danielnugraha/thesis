#!/bin/bash
set -e
cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"/

echo "Starting server"
python3 server.py --pool-size=3 --num-rounds=100 --num-clients-per-round=3 &
sleep 10  # Sleep for 30s to give the server enough time to start

for i in `seq 0 2`; do
    echo "Starting client $i"
    python3 client.py --partition-id=$i --num-partitions=3 --partitioner-type=linear --dataloader=cpu_act --sample-rate=0.5 &
done

# Enable CTRL+C to stop all background processes
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
# Wait for all background processes to complete
wait

#implement local_evaluation
#check other datasets
#add datasets and sample rate to arguments

# checkout pandas feature check
# training duration metrics (speed up on decrease of 0.1 step subsampling)
# check the model size communicated to the server
# compare with goss and native subsampling
# add 3 more datasets
# centralized 1.0 vs uniform (best of subsample rate) vs linear (best of subsample rate)
# Next meeting TUM Innenstadt Tuesday afternoon 2pm (try to find seats in StudiTUM).
# run uniform sampling
# try to visualise federated learning with xgboost and subsampling
