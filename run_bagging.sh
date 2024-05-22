#!/bin/bash
set -e
cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"/

echo "Starting server"
python3 server.py --pool-size=10 --num-rounds=10 --num-clients-per-round=10 &
sleep 10  # Sleep for 30s to give the server enough time to start

for i in `seq 0 9`; do
    echo "Starting client $i"
    python3 client.py --partition-id=$i --num-partitions=10 --partitioner-type=uniform &
done

# Enable CTRL+C to stop all background processes
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
# Wait for all background processes to complete
wait

#implement local_evaluation
#check other datasets
#add datasets and sample rate to arguments
