#!/bin/bash
set -e

start_server() {
    local num_clients=$1
    local sample_rate=$2
    local dataloader=$3
    local num_rounds=$4
    local partitioner_type=$5
    local sampling_method=$6
    echo "Starting server with sample rate $sample_rate for $num_clients clients, dataloader $dataloader, num_rounds $num_rounds, partitioner_type $partitioner_type, and sampling method $sampling_method"
    python3 server.py --pool-size=$num_clients --num-rounds=$num_rounds --partitioner-type=$partitioner_type --num-clients-per-round=$num_clients --dataloader=$dataloader --sample-rate=$sample_rate --sampling-method=$sampling_method &
    sleep 10
}

start_clients() {
    local num_clients=$1
    local sample_rate=$2
    local dataloader=$3
    local partitioner_type=$4
    local sampling_method=$5
    for i in $(seq 0 $(($num_clients-1))); do
        echo "Starting client $i with sample rate $sample_rate for $num_clients clients, dataloader $dataloader, partitioner_type $partitioner_type, and sampling method $sampling_method"
        python3 client.py --partition-id=$i --num-partitions=$num_clients --partitioner-type=$partitioner_type --dataloader=$dataloader --sample-rate=$sample_rate --sampling-method=$sampling_method &
    done
}

run_repetitions() {
    local num_clients=$1
    local dataloader=$2
    local num_rounds=$3
    local partitioner_type=$4
    local sampling_method=$5
    local sample_rates=("${!6}")

    for sample_rate in "${sample_rates[@]}"; do
        echo "Starting repetition with sample rate $sample_rate for $num_clients clients, dataloader $dataloader, num_rounds $num_rounds, partitioner_type $partitioner_type, and sampling method $sampling_method"
        start_server $num_clients $sample_rate $dataloader $num_rounds $partitioner_type $sampling_method
        start_clients $num_clients $sample_rate $dataloader $partitioner_type $sampling_method

        trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
        wait

        echo "Completed repetition with sample rate $sample_rate for $num_clients clients, dataloader $dataloader, num_rounds $num_rounds, and sampling method $sampling_method"
        echo "Sleeping for 10 seconds before starting the next repetition..."
        sleep 10
    done

    echo "All repetitions for $num_clients clients completed."
}

cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"/

if [ "$#" -lt 5 ]; then
    echo "Usage: $0 <comma-separated-list-of-client-numbers> <comma-separated-partitioner-types> <comma-separated-dataloaders> <comma-separated-num-rounds> <comma-separated-sampling-methods> [comma-separated-sample-rates]"
    exit 1
fi

IFS=',' read -ra CLIENT_NUMS <<< "$1"
IFS=',' read -ra PARTITIONER_TYPES <<< "$2"
IFS=',' read -ra DATALOADERS <<< "$3"
IFS=',' read -ra NUM_ROUNDS <<< "$4"
IFS=',' read -ra SAMPLING_METHODS <<< "$5"
if [ -n "$6" ]; then
    IFS=',' read -ra SAMPLE_RATES <<< "$6"
else
    SAMPLE_RATES=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0)
fi

for dataloader in "${DATALOADERS[@]}"; do
    for sampling_method in "${SAMPLING_METHODS[@]}"; do
        for num_rounds in "${NUM_ROUNDS[@]}"; do
            for num_clients in "${CLIENT_NUMS[@]}"; do
                if [ "$num_clients" -eq 1 ]; then
                    echo "Starting centralized training with dataloader $dataloader, num_rounds $num_rounds, and sampling method $sampling_method"
                    python3 centralized.py --num-rounds $num_rounds --sampling-method $sampling_method --dataloader $dataloader
                else
                    for partitioner_type in "${PARTITIONER_TYPES[@]}"; do
                        run_repetitions $num_clients $dataloader $num_rounds $partitioner_type $sampling_method SAMPLE_RATES[@]
                    done
                fi
            done
        done
    done
done

# delete features fromo higgs rerun
# zip files and send to William
# also plot the correlation matrix
# finish the covertype experiment and send to William
# 1 more dataset check from Slack chat
# communication cost and training time plot
