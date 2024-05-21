#!/bin/bash
set -e
cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"/

flower-superlink --insecure &
sl_pid=$!
sleep 3

pids=()

for i in `seq 0 9`; do
    echo "Starting client $i"
    flower-client-app flwr-next.client:app --insecure &
    pids+=($!)
done

sleep 3
flower-server-app flwr-next.server:app --insecure &
pid=$!

# Enable CTRL+C to stop all background processes
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM

wait $pid

kill $sl_pid
echo "Killing all client processes..."
  for pid in "${pids[@]}"; do
    kill $pid
  done
