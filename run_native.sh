#!/bin/bash
set -e

ls

./run_bagging.sh 1,3,5,10 uniform,linear wine_quality 50 native
wait

./run_bagging.sh 1,3,5,10 uniform,linear covertype,dionis,higgs,allstate_claims_severity 200 native
wait

./run_bagging.sh 1,3,5,10 uniform,linear helena,road_safety,jannis,house_sales,diamonds 100 native
wait
