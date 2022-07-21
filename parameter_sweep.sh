#!/bin/bash

for param1 in 0.1 0.2 0.3; do
  for param2 in 0.001 0.0001 0.00001; do
    bash submit_training_job.sh dpcgan_job_${param1}_${param2} 1 "python3 main.py --param1 ${param1} --param2 ${param2}"
  done
done
