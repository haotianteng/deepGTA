#!/bin/bash
source activate tensorflow
for i in {1..1000}
do
	python /Users/haotian.teng/Documents/deepGTA/NN/deepGTA/deepGTA_train.py --dummy_data true, --record_file "/Users/haotian.teng/Documents/deepGTA/NN/Test/record/R_squared_record_$i.txt"

done
