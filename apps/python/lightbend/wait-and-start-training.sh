#!/bin/bash

filename=$GENERATION_COMPLETE_FILE_NAME
while [ ! -f $filename ]
do
  sleep 2
done
echo "Data generation completed .."
cat $filename

echo "Starting training .."
analytics-zoo/scripts/spark-submit-with-zoo.sh dnn_anomaly_bigdl.py

echo "Training done"

# remove the generation complete file
rm $GENERATION_COMPLETE_FILE_NAME

datafile=$(ls -l $DATA_FILE_NAME)
echo $datafile

pbfile=$(ls -l $MODEL_PB_FILE_NAME)
echo $pbfile

attribfile=$(ls -l $MODEL_ATTRIB_FILE_NAME)
echo $attribfile
cat $MODEL_ATTRIB_FILE_NAME

echo date > $TRAINING_COMPLETE_FILE_NAME
