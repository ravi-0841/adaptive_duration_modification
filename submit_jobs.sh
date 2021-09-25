#! /bin/bash

hidden_dims=( 128 256 )
encoder_decoder_layers=( 6 10 12 )

counter=1

for l in "${encoder_decoder_layers[@]}"
do
    for d in "${hidden_dims[@]}"
    do
        sbatch -J $counter-vesus-conv -o "./txt_files/layers_${l}_dim_${d}_vesus_conv_5ms_sum_drop_9.txt" convolutional_job_script.sh $d $l
        counter=$((counter+1))
    done
done
