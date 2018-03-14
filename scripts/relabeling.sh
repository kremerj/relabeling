#!/bin/sh
# in case of slurm:
# sbatch --cpus-per-task=30 --output=log/$DATASET-$EXPERIMENT-slurm.out --export=ALGORITHM="",TRIAL=-1,DATASET=$DATASET,EXPERIMENT=$EXPERIMENT scripts/relabeling.sbatch

for EXPERIMENT in std rho C kappa; do
	for DATASET in ad a1a covtype w1a cod-rna ijcnn1 mushrooms; do
		python experiment.py --algorithm "" --experiment $EXPERIMENT --dataset $DATASET --trial -1 --jobs -1
	done
done
