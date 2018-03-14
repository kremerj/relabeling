#!/usr/bin/env bash

for EXPERIMENT in std C rho kappa burnin unbiased deep; do
	python plot.py --experiment $EXPERIMENT
done
