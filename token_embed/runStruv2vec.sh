#!/bin/bash --login
#$ -cwd

#$ -pe smp.pe 32

module load tools/env/proxy

source activate python3.7.0

python getAllStruc2Vec.py

