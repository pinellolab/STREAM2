#!/bin/bash
ENVS=$(conda env list | awk '{print $1}' )
if [[ $ENVS = *"$1"* ]]; then
   #source /data/pinello/SHARED_SOFTWARE/anaconda_latest/etc/profile.d/conda.sh
   #source ~/anaconda3/etc/profile.d/conda.sh
   conda activate $1
   echo "change conda env to $1"
   echo 'Start ChromVAR computation' 
   parent_path=$( cd "$(dirname "${BASH_SOURCE[0]}")" ; pwd -P )
   R CMD BATCH --no-save --no-restore "--args input='$2' species='$3' genome='$4' feature='$5' n_jobs=$6" $parent_path/stream2_chromVar.R $2/stream2_chromVar.out &
   date
   wait
else 
   echo "Error: Please provide a valid virtual environment. or create a new environment by run 'conda create -n stream2_chromVar R bioconductor-chromvar' "
   exit
fi;