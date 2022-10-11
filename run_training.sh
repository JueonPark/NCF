#!/bin/bash
shopt -s extglob

for filename in xla_hlo/*; do
  if [[ $filename == *"offline_execution_result"* ]]
  then
    echo $filename
    continue
  else
    rm $filename
  fi
done
rm xla_hlo/*/*

# XLA
export DATADIR=/home/shared/ILSVRC2012
export TF_XLA_FLAGS="--tf_xla_clustering_debug --tf_xla_auto_jit=2"
export TF_DUMP_GRAPH_PREFIX="./xla_hlo"

# simulation
export TRACER_TOOL=/home/jueonpark/cxl-simulator/multi_gpu_simulator/util/tracer_nvbit/tracer_tool/tracer_tool.so
export POST_PROCESSING=/home/jueonpark/cxl-simulator/multi_gpu_simulator/util/tracer_nvbit/tracer_tool/traces-processing/post-traces-processing
# caution!
# for image size 32 baseline: 3071
# for image size 32 fo: 3048?
# for image size 224 and ViT base configuration:
# - 1589-2002
# - 4403-4898
# for image size 224 and ViT base configuration, with FO:
# - 1582-1939
# - 4336-4795 
export DYNAMIC_KERNEL_LIMIT_START=4336
export DYNAMIC_KERNEL_LIMIT_END=4795

# additional runtime environment variables for tensorflow
# export TF_CPP_MIN_VLOG_LEVEL=1
# export ENABLE_CONSOLE=true

# execution options:
# $1:
# - vanila for no
# - pm for pattern matching
# - fo for fusion offloading
# - pm_fo for both pattern matching & fusion offlaoding
# - ideal for ideal offloading
# - pm_ideal for both pattern matching & ideal offloading
# $2: trace generation
# - keyword "trace" given
# $3: xla_ndpx_use_offline_result
# - 0 for using GPU results
# - 1 for using SIM results
# - on default(no $3 input), use GPU results
if [ $1 = "vanila" ]
then
  export XLA_FLAGS="--xla_dump_hlo_as_text  --xla_dump_to=./xla_hlo "
elif [ $1 = "pm" ]
then
  export XLA_FLAGS="--xla_dump_hlo_as_text --xla_gpu_use_cudnn_batchnorm=true --xla_gpu_use_ndp_batchnorm=true --xla_gpu_use_ndp_bert_pattern=true --xla_gpu_use_ndp_optimizer=true --xla_dump_to=./xla_hlo "
elif [ $1 = "fo" ]
then
  if [ $# = 3 ]
  then
    export XLA_FLAGS="--xla_dump_hlo_as_text --xla_ndpx_use_fusion_offloading=true --xla_ndpx_use_offline_result=$3 --xla_dump_to=./xla_hlo "
  else
    export XLA_FLAGS="--xla_dump_hlo_as_text --xla_ndpx_use_fusion_offloading=true --xla_ndpx_use_offline_result=0 --xla_dump_to=./xla_hlo "
  fi
elif [ $1 = "pm_fo" ]
then
  if [ $# = 3 ]
  then
    export XLA_FLAGS="--xla_dump_hlo_as_text --xla_gpu_use_cudnn_batchnorm=true --xla_gpu_use_ndp_batchnorm=true --xla_gpu_use_ndp_bert_pattern=true --xla_gpu_use_ndp_optimizer=true --xla_dump_hlo_as_text --xla_ndpx_use_fusion_offloading=true --xla_ndpx_use_offline_result=$3 --xla_dump_to=./xla_hlo "
  else
    export XLA_FLAGS="--xla_dump_hlo_as_text --xla_gpu_use_cudnn_batchnorm=true --xla_gpu_use_ndp_batchnorm=true --xla_gpu_use_ndp_bert_pattern=true --xla_gpu_use_ndp_optimizer=true --xla_dump_hlo_as_text --xla_ndpx_use_fusion_offloading=true --xla_ndpx_use_offline_result=0 --xla_dump_to=./xla_hlo "
  fi
elif [ $1 = "ideal" ]
then
  export XLA_FLAGS="--xla_dump_hlo_as_text --xla_ndpx_use_fusion_offloading=true --xla_ndpx_use_ideal_offloading --xla_dump_to=./xla_hlo "
elif [ $1 = "pm_ideal" ]
then
  export XLA_FLAGS="--xla_dump_hlo_as_text --xla_gpu_use_cudnn_batchnorm=true --xla_gpu_use_ndp_batchnorm=true --xla_gpu_use_ndp_bert_pattern=true --xla_gpu_use_ndp_optimizer=true --xla_dump_hlo_as_text --xla_ndpx_use_fusion_offloading=true --xla_ndpx_use_ideal_offloading --xla_dump_to=./xla_hlo "
else
  echo "flags: vanila, pm, fo, pm_fo, ideal, idea_fo"
	exit 0
fi

# whether to get trace or not
if [ $# -ge 2 ] && [ $2 = "trace" ]
then
  # LD_PRELOAD=$TRACER_TOOL python GMF.py --dataset ml-1m --epochs 1 --batch_size 256 --num_factors 8 --regs [0,0] --num_neg 4 --lr 0.001 --learner adam --verbose 1 --out 1
  LD_PRELOAD=$TRACER_TOOL python MLP.py --dataset ml-1m --epochs 1 --batch_size 256 --layers [64,32,16,8] --reg_layers [0,0,0,0] --num_neg 4 --lr 0.001 --learner adam --verbose 1 --out 1
  # LD_PRELOAD=$TRACER_TOOL python NeuMF.py --dataset ml-1m --epochs 1 --batch_size 256 --num_factors 8 --layers [64,32,16,8] --reg_mf 0 --reg_layers [0,0,0,0] --num_neg 4 --lr 0.001 --learner adam --verbose 1 --out 1
  $POST_PROCESSING ./traces/kernelslist
else
  # python GMF.py --dataset ml-1m --epochs 1 --batch_size 256 --num_factors 8 --regs [0,0] --num_neg 4 --lr 0.001 --learner adam --verbose 1 --out 1
  python MLP.py --dataset ml-1m --epochs 1 --batch_size 256 --layers [64,32,16,8] --reg_layers [0,0,0,0] --num_neg 4 --lr 0.001 --learner adam --verbose 1 --out 1
  # python NeuMF.py --dataset ml-1m --epochs 20 --batch_size 256 --num_factors 8 --layers [64,32,16,8] --reg_mf 0 --reg_layers [0,0,0,0] --num_neg 4 --lr 0.001 --learner adam --verbose 1 --out 1
fi
