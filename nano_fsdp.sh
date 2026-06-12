#!/bin/bash

#SBATCH --account nemotron_sw_pre
#SBATCH -p batch
#SBATCH --mem=0
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --dependency=singleton
#SBATCH --job-name=nano_fsdp

export NCCL_IB_SL=1
export NCCL_IB_TIMEOUT=19
export UB_TIMEOUT=720
export NVTE_FWD_LAYERNORM_SM_MARGIN=16
export NVTE_BWD_LAYERNORM_SM_MARGIN=16
export TORCHINDUCTOR_WORKER_START=fork
#export NVTE_FUSED_ATTN=0  # Disable cuDNN fused attention.
export NCCL_P2P_NET_CHUNKSIZE=2097152
export NCCL_DEBUG=WARN
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NVTE_CPU_OFFLOAD_V1=1
export NVTE_USE_CUTLASS_GROUPED_GEMM=0
export NVTE_USE_FAST_MATH=1

export NCCL_SHM_DISABLE=1
export NCCL_PROTO=simple
export NCCL_NVLS_ENABLE=0
#export NCCL_SYM_GIN_KERNELS_ENABLE=1
export NVTE_CUTEDSL_FUSED_GROUPED_MLP=1

export NUM_OF_TOKENS_PER_CHUNK_COMBINE_API=128
#export NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN=64
export NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN=32
export USE_MNNVL=1

EXIT_INTERVAL=1000

export WANDB_RESUME="allow"

# Mandatory environment variable checks
if [[ -z "${WANDB_API_KEY:-}" ]]; then
    echo "ERROR: WANDB_API_KEY is not set. Please export WANDB_API_KEY before running." >&2
    exit 1
fi
if [[ -z "${LUSTRE_ROOT:-}" ]]; then
    echo "ERROR: LUSTRE_ROOT is not set. Please export LUSTRE_ROOT before running." >&2
    exit 1
fi

# ROOT PATHS
ASSETS_ROOT="${LUSTRE_ROOT}/assets/nemotron"

DATACACHE_DIR="${ASSETS_ROOT}/data-cache"
TOKENIZER_MODEL_PATH="${ASSETS_ROOT}/tokenizers/multiMixV8.gpt4o_nc_sd.500000.128k.vocab.json"
BLEND_PATH="${ASSETS_ROOT}/blend_files/1t_singlephase.json"

MEGATRON_LM_DIR="${LUSTRE_ROOT}/Megatron-LM"
OUTPUT_ROOT="${LUSTRE_ROOT}/logs"
IMAGE="${ASSETS_ROOT}/30u1719b2bbb.sqsh"
########################################################
#### CHANGES SHOULD NOT BE NEEDED BEYOND THIS POINT ####
########################################################

DATETIME=`date +'date_%y-%m-%d_time_%H-%M-%S'`
IFS=':' read -r -a array <<< "${SLURM_JOB_NAME}"
NAME="${array[1]}"

if [ -n "${SLURM_JOB_ID:-}" ] ; then
    SCRIPT_PATH=$(scontrol show job "$SLURM_JOB_ID" | awk -F= '/Command=/{print $2}')
    ENV_LOG_FILENAME=${NAME}_${SLURM_JOB_ID}_${DATETIME}.env.log
else
    SCRIPT_PATH=$(realpath "$0")
    ENV_LOG_FILENAME=${NAME}_${DATETIME}.env.log
fi

RUN_DIR="${OUTPUT_ROOT}/${NAME}"
LOGS_DIR="${RUN_DIR}/logs"
CHECKPOINT_DIR="${RUN_DIR}/checkpoints"
TENSORBOARD_DIR="${RUN_DIR}/tensorboard"

# Mamba triton cache.
#export TRITON_CACHE_DIR="${OUTPUT_ROOT}/triton-cache"
export TRITON_CACHE_DIR="/tmp/triton_cache_${SLURM_NODEID}"
#export TRITON_CACHE_DIR="/tmp/triton_cache_${SLURM_JOB_ID}"
#TRITON_CACHE_MANAGER="megatron.core.ssm.triton_cache_manager:ParallelFileCacheManager"
#export TRITON_CACHE_DIR="/tmp/triton_cache/"

mkdir -p ${LOGS_DIR}
mkdir -p ${CHECKPOINT_DIR}
mkdir -p ${DATACACHE_DIR}
mkdir -p ${TENSORBOARD_DIR}

################################################################
### Log environment
################################################################
echo "<< START PATHS >>" |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
echo "IMAGE=${IMAGE}" |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
echo "OUTPUT_ROOT=${OUTPUT_ROOT}" |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
echo "MEGATRON_LM_DIR=${MEGATRON_LM_DIR}" |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
echo "RUN_DIR=${RUN_DIR}" |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
echo "LOGS_DIR=${LOGS_DIR}" |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
echo "CHECKPOINT_DIR=${CHECKPOINT_DIR}" |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
echo "DATACACHE_DIR=${DATACACHE_DIR}" |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
echo "TENSORBOARD_DIR=${TENSORBOARD_DIR}" |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
echo "<< END PATHS >>" |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
echo -e "\n\n" |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}

echo "<< START GIT >>" |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
echo "GIT LOG" |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
git -C ${MEGATRON_LM_DIR} log --oneline -1 |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
echo -e "\n\n" |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
echo "GIT STATUS" |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
git -C ${MEGATRON_LM_DIR} status --porcelain --branch |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
echo -e "\n\n" |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
echo "GIT DIFF" |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
git -C ${MEGATRON_LM_DIR} diff |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
echo "<< END GIT >>" |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
echo -e "\n\n" |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}

echo "<< START ENV >>" |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
env |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
echo "<< END ENV >>" |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}




# # Switch from phase 1 to 2 at 60% (iteration 28K)
# BLEND_PATH="/lustre/fsw/portfolios/llmservice/users/rwaleffe/blend_files/N5.5-phase1-FINAL-1-66th.json"
# #BLEND_PATH="/lustre/fsw/portfolios/llmservice/users/rwaleffe/blend_files/N5.5-phase2-FINAL-1-66th.json"

# # Copy scripts.
# mkdir -p ${RUN_DIR}/scripts/data
# cp ${SCRIPT_PATH} ${RUN_DIR}/scripts
# #cp ${BLEND_PATH} ${RUN_DIR}/scripts/data

SEQ_LEN=8192
TRAIN_SAMPLES=122_070_313
LR_WARMUP_SAMPLES=1_024_000
LR_DECAY_SAMPLES=122_070_313
LR_WSD_DECAY_SAMPLES=18_310_547

options=" \
        --moe-router-score-function sigmoid \
        --moe-grouped-gemm \
        --num-experts 128 \
        --moe-router-topk 6 \
        --moe-aux-loss-coeff 1e-4 \
        --moe-router-topk-scaling-factor 2.5 \
        --moe-router-enable-expert-bias \
        --moe-router-dtype fp32 \
        --moe-router-load-balancing-type seq_aux_loss \
        --moe-shared-expert-intermediate-size 3712 \
        --moe-permute-fusion \
        --moe-flex-dispatcher-backend hybridep \
        --moe-token-dispatcher-type flex \
        --moe-hybridep-num-sms 32 \
        --moe-router-fusion \
        \
        --num-workers 1 \
        --disable-gloo-process-groups \
        --ckpt-format torch_dist \
        --load ${CHECKPOINT_DIR} \
        --save ${CHECKPOINT_DIR} \
        --save-interval 500 \
        --save-retain-interval 2000 \
        --ckpt-fully-parallel-save \
        --ckpt-fully-parallel-load \
        --async-save \
        --use-persistent-ckpt-worker \
        --ckpt-assume-constant-structure \
        \
        --squared-relu \
        --no-mmap-bin-files \
        --distributed-timeout-minutes 10 \
        --exit-duration-in-mins 1430 \
        --no-create-attention-mask-in-dataloader \
        \
        --overlap-grad-reduce \
        --overlap-param-gather \
        --tensor-model-parallel-size 1 \
        --expert-model-parallel-size 32 \
        --expert-tensor-parallel-size 1 \
        --pipeline-model-parallel-size 1 \
        --use-distributed-optimizer \
        --high-priority-stream-groups ep \
        --ddp-num-buckets 24 \
        --grad-reduce-in-bf16 \
        --ddp-reduce-scatter-with-fp32-accumulation \
        --hybrid-layer-pattern MEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEMEM*EMEMEMEME \
        --spec megatron.core.models.hybrid.hybrid_layer_specs hybrid_stack_spec \
        --hidden-size 2688 \
        --num-attention-heads 32 \
        --group-query-attention \
        --num-query-groups 2 \
        --mamba-num-heads 64 \
        --untie-embeddings-and-output-weights \
        --init-method-std 0.0173 \
        --position-embedding-type none \
        --ffn-hidden-size 1856 \
        --kv-channels 128 \
        --seq-length ${SEQ_LEN} \
        --max-position-embeddings ${SEQ_LEN} \
        --train-samples ${TRAIN_SAMPLES} \
        --lr-decay-style WSD \
        --lr-warmup-samples ${LR_WARMUP_SAMPLES} \
        --lr-decay-samples ${LR_DECAY_SAMPLES} \
        --lr-wsd-decay-style minus_sqrt \
        --lr-wsd-decay-samples ${LR_WSD_DECAY_SAMPLES} \
        --data-cache-path ${DATACACHE_DIR} \
        --tiktoken-pattern v2 \
        --tokenizer-type TikTokenizer \
        --tokenizer-model ${TOKENIZER_MODEL_PATH} \
        --distributed-backend nccl \
        --micro-batch-size 1 \
        --global-batch-size 768 \
        --lr 1.2e-3 \
        --min-lr 1.2e-5 \
        --weight-decay 0.1 \
        --clip-grad 1.0 \
        --attention-dropout 0.0 \
        --hidden-dropout 0.0 \
        --disable-bias-linear \
        --normalization RMSNorm \
        --adam-beta1 0.9 \
        --adam-beta2 0.95 \
        --log-interval 1 \
        --log-params-norm \
        --log-num-zeros-in-grad \
        --log-throughput \
        --log-device-memory-used \
        --eval-interval 1000 \
        --eval-iters 14 \
        --bf16 \
        --use-mcore-models \
        --enable-experimental \
        --manual-gc \
        --manual-gc-interval 100 \
        --use-fused-weighted-squared-relu \
        --cross-entropy-loss-fusion \
        --cross-entropy-fusion-impl native \
        --exit-interval ${EXIT_INTERVAL} \
        --tensorboard-dir ${TENSORBOARD_DIR} \
        --use-transformer-engine-op-fuser \
        --per-split-data-args-path ${BLEND_PATH} \
        --log-memory-interval 1000 \
        --log-progress \
        --log-energy \
        --logging-level 20 \
        --check-weight-hash-across-dp-replicas-interval 20000 \
        --disable-straggler-on-startup \
        --straggler-minmax-count 16 \
        --timing-log-option minmax \
        --attention-backend flash \
        --te-rng-tracker"
        #--save ${CHECKPOINT_DIR} \
        #--load ${CHECKPOINT_DIR} \
        #--save-interval 2000 \
        #--recompute-granularity selective \
        #--recompute-modules layernorm \
        #--moe-shared-expert-overlap \
        #--moe-shared-expert-compute-before-router \
        #--hybrid-override-pattern MEMEMEM*EMEMEMEMEM*EMEMEMEMEM*EMEMEMEMEM*EMEMEMEMEM*EMEMEMEMEM*EMEMEMEMEM*EMEMEMEMEM*EMEMEMEMEM*EMEMEMEMEM*EMEMEMEMEME \
        #--hybrid-override-pattern \"ME|ME|ME|M*E|ME|ME|ME|ME|M*E|ME|ME|ME|ME|M*E|ME|ME|ME|ME|M*E|ME|ME|ME|ME|M*E|ME|ME|ME|ME|M*E|ME|ME|ME|ME|M*E|ME|ME|ME|ME|M*E|ME|ME|ME|ME|M*E|ME|ME|ME|ME|M*E|ME|ME|ME|ME|ME\" \
        
        #--moe-hybridep-permute-fusion \
        #--enable-cuda-graph \
        #--cuda-graph-scope mamba attn moe_router \
        #--te-rng-tracker \
        #--offload-optimizer-states \


# --fine-grained-activation-offloading \
# --offload-modules moe_act \
# --recompute-granularity selective \
# --recompute-modules moe_act \

mxfp8_options=" \
    --moe-router-padding-for-quantization \
    --fp8-format e4m3 \
    --fp8-recipe mxfp8 \
    --fp8-param-gather \
    --reuse-grad-buf-for-mxfp8-param-ag"

fsdp_options=" \
    --use-megatron-fsdp \
    --num-distributed-optimizer-instances 2 \
    --outer-dp-sharding-strategy optim \
    --data-parallel-sharding-strategy optim_grads_params \
    --no-gradient-accumulation-fusion \
    --ckpt-format fsdp_dtensor \
    --megatron-fsdp-grad-comm-dtype bf16 \
    --megatron-fsdp-main-params-dtype fp32 \
    --megatron-fsdp-main-grads-dtype bf16"
    #--use-nccl-ub \
    #--use-sharp \

wandb_options=" \
    --wandb-project nemotron_convergence \
    --wandb-exp-name nano_fsdp \
    --wandb-save-dir ${RUN_DIR}/wandb/ \
    --wandb-entity nvidia"

#nsys_cmd="nsys profile -s none -t nvtx,cuda-sw -o ${RUN_DIR}/${NAME}_node${SLURM_NODEID}_rank${SLURM_PROCID} --force-overwrite true --cuda-graph-trace=node --capture-range=cudaProfilerApi --capture-range-end=stop"
#run_cmd="${nsys_cmd} python -u ${MEGATRON_LM_DIR}/pretrain_mamba.py ${options} ${mxfp8_options} ${mtp_options} ${fsdp_options} ${profile_options}"

export PYTHONPATH=${MEGATRON_LM_DIR}:${PYTHONPATH}

run_cmd="python -u ${MEGATRON_LM_DIR}/pretrain_hybrid.py ${options} ${mxfp8_options} ${fsdp_options} ${wandb_options} ${mock_options}"

srun -l --mpi=pmix \
    --container-image "${IMAGE}" \
    --container-mounts "/lustre:/lustre" \
    --output="${LOGS_DIR}/%x_%j_${DATETIME}.log" \
    ${ASSETS_ROOT}/bindpcie \
    sh -c "${run_cmd}"

    #numactl --cpunodebind=$((SLURM_LOCALID/2)) --membind=$((SLURM_LOCALID/2)) \
