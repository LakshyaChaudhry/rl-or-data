# AWS launch notes (do these on day 1; quotas take time)

## 0. Redeem credits and request quota
- Redeem the YC Startup School AWS credits on the account you will use; note the expiry date in the lab notebook.
- Request EC2 quota increases now: "Running On-Demand P instances" (vCPUs) and "Running On-Demand G and VT instances". Ask for enough vCPUs for one p5.4xlarge (16 vCPU) and one g6e.12xlarge (48 vCPU). Also request the Spot equivalents.

## 1. Instance choice (one GPU per run)
| Instance | GPU | Use |
|---|---|---|
| p5.4xlarge | 1× H100 80GB | Preferred for GRPO runs. On-demand/Spot availability is region-dependent (at launch: London, Mumbai, Jakarta, Tokyo, São Paulo; US regions via Capacity Blocks). Check your region first. |
| g6e.xlarge / g6e.2xlarge | 1× L40S 48GB | Fallback for everything. ~1/3 of H100 throughput, much cheaper. Fine for 4B LoRA + colocated vLLM at a 2048 cap. |
| g6e.12xlarge | 4× L40S | Run 3 seeds in parallel, one per GPU (set CUDA_VISIBLE_DEVICES per process). |
| g5.* (A10G 24GB) | — | Do not use: too tight for policy + reference + vLLM KV cache at 4B. |

## 2. Image and disk
- Deep Learning AMI (Ubuntu 22.04, PyTorch) or plain Ubuntu 22.04 + NVIDIA driver 550+.
- 500 GB gp3 root volume. Model cache at /data/hf (see setup_gpu.sh).

## 3. First boot
```
git clone <repo> && cd rl-or-data
cp .env.example .env && nano .env       # HF_TOKEN, RLORDATA_BUCKET, region, GPU rate
bash setup/setup_gpu.sh
make test
```

## 4. Spot usage
- Fine for RFT sampling and evals (idempotent). For GRPO runs use on-demand, or Spot with `--resume` (checkpoints every 25 steps to S3).

## 5. Cost guardrails
- Idle shutdown is installed by setup_gpu.sh (30 min at <5% GPU util). Never disable it.
- Every training script prints an estimated and actual GPU-hour cost using RLORDATA_GPU_RATE_USD_PER_HOUR.
- Budget ceiling for the project: $3,000 of the $10k pool. Track spend weekly in the lab notebook.
