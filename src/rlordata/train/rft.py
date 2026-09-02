"""RFT arms (SPEC §8 arms 1-2): sample -> core.verify -> core.rft_select -> SFT with core.sft_loss. AGENT-OWNED; tasks/03.

The trainer loop (LoRA via PEFT, AdamW, cosine schedule, batching, checkpointing, logging) is yours.
The loss must be computed by rlordata.core.sft_loss on rlordata.core.logprobs output — do not use a
library loss for the result-bearing runs, so the number is the one Laksh can explain.
Sweep grid and model selection on val_mixed_100 per SPEC §9-10; log every config.
"""

from __future__ import annotations
