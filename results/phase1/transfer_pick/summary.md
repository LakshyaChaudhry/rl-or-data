# eval summary — runs/transfer_pick

status: finished

```
model                              split               n greedy [95% CI]         mean@8  pass@8  trunc% g/T1  at-cap% g/T1 flag
------------------------------------------------------------------------------------------------------------------------------------
Qwen/Qwen3-4B-Base                 gsm8k_500         500 0.880 [0.850,0.908]        -       -    0.0/  -     0.0/  -     
Qwen/Qwen3-4B-Base                 rg_basic_arithmetic_300  300 0.757 [0.707,0.803]        -       -    0.7/  -     0.7/  -     
Qwen/Qwen3-4B-Base                 rg_count_primes_300  300 0.023 [0.007,0.040]        -       -    9.3/  -     9.3/  -     TRUNC>5%

per-tier truncation % / extraction-failure % (per split and decoding; ood_hard_200 is reported, never flagged)
model                              split            decoding        untiered
----------------------------------------------------------------------------
Qwen/Qwen3-4B-Base                 gsm8k_500        greedy       0.0/0.0  
Qwen/Qwen3-4B-Base                 rg_basic_arithmetic_300 greedy       0.7/0.7  
Qwen/Qwen3-4B-Base                 rg_count_primes_300 greedy       9.3/9.3  
```
