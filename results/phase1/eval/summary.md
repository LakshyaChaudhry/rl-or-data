# eval summary — results/phase1/eval (Phase 1, SPEC v1.6 splits, seed 1, cap 4352)

```
model                              split               n greedy [95% CI]         mean@8  pass@8  trunc% g/T1  at-cap% g/T1 flag
------------------------------------------------------------------------------------------------------------------------------------
Qwen/Qwen2.5-7B-Instruct           ood_hard_200      200 0.280 [0.220,0.345]      0.276   0.570 19.0/ 2.9   19.0/ 2.9    trunc>5% (reported)
Qwen/Qwen2.5-7B-Instruct           test_300          300 0.653 [0.600,0.707]      0.593   0.857  2.3/ 0.2    2.3/ 0.2    
Qwen/Qwen2.5-7B-Instruct           val_mixed_100     100 0.650 [0.560,0.740]      0.604   0.850  3.0/ 0.0    3.0/ 0.0    
Qwen/Qwen3-4B                      ood_hard_200      200 0.455 [0.385,0.520]      0.459   0.715 22.5/ 4.2   22.5/ 4.2    trunc>5% (reported)
Qwen/Qwen3-4B                      test_300          300 0.747 [0.697,0.797]      0.777   0.933  4.0/ 0.1    4.0/ 0.1    
Qwen/Qwen3-4B                      val_mixed_100     100 0.830 [0.750,0.900]      0.796   0.930  3.0/ 0.1    3.0/ 0.1    
Qwen/Qwen3-4B-Base                 ood_hard_200      200 0.250 [0.190,0.315]      0.173   0.520 13.0/ 3.1   13.0/ 3.1    trunc>5% (reported)
Qwen/Qwen3-4B-Base                 test_300          300 0.673 [0.617,0.727]      0.425   0.800  4.3/ 0.3    4.3/ 0.3    
Qwen/Qwen3-4B-Base                 val_mixed_100     100 0.600 [0.510,0.700]      0.420   0.830  1.0/ 0.2    1.0/ 0.2    
meta-llama/Llama-3.1-8B-Instruct   ood_hard_200      200 0.090 [0.050,0.130]      0.113   0.405 47.5/38.8   47.5/38.8    trunc>5% (reported)
meta-llama/Llama-3.1-8B-Instruct   test_300          300 0.373 [0.320,0.427]      0.335   0.693 11.3/15.2   11.3/15.2    TRUNC>5%
meta-llama/Llama-3.1-8B-Instruct   val_mixed_100     100 0.500 [0.400,0.600]      0.384   0.730  8.0/14.5    8.0/14.5    TRUNC>5%

per-tier truncation % / extraction-failure % (per split and decoding; ood_hard_200 is reported, never flagged)
model                              split            decoding            easy        medium          hard
--------------------------------------------------------------------------------------------------------
Qwen/Qwen2.5-7B-Instruct           ood_hard_200     greedy      14.3/14.3   21.3/21.3   18.7/18.7 
Qwen/Qwen2.5-7B-Instruct           ood_hard_200     mean_at_k    1.8/1.8     1.3/1.3     3.5/3.5  
Qwen/Qwen2.5-7B-Instruct           test_300         greedy       0.0/0.0     3.0/3.0     4.0/4.0  
Qwen/Qwen2.5-7B-Instruct           test_300         mean_at_k    0.0/0.0     0.0/0.0     0.6/0.6  
Qwen/Qwen2.5-7B-Instruct           test_300         pass_at_k    0.0/0.0     0.0/0.0     0.6/0.6  
Qwen/Qwen2.5-7B-Instruct           val_mixed_100    greedy       0.0/0.0     6.1/6.1     2.9/2.9  
Qwen/Qwen2.5-7B-Instruct           val_mixed_100    mean_at_k    0.0/0.0     0.0/0.0     0.0/0.0  
Qwen/Qwen3-4B                      ood_hard_200     greedy      14.3/14.3   21.3/21.3   23.7/23.7 
Qwen/Qwen3-4B                      ood_hard_200     mean_at_k    0.9/0.9     4.0/4.0     4.7/4.7  
Qwen/Qwen3-4B                      test_300         greedy       2.0/2.0     4.0/4.0     6.0/6.0  
Qwen/Qwen3-4B                      test_300         mean_at_k    0.0/0.0     0.0/0.0     0.4/0.4  
Qwen/Qwen3-4B                      test_300         pass_at_k    0.0/0.0     0.1/0.1     0.3/0.3  
Qwen/Qwen3-4B                      val_mixed_100    greedy       0.0/0.0     0.0/0.0     8.8/8.8  
Qwen/Qwen3-4B                      val_mixed_100    mean_at_k    0.4/0.4     0.0/0.0     0.0/0.0  
Qwen/Qwen3-4B-Base                 ood_hard_200     greedy       7.1/7.1    17.0/17.0   12.2/12.2 
Qwen/Qwen3-4B-Base                 ood_hard_200     mean_at_k    3.6/3.6     2.1/2.4     3.4/3.7  
Qwen/Qwen3-4B-Base                 test_300         greedy       3.0/3.0     5.0/5.0     5.0/5.0  
Qwen/Qwen3-4B-Base                 test_300         mean_at_k    0.4/0.6     0.1/0.2     0.5/1.0  
Qwen/Qwen3-4B-Base                 test_300         pass_at_k    0.3/1.0     0.4/0.7     1.0/1.4  
Qwen/Qwen3-4B-Base                 val_mixed_100    greedy       0.0/0.0     3.0/3.0     0.0/0.0  
Qwen/Qwen3-4B-Base                 val_mixed_100    mean_at_k    0.0/0.4     0.0/0.8     0.7/1.1  
meta-llama/Llama-3.1-8B-Instruct   ood_hard_200     greedy      42.9/42.9   38.3/38.3   51.1/51.1 
meta-llama/Llama-3.1-8B-Instruct   ood_hard_200     mean_at_k   25.0/25.0   34.0/34.0   41.7/41.7 
meta-llama/Llama-3.1-8B-Instruct   test_300         greedy       6.0/6.0    11.0/11.0   17.0/17.0 
meta-llama/Llama-3.1-8B-Instruct   test_300         mean_at_k    9.2/9.2    13.5/13.5   22.9/22.9 
meta-llama/Llama-3.1-8B-Instruct   test_300         pass_at_k    8.2/8.2    13.9/13.9   22.3/22.3 
meta-llama/Llama-3.1-8B-Instruct   val_mixed_100    greedy       6.1/6.1     6.1/6.1    11.8/11.8 
meta-llama/Llama-3.1-8B-Instruct   val_mixed_100    mean_at_k    6.1/6.1    12.9/12.9   24.3/24.3 
```
