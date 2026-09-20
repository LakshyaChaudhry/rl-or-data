# rft_curated — chosen lr=1e-05, epochs=4 (selected on val_mixed_100 only)

seed    val greedy [95% CI]   test greedy [95% CI]    ood greedy [95% CI]  mean@8  pass@8  trunc%  xfail% prompts   avail    used     tokens  steps  GPU-h       $
   1   0.660 [0.570,0.750]    0.673 [0.620,0.723]    0.300 [0.240,0.365]    0.514   0.836     2.7     2.7      73   19200    6653   13377076   1664   1.95    8.38
   2   0.660 [0.560,0.750]    0.657 [0.603,0.710]    0.290 [0.225,0.355]    0.510   0.829     3.3     3.3      73   19200    6653   13377076   1664   1.96    8.42
   3   0.630 [0.540,0.720]    0.677 [0.623,0.730]    0.265 [0.205,0.325]    0.511   0.826     3.3     3.3      73   19200    6653   13377076   1664   1.95    8.36

val greedy: 0.650 ± 0.017 over 3 seeds ['0.660', '0.660', '0.630']
test greedy: 0.669 ± 0.011 over 3 seeds ['0.673', '0.657', '0.677']
ood greedy: 0.285 ± 0.018 over 3 seeds ['0.300', '0.290', '0.265']
test_mean_at_8: 0.512 ± 0.002 ['0.514', '0.510', '0.511']
test_pass_at_8: 0.830 ± 0.005 ['0.836', '0.829', '0.826']

'!' = truncation > 5 % on that split (not a headline number). ood truncation is reported, not flagged.
No arm-vs-arm claim here (Phase 4).
