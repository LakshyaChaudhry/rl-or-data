# rft_mixed — chosen lr=5e-05, epochs=4 (selected on val_mixed_100 only)

seed    val greedy [95% CI]   test greedy [95% CI]    ood greedy [95% CI]  mean@8  pass@8  trunc%  xfail% prompts   avail    used     tokens  steps  GPU-h       $
   1   0.670 [0.580,0.760]    0.707 [0.653,0.757]    0.280 [0.220,0.345]    0.550   0.849     2.7     2.7      95   19200    7961   15852320   1992   2.34   10.03
   2   0.660 [0.560,0.750]    0.687 [0.633,0.740]    0.280 [0.220,0.345]    0.540   0.848     1.7     1.7      95   19200    7961   15852320   1992   2.33    9.99
   3   0.630 [0.540,0.720]    0.693 [0.640,0.743]    0.260 [0.200,0.320]    0.541   0.849     1.7     1.7      95   19200    7961   15852320   1992   2.31    9.89

val greedy: 0.653 ± 0.021 over 3 seeds ['0.670', '0.660', '0.630']
test greedy: 0.696 ± 0.010 over 3 seeds ['0.707', '0.687', '0.693']
ood greedy: 0.273 ± 0.012 over 3 seeds ['0.280', '0.280', '0.260']
test_mean_at_8: 0.544 ± 0.006 ['0.550', '0.540', '0.541']
test_pass_at_8: 0.849 ± 0.001 ['0.849', '0.848', '0.849']

'!' = truncation > 5 % on that split (not a headline number). ood truncation is reported, not flagged.
No arm-vs-arm claim here (Phase 4).
