# rft_easy — chosen lr=5e-05, epochs=8 (selected on val_mixed_100 only)

seed    val greedy [95% CI]   test greedy [95% CI]    ood greedy [95% CI]  mean@8  pass@8  trunc%  xfail% prompts   avail    used     tokens  steps  GPU-h       $
   1   0.670 [0.580,0.760]    0.663 [0.610,0.717]    0.250 [0.195,0.310]    0.538   0.826     2.3     2.3     100   19200   13834   47237096   6920   7.11   30.49
   2   0.630 [0.530,0.720]    0.647 [0.593,0.700]    0.260 [0.200,0.320]    0.543   0.833     3.3     3.3     100   19200   13834   47237096   6920   7.12   30.55
   3   0.670 [0.580,0.760]    0.667 [0.613,0.717]    0.255 [0.195,0.315]    0.525   0.841     1.7     1.7     100   19200   13834   47237096   6920   7.05   30.25

val greedy: 0.657 ± 0.023 over 3 seeds ['0.670', '0.630', '0.670']
test greedy: 0.659 ± 0.011 over 3 seeds ['0.663', '0.647', '0.667']
ood greedy: 0.255 ± 0.005 over 3 seeds ['0.250', '0.260', '0.255']
test_mean_at_8: 0.535 ± 0.009 ['0.538', '0.543', '0.525']
test_pass_at_8: 0.833 ± 0.008 ['0.826', '0.833', '0.841']

'!' = truncation > 5 % on that split (not a headline number). ood truncation is reported, not flagged.
No arm-vs-arm claim here (Phase 4).
