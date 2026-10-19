**Comprehensiveness (higher = more faithful), mean [95% bootstrap CI]**

| method | k=1 | k=3 | k=5 |
|---|---|---|---|
| attention | 0.028 [0.013, 0.043] | 0.053 [0.014, 0.091] | 0.095 [0.049, 0.139] |
| attn_x_grad | 0.087 [0.062, 0.113] | 0.194 [0.154, 0.231] | 0.234 [0.189, 0.276] |
| sae | -0.038 [-0.056, -0.020] | 0.039 [0.005, 0.074] | 0.080 [0.040, 0.120] |
| raw_gradxact | 0.028 [0.007, 0.050] | 0.073 [0.040, 0.107] | 0.107 [0.071, 0.142] |
| last_k | 0.032 [0.017, 0.047] | 0.040 [0.024, 0.055] | 0.073 [0.054, 0.090] |
| random | 0.003 [-0.012, 0.017] | 0.010 [-0.016, 0.036] | 0.039 [0.008, 0.069] |

**Sufficiency (lower = more faithful), mean [95% bootstrap CI]**

| method | k=1 | k=3 | k=5 |
|---|---|---|---|
| attention | 0.230 [0.210, 0.249] | 0.182 [0.152, 0.212] | 0.145 [0.109, 0.181] |
| attn_x_grad | 0.170 [0.146, 0.195] | 0.075 [0.046, 0.102] | -0.001 [-0.033, 0.029] |
| sae | 0.204 [0.180, 0.228] | 0.142 [0.115, 0.169] | 0.087 [0.055, 0.118] |
| raw_gradxact | 0.183 [0.154, 0.212] | 0.122 [0.088, 0.154] | 0.078 [0.039, 0.116] |
| last_k | 0.229 [0.209, 0.248] | 0.186 [0.151, 0.220] | 0.149 [0.104, 0.195] |
| random | 0.239 [0.216, 0.263] | 0.220 [0.191, 0.248] | 0.187 [0.157, 0.216] |

**Rank agreement (mean over sentences)**

```
rho_attn_sae        0.095
rho_attngrad_sae    0.418
rho_sae_raw        -0.155
resid_ratio         5.574
top1_attn_sae       0.500
top1_sae_raw        0.095
```

**Few-shot accuracy (SST-2 val):** 0.658 | **N evaluated:** 200 | class counts: {0: 100, 1: 100}

**SAE spliced-in:** mean margin 0.220 (original 0.228), fraction still correct 0.680

**Held-out feature ablation (scope=span)**

```
cls  K   top  top_by_act  bottom  rand_matched_mean  rand_matched_sd  rand_pool_mean  top_vs_matched_dz  top_vs_matched_p
NEG  1 0.010      -0.001  -0.010              0.000            0.003           0.000              0.778               0.0
NEG  5 0.023       0.006  -0.014             -0.005            0.004           0.000              1.459               0.0
NEG 10 0.029      -0.004  -0.021             -0.010            0.008          -0.000              1.502               0.0
NEG 20 0.039       0.000  -0.021             -0.011            0.004           0.000              1.781               0.0
NEG 50 0.048       0.011  -0.028             -0.014            0.003           0.000              1.909               0.0
POS  1 0.034       0.005  -0.003              0.002            0.001           0.000              2.491               0.0
POS  5 0.062       0.024  -0.010              0.009            0.006           0.000              2.358               0.0
POS 10 0.072       0.044  -0.017              0.009            0.007           0.001              2.608               0.0
POS 20 0.088       0.077  -0.024              0.006            0.003           0.000              2.844               0.0
POS 50 0.101       0.082  -0.036             -0.002            0.006           0.000              2.922               0.0
```

**Direction baseline (non-SAE)**

```
cls scope  n_test  drop_diffmean_dir  null_mean  null_sd  dz_vs_null  p_vs_null
NEG  span      51             -0.017     -0.002    0.015      -0.380      0.009
POS  span      49              0.123     -0.006    0.025       2.164      0.000
```