# Arena / human-preference diagnostics

## Top scores by group

### category=dbbench-std

| agent   |         btl |   copeland |     hodge |   kemeny_local |   minimax |   rank_centrality |   ranked_pairs |   schulze |   ste_plugin_uc |   ste_posterior_edge_uc |   winrate |
|:--------|------------:|-----------:|----------:|---------------:|----------:|------------------:|---------------:|----------:|----------------:|------------------------:|----------:|
| agent_a |  5.25397    |          2 |  0.863498 |              3 |     0.001 |        1          |              2 |         2 |        0.999941 |                    0.81 |  0.791667 |
| agent_b | -1.6836e-18 |          0 |  0        |              2 |    -1     |        6.0633e-13 |              0 |         1 |        0.855466 |                    0.25 |  0.5      |
| agent_c | -5.25397    |         -2 | -0.863498 |              1 |    -1.001 |        0          |             -2 |         0 |        0.718171 |                    0.09 |  0.208333 |

### category=os-std

| agent   |   btl |   copeland |   hodge |   kemeny_local |   minimax |   rank_centrality |   ranked_pairs |   schulze |   ste_plugin_uc |   ste_posterior_edge_uc |   winrate |
|:--------|------:|-----------:|--------:|---------------:|----------:|------------------:|---------------:|----------:|----------------:|------------------------:|----------:|
| agent_c |     0 |          0 |       0 |              1 |         0 |          0.333333 |              0 |         0 |        0.859514 |                    0.57 |       0.5 |
| agent_b |     0 |          0 |       0 |              2 |         0 |          0.333333 |              0 |         0 |        0.877551 |                    0.54 |       0.5 |
| agent_a |     0 |          0 |       0 |              3 |         0 |          0.333333 |              0 |         0 |        0.859514 |                    0.43 |       0.5 |

### global

| agent   |          btl |   copeland |     hodge |   kemeny_local |      minimax |   rank_centrality |   ranked_pairs |   schulze |   ste_plugin_uc |   ste_posterior_edge_uc |   winrate |
|:--------|-------------:|-----------:|----------:|---------------:|-------------:|------------------:|---------------:|----------:|----------------:|------------------------:|----------:|
| agent_a |  1.04465     |          2 |  0.678765 |              3 |  0.000666667 |         0.684211  |              2 |         2 |        0.995297 |                    0.74 |  0.729167 |
| agent_b |  1.08414e-17 |          0 |  0        |              2 | -0.333333    |         0.263158  |              0 |         1 |        0.730442 |                    0.33 |  0.5      |
| agent_c | -1.04465     |         -2 | -0.678765 |              1 | -1.00067     |         0.0526316 |             -2 |         0 |        0.642558 |                    0.13 |  0.270833 |

## Selected-set dominance/error diagnostics

| group                | method                |   k_selected |   cross_pair_coverage |   external_attack_rate |   selected_dominance_rate |   dominance_gap |
|:---------------------|:----------------------|-------------:|----------------------:|-----------------------:|--------------------------:|----------------:|
| category=dbbench-std | ste_posterior_edge_uc |            1 |                   1   |                    0   |                       1   |             1   |
| category=dbbench-std | ste_plugin_uc         |            1 |                   1   |                    0   |                       1   |             1   |
| category=dbbench-std | btl                   |            1 |                   1   |                    0   |                       1   |             1   |
| category=dbbench-std | winrate               |            1 |                   1   |                    0   |                       1   |             1   |
| category=dbbench-std | rank_centrality       |            1 |                   1   |                    0   |                       1   |             1   |
| category=dbbench-std | hodge                 |            1 |                   1   |                    0   |                       1   |             1   |
| category=dbbench-std | copeland              |            1 |                   1   |                    0   |                       1   |             1   |
| category=dbbench-std | schulze               |            1 |                   1   |                    0   |                       1   |             1   |
| category=dbbench-std | minimax               |            1 |                   1   |                    0   |                       1   |             1   |
| category=dbbench-std | ranked_pairs          |            1 |                   1   |                    0   |                       1   |             1   |
| category=dbbench-std | kemeny_local          |            1 |                   1   |                    0   |                       1   |             1   |
| category=os-std      | ste_posterior_edge_uc |            1 |                   0.5 |                    0.5 |                       0.5 |             0   |
| category=os-std      | ste_plugin_uc         |            1 |                   1   |                    0.5 |                       0.5 |             0   |
| category=os-std      | btl                   |            1 |                   0.5 |                    0.5 |                       0.5 |             0   |
| category=os-std      | winrate               |            1 |                   0.5 |                    0.5 |                       0.5 |             0   |
| category=os-std      | rank_centrality       |            1 |                   0.5 |                    0.5 |                       0.5 |             0   |
| category=os-std      | hodge                 |            1 |                   0.5 |                    0.5 |                       0.5 |             0   |
| category=os-std      | copeland              |            1 |                   0.5 |                    0.5 |                       0.5 |             0   |
| category=os-std      | schulze               |            1 |                   0.5 |                    0.5 |                       0.5 |             0   |
| category=os-std      | minimax               |            1 |                   0.5 |                    0.5 |                       0.5 |             0   |
| category=os-std      | ranked_pairs          |            1 |                   0.5 |                    0.5 |                       0.5 |             0   |
| category=os-std      | kemeny_local          |            1 |                   0.5 |                    0.5 |                       0.5 |             0   |
| global               | ste_posterior_edge_uc |            1 |                   1   |                    0.2 |                       0.8 |             0.6 |
| global               | ste_plugin_uc         |            1 |                   1   |                    0.2 |                       0.8 |             0.6 |
| global               | btl                   |            1 |                   1   |                    0.2 |                       0.8 |             0.6 |
| global               | winrate               |            1 |                   1   |                    0.2 |                       0.8 |             0.6 |
| global               | rank_centrality       |            1 |                   1   |                    0.2 |                       0.8 |             0.6 |
| global               | hodge                 |            1 |                   1   |                    0.2 |                       0.8 |             0.6 |
| global               | copeland              |            1 |                   1   |                    0.2 |                       0.8 |             0.6 |
| global               | schulze               |            1 |                   1   |                    0.2 |                       0.8 |             0.6 |
| global               | minimax               |            1 |                   1   |                    0.2 |                       0.8 |             0.6 |
| global               | ranked_pairs          |            1 |                   1   |                    0.2 |                       0.8 |             0.6 |
| global               | kemeny_local          |            1 |                   1   |                    0.2 |                       0.8 |             0.6 |

No high-confidence 3-cycles found under the configured count/confidence thresholds.

## Bootstrap top-1 and selected-set stability

| group                | method                |   top1_entropy | modal_top1   |   modal_freq |   mean_pairwise_set_jaccard |
|:---------------------|:----------------------|---------------:|:-------------|-------------:|----------------------------:|
| category=dbbench-std | btl                   |             -0 | agent_a      |          1   |                           1 |
| category=dbbench-std | copeland              |             -0 | agent_a      |          1   |                           1 |
| category=dbbench-std | hodge                 |             -0 | agent_a      |          1   |                           1 |
| category=dbbench-std | kemeny_local          |             -0 | agent_a      |          1   |                           1 |
| category=dbbench-std | minimax               |             -0 | agent_a      |          1   |                           1 |
| category=dbbench-std | rank_centrality       |             -0 | agent_a      |          1   |                           1 |
| category=dbbench-std | ranked_pairs          |             -0 | agent_a      |          1   |                           1 |
| category=dbbench-std | schulze               |             -0 | agent_a      |          1   |                           1 |
| category=dbbench-std | ste_plugin_uc         |             -0 | agent_a      |          1   |                           1 |
| category=dbbench-std | ste_posterior_edge_uc |             -0 | agent_a      |          1   |                           1 |
| category=dbbench-std | winrate               |             -0 | agent_a      |          1   |                           1 |
| category=os-std      | btl                   |              1 | agent_c      |          0.5 |                           0 |
| category=os-std      | copeland              |              1 | agent_c      |          0.5 |                           0 |
| category=os-std      | hodge                 |              1 | agent_c      |          0.5 |                           0 |
| category=os-std      | kemeny_local          |              1 | agent_c      |          0.5 |                           0 |
| category=os-std      | minimax               |              1 | agent_c      |          0.5 |                           0 |
| category=os-std      | rank_centrality       |              1 | agent_c      |          0.5 |                           0 |
| category=os-std      | ranked_pairs          |              1 | agent_c      |          0.5 |                           0 |
| category=os-std      | schulze               |              1 | agent_c      |          0.5 |                           0 |
| category=os-std      | ste_plugin_uc         |              1 | agent_c      |          0.5 |                           0 |
| category=os-std      | ste_posterior_edge_uc |              1 | agent_c      |          0.5 |                           0 |
| category=os-std      | winrate               |              1 | agent_c      |          0.5 |                           0 |
| global               | btl                   |              1 | agent_a      |          0.5 |                           0 |
| global               | copeland              |              1 | agent_a      |          0.5 |                           0 |
| global               | hodge                 |              1 | agent_a      |          0.5 |                           0 |
| global               | kemeny_local          |              1 | agent_a      |          0.5 |                           0 |
| global               | minimax               |              1 | agent_a      |          0.5 |                           0 |
| global               | rank_centrality       |              1 | agent_a      |          0.5 |                           0 |
| global               | ranked_pairs          |              1 | agent_a      |          0.5 |                           0 |
| global               | schulze               |              1 | agent_a      |          0.5 |                           0 |
| global               | ste_plugin_uc         |              1 | agent_a      |          0.5 |                           0 |
| global               | ste_posterior_edge_uc |              1 | agent_a      |          0.5 |                           0 |
| global               | winrate               |              1 | agent_a      |          0.5 |                           0 |
