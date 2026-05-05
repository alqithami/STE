# Arena / human-preference diagnostics

## Top scores by group

### category=coding

| agent       |      btl |   copeland |     hodge |   kemeny_local |   minimax |   rank_centrality |   ranked_pairs |   schulze |   ste_plugin_uc |   ste_posterior_edge_uc |   winrate |
|:------------|---------:|-----------:|----------:|---------------:|----------:|------------------:|---------------:|----------:|----------------:|------------------------:|----------:|
| model_gamma |  1.60782 |          1 |  0.274653 |              2 | -0.999667 |          0.333333 |              1 |         2 |        0.86469  |                   0.725 |  0.583333 |
| model_beta  |  1.60782 |          1 |  0.274653 |              3 | -0.999667 |          0.333333 |              2 |         2 |        0.86469  |                   0.71  |  0.583333 |
| model_alpha |  1.60782 |          1 |  0.274653 |              4 | -0.999667 |          0.333333 |              0 |         2 |        0.86469  |                   0.68  |  0.583333 |
| model_delta | -4.82346 |         -3 | -0.823959 |              1 | -1.001    |          0        |             -3 |         0 |        0.833485 |                   0.195 |  0.25     |

### category=writing

| agent       |   btl |   copeland |   hodge |   kemeny_local |   minimax |   rank_centrality |   ranked_pairs |   schulze |   ste_plugin_uc |   ste_posterior_edge_uc |   winrate |
|:------------|------:|-----------:|--------:|---------------:|----------:|------------------:|---------------:|----------:|----------------:|------------------------:|----------:|
| model_beta  |     0 |          0 |       0 |              3 |        -1 |          0.333333 |              0 |         1 |        0.863784 |                   0.705 |       0.5 |
| model_alpha |     0 |          0 |       0 |              2 |        -1 |          0.333333 |             -1 |         1 |        0.863784 |                   0.685 |       0.5 |
| model_gamma |     0 |          0 |       0 |              1 |        -1 |          0.333333 |              1 |         1 |        0.863784 |                   0.67  |       0.5 |

### global

| agent       |      btl |   copeland |     hodge |   kemeny_local |      minimax |   rank_centrality |   ranked_pairs |   schulze |   ste_plugin_uc |   ste_posterior_edge_uc |   winrate |
|:------------|---------:|-----------:|----------:|---------------:|-------------:|------------------:|---------------:|----------:|----------------:|------------------------:|----------:|
| model_gamma |  1.60782 |          1 |  0.274653 |              2 |  0.000333333 |          0.333333 |              1 |         1 |        0.877846 |                   0.59  |  0.583333 |
| model_beta  |  1.60782 |          1 |  0.274653 |              3 |  0.000333333 |          0.333333 |              1 |         1 |        0.877846 |                   0.535 |  0.583333 |
| model_alpha |  1.60782 |          1 |  0.274653 |              4 |  0.000333333 |          0.333333 |              1 |         1 |        0.877846 |                   0.51  |  0.583333 |
| model_delta | -4.82345 |         -3 | -0.823959 |              1 | -1.001       |          0        |             -3 |         0 |        0.833484 |                   0.205 |  0.25     |

## Selected-set dominance/error diagnostics

| group            | method                |   k_selected |   cross_pair_coverage |   external_attack_rate |   selected_dominance_rate |   dominance_gap |
|:-----------------|:----------------------|-------------:|----------------------:|-----------------------:|--------------------------:|----------------:|
| category=coding  | ste_posterior_edge_uc |            1 |                     1 |               0.333333 |                  0.666667 |        0.333333 |
| category=coding  | ste_plugin_uc         |            1 |                     1 |               0.333333 |                  0.666667 |        0.333333 |
| category=coding  | btl                   |            1 |                     1 |               0.333333 |                  0.666667 |        0.333333 |
| category=coding  | winrate               |            1 |                     1 |               0.333333 |                  0.666667 |        0.333333 |
| category=coding  | rank_centrality       |            1 |                     1 |               0.333333 |                  0.666667 |        0.333333 |
| category=coding  | hodge                 |            1 |                     1 |               0.333333 |                  0.666667 |        0.333333 |
| category=coding  | copeland              |            1 |                     1 |               0.333333 |                  0.666667 |        0.333333 |
| category=coding  | schulze               |            1 |                     1 |               0.333333 |                  0.666667 |        0.333333 |
| category=coding  | minimax               |            1 |                     1 |               0.333333 |                  0.666667 |        0.333333 |
| category=coding  | ranked_pairs          |            1 |                     1 |               0.333333 |                  0.666667 |        0.333333 |
| category=coding  | kemeny_local          |            1 |                     1 |               0.333333 |                  0.666667 |        0.333333 |
| category=writing | ste_posterior_edge_uc |            1 |                     1 |               0.5      |                  0.5      |        0        |
| category=writing | ste_plugin_uc         |            1 |                     1 |               0.5      |                  0.5      |        0        |
| category=writing | btl                   |            1 |                     1 |               0.5      |                  0.5      |        0        |
| category=writing | winrate               |            1 |                     1 |               0.5      |                  0.5      |        0        |
| category=writing | rank_centrality       |            1 |                     1 |               0.5      |                  0.5      |        0        |
| category=writing | hodge                 |            1 |                     1 |               0.5      |                  0.5      |        0        |
| category=writing | copeland              |            1 |                     1 |               0.5      |                  0.5      |        0        |
| category=writing | schulze               |            1 |                     1 |               0.5      |                  0.5      |        0        |
| category=writing | minimax               |            1 |                     1 |               0.5      |                  0.5      |        0        |
| category=writing | ranked_pairs          |            1 |                     1 |               0.5      |                  0.5      |        0        |
| category=writing | kemeny_local          |            1 |                     1 |               0.5      |                  0.5      |        0        |
| global           | ste_posterior_edge_uc |            1 |                     1 |               0.4      |                  0.6      |        0.2      |
| global           | ste_plugin_uc         |            1 |                     1 |               0.4      |                  0.6      |        0.2      |
| global           | btl                   |            1 |                     1 |               0.4      |                  0.6      |        0.2      |
| global           | winrate               |            1 |                     1 |               0.4      |                  0.6      |        0.2      |
| global           | rank_centrality       |            1 |                     1 |               0.4      |                  0.6      |        0.2      |
| global           | hodge                 |            1 |                     1 |               0.4      |                  0.6      |        0.2      |
| global           | copeland              |            1 |                     1 |               0.4      |                  0.6      |        0.2      |
| global           | schulze               |            1 |                     1 |               0.4      |                  0.6      |        0.2      |
| global           | minimax               |            1 |                     1 |               0.4      |                  0.6      |        0.2      |
| global           | ranked_pairs          |            1 |                     1 |               0.4      |                  0.6      |        0.2      |
| global           | kemeny_local          |            1 |                     1 |               0.4      |                  0.6      |        0.2      |

No high-confidence 3-cycles found under the configured count/confidence thresholds.

## Bootstrap top-1 and selected-set stability

| group            | method                |   top1_entropy | modal_top1   |   modal_freq |   mean_pairwise_set_jaccard |
|:-----------------|:----------------------|---------------:|:-------------|-------------:|----------------------------:|
| category=coding  | kemeny_local          |             -0 | model_alpha  |          1   |                           1 |
| category=coding  | ranked_pairs          |             -0 | model_beta   |          1   |                           1 |
| category=coding  | btl                   |             -0 | model_alpha  |          1   |                           0 |
| category=coding  | copeland              |             -0 | model_alpha  |          1   |                           0 |
| category=coding  | hodge                 |             -0 | model_alpha  |          1   |                           0 |
| category=coding  | minimax               |             -0 | model_alpha  |          1   |                           0 |
| category=coding  | rank_centrality       |             -0 | model_gamma  |          1   |                           0 |
| category=coding  | schulze               |             -0 | model_alpha  |          1   |                           0 |
| category=coding  | ste_plugin_uc         |             -0 | model_alpha  |          1   |                           0 |
| category=coding  | ste_posterior_edge_uc |              1 | model_gamma  |          0.5 |                           0 |
| category=coding  | winrate               |             -0 | model_alpha  |          1   |                           0 |
| category=writing | kemeny_local          |             -0 | model_beta   |          1   |                           1 |
| category=writing | ranked_pairs          |             -0 | model_gamma  |          1   |                           1 |
| category=writing | ste_posterior_edge_uc |             -0 | model_beta   |          1   |                           1 |
| category=writing | btl                   |             -0 | model_alpha  |          1   |                           0 |
| category=writing | copeland              |             -0 | model_alpha  |          1   |                           0 |
| category=writing | hodge                 |             -0 | model_alpha  |          1   |                           0 |
| category=writing | minimax               |             -0 | model_alpha  |          1   |                           0 |
| category=writing | rank_centrality       |             -0 | model_alpha  |          1   |                           0 |
| category=writing | schulze               |             -0 | model_alpha  |          1   |                           0 |
| category=writing | ste_plugin_uc         |             -0 | model_alpha  |          1   |                           0 |
| category=writing | winrate               |             -0 | model_alpha  |          1   |                           0 |
| global           | btl                   |              1 | model_beta   |          0.5 |                           0 |
| global           | copeland              |              1 | model_beta   |          0.5 |                           0 |
| global           | hodge                 |              1 | model_beta   |          0.5 |                           0 |
| global           | kemeny_local          |              1 | model_beta   |          0.5 |                           0 |
| global           | minimax               |              1 | model_beta   |          0.5 |                           0 |
| global           | rank_centrality       |              1 | model_beta   |          0.5 |                           0 |
| global           | ranked_pairs          |              1 | model_beta   |          0.5 |                           0 |
| global           | schulze               |              1 | model_beta   |          0.5 |                           0 |
| global           | ste_plugin_uc         |              1 | model_beta   |          0.5 |                           0 |
| global           | ste_posterior_edge_uc |              1 | model_beta   |          0.5 |                           0 |
| global           | winrate               |              1 | model_beta   |          0.5 |                           0 |
