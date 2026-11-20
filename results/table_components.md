| configuration                                                           |   seeds | dev           | test_time     | test_gene     | test_db       |   cross_validation |
|:------------------------------------------------------------------------|--------:|:--------------|:--------------|:--------------|:--------------|-------------------:|
| Descriptors only (no language model)                                    |       3 | 0.942 ± 0.000 | 0.956 ± 0.001 | 0.721 ± 0.021 | 0.723 ± 0.003 |            nan     |
| Frozen language-model window embeddings (no adaptation)                 |       5 | 0.920 ± 0.001 | 0.945 ± 0.001 | 0.767 ± 0.005 | 0.675 ± 0.004 |            nan     |
| Frozen full-protein language-model values + descriptors (no adaptation) |       3 | 0.965 ± 0.001 | 0.974 ± 0.000 | 0.768 ± 0.018 | 0.747 ± 0.002 |              0.877 |
| Adapted language model, 65-residue window                               |       3 | 0.932 ± 0.001 | 0.950 ± 0.002 | 0.788 ± 0.006 | 0.672 ± 0.006 |            nan     |
| + full-protein language-model values                                    |       3 | 0.939 ± 0.002 | 0.959 ± 0.001 | 0.812 ± 0.007 | 0.692 ± 0.009 |            nan     |
| + thermodynamic objective                                               |       3 | 0.939 ± 0.002 | 0.961 ± 0.001 | 0.805 ± 0.003 | 0.688 ± 0.007 |              0.863 |
| + conservation                                                          |       2 | 0.952 ± 0.001 | 0.971 ± 0.001 | 0.848 ± 0.003 | 0.702 ± 0.007 |            nan     |
| + structure annotation                                                  |       2 | 0.946 ± 0.001 | 0.963 ± 0.001 | 0.819 ± 0.003 | 0.695 ± 0.011 |            nan     |
| + gene-level descriptors                                                |       3 | 0.955 ± 0.001 | 0.971 ± 0.001 | 0.769 ± 0.024 | 0.742 ± 0.004 |            nan     |
| + all three (released model)                                            |       3 | 0.966 ± 0.001 | 0.978 ± 0.000 | 0.803 ± 0.018 | 0.764 ± 0.005 |              0.899 |
| Released model + allele frequency (diagnostic only)                     |       2 | 0.992 ± 0.000 | 0.993 ± 0.000 | 0.913 ± 0.008 | 0.814 ± 0.006 |            nan     |
| Adapted 35M language model, window only                                 |       3 | 0.931 ± 0.002 | 0.952 ± 0.001 | 0.769 ± 0.003 | 0.692 ± 0.006 |            nan     |