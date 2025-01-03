# SemNeSyKTRecSys
Semantic Neuro-Symbolic Knowledge Transfer for Recommender Systems

- tuning based on NDCG@10
- decide how to encode the ranking information inside LikesSource
- define the BPR loss in LTN -> almost done -> to be implemented
- define the LTN loader for the complete model -> define cold start users in target -> define all the pairs from which to sample
based on this information. Then, we need to define for which of these pairs the axioms are satisfied, namely the user likes
or dislikes something connected in the source domain. We should define all these pairs and define a DIAG on them