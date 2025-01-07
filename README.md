# SemNeSyKTRecSys
Semantic Neuro-Symbolic Knowledge Transfer for Recommender Systems

- tuning based on NDCG@10
- understand the hyper-parameters of the LTN model:
  - p for forall
  - p for sat agg
  - k for top k of the source domain (are there alternatives to tuning?)
  - understand if 0 is a good number to represent negatives (we could check the average of the negatives after training)