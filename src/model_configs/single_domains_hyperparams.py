from src.model_configs.ModelConfig import MfHyperParams

mf_books_hyperparams = MfHyperParams(
    n_factors=200, learning_rate=0.0001810210202954595, weight_decay=0.00026616161728996144, batch_size=256
)

mf_movies_hyperparams = MfHyperParams(
    n_factors=200, learning_rate=0.00019603650397325152, weight_decay=0.07996262790412291, batch_size=512
)

mf_music_hyperparams = MfHyperParams(
    n_factors=200, learning_rate=0.000404269153265299, weight_decay=0.07673355263885923, batch_size=512
)
