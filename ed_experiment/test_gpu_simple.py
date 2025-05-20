# Utwórz prosty skrypt testowy test_gpu.py
from llama_cpp import Llama


# To powinno wydrukować informacje o kompilacji
llama = Llama(
    model_path="models/meta-llama-3-8b-instruct.Q4_K_M.gguf", 
    n_gpu_layers=-1,  # Użyj wszystkich warstw na GPU
    main_gpu=0,       # Użyj pierwszej karty GPU (indeks 0)
    tensor_split=None,  # Domyślnie użyj tylko jednej karty
    chat_format="llama-3",
    logits_all=True,
    verbose=True      # Włącz tryb verbose dla diagnozy
)
