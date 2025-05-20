#!/usr/bin/env python
# generate_multi_gpu.py
import argparse
import json
import os
import itertools
import torch
import numpy as np
import tqdm
import logging
import threading
import time
from datetime import datetime
from llama_cpp import Llama

# Konfiguracja loggera
def setup_logger(log_file=None):
    logger = logging.getLogger("multi_gpu_inference")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    # Handler dla konsoli
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Handler dla pliku, jeśli podano
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

def load_prompts(path):
    """Wczytuje prompty z pliku. Obsługuje zarówno format JSON, jak i zwykły tekst."""
    logger.info(f"Wczytywanie promptów z pliku: {path}")
    try:
        with open(path, "r", encoding="utf-8") as f:
            prompts = [json.loads(line)["prompt"] if line.strip().startswith("{")
                      else line.strip() for line in f if line.strip()]
        logger.info(f"Wczytano {len(prompts)} promptów")
        return prompts
    except Exception as e:
        logger.error(f"Błąd podczas wczytywania promptów: {e}")
        raise

def split_prompts(prompts, gpu_split_ratio=0.5):
    """Dzieli prompty między GPU według podanego stosunku."""
    split_point = int(len(prompts) * gpu_split_ratio)
    return prompts[:split_point], prompts[split_point:]

def process_batch(llama, prompts_batch, temps, max_tokens, out_file_prefix, gpu_id):
    """Przetwarzanie wsadowe na jednym GPU."""
    logger.info(f"GPU {gpu_id}: Rozpoczęcie przetwarzania {len(prompts_batch)} promptów")
    start_time = time.time()
    
    out_logits, meta = [], []
    
    # Nazwa pliku tymczasowego dla tego GPU
    temp_out_file = f"{out_file_prefix}_gpu{gpu_id}_temp.pt"
    save_interval = max(1, len(list(itertools.product(temps, prompts_batch))) // 10)  # Zapis co ~10% postępu
    
    # Licznik kombinacji
    total_combinations = len(list(itertools.product(temps, prompts_batch)))
    processed = 0
    
    for t, prompt in tqdm.tqdm(list(itertools.product(temps, prompts_batch)), 
                             desc=f"GPU {gpu_id}", position=gpu_id):
        # Tokenizacja wejścia
        try:
            input_tokens = llama.tokenize(prompt.encode('utf-8'))
            
            # Czyszczenie stanu modelu
            llama.reset()

            # Generowanie odpowiedzi z logitami
            response = llama.create_completion(
                prompt, 
                max_tokens=max_tokens, 
                temperature=t,
                logprobs=5
            )
            
            processed += 1
            if processed % 5 == 0:  # Log co 5 promptów dla czytelności
                logger.info(f"GPU {gpu_id} - Przetworzono {processed}/{total_combinations}: '{prompt[:30]}...' (Temp: {t})")
            
            # Pozyskiwanie logitów
            if hasattr(llama, "_ctx") and hasattr(llama, "scores"):
                logits_array = np.array(llama.scores, dtype=np.float32)
                logits_tensor = torch.from_numpy(logits_array)
                
                out_logits.append(logits_tensor)
                meta.append({
                    "prompt": prompt, 
                    "temp": t, 
                    "seq_len": logits_array.shape[0], 
                    "gpu_id": gpu_id,
                    "timestamp": datetime.now().isoformat()
                })
                
                # Zapis pośredni co pewien interwał
                if processed % save_interval == 0:
                    try:
                        torch.save({"logits": out_logits, "meta": meta}, temp_out_file)
                        logger.info(f"GPU {gpu_id} - Zapisano {processed}/{total_combinations} wyników do pliku tymczasowego")
                    except Exception as e:
                        logger.error(f"GPU {gpu_id} - Błąd podczas zapisywania pliku tymczasowego: {e}")
            else:
                logger.warning(f"GPU {gpu_id} - Nie można uzyskać dostępu do logitów dla promptu: '{prompt[:30]}...'")
                
        except Exception as e:
            logger.error(f"GPU {gpu_id} - Błąd podczas przetwarzania promptu '{prompt[:30]}...': {e}")
    
    # Zapis końcowy wyników z tego GPU
    final_out_file = f"{out_file_prefix}_gpu{gpu_id}.pt"
    try:
        if out_logits:
            torch.save({"logits": out_logits, "meta": meta}, final_out_file)
            logger.info(f"GPU {gpu_id} - Zapisano końcowy plik {final_out_file} z {len(out_logits)} wynikami")
            
            if os.path.exists(temp_out_file):
                os.remove(temp_out_file)
                logger.info(f"GPU {gpu_id} - Usunięto plik tymczasowy {temp_out_file}")
        else:
            logger.warning(f"GPU {gpu_id} - Brak wyników do zapisania")
    except Exception as e:
        logger.error(f"GPU {gpu_id} - Błąd podczas zapisywania końcowego pliku: {e}")
    
    elapsed_time = time.time() - start_time
    logger.info(f"GPU {gpu_id} - Zakończono przetwarzanie w {elapsed_time:.2f} sekund")

def main():
    # Parsowanie argumentów linii poleceń
    parser = argparse.ArgumentParser(
        description="Równoległe przetwarzanie modeli LLM na dwóch GPU",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--model", required=True, help="Ścieżka do modelu GGUF")
    parser.add_argument("--prompts", required=True, help="Ścieżka do pliku z promptami")
    parser.add_argument("--temps", nargs="+", type=float, default=[0.7, 1.0, 1.3], 
                      help="Lista temperatur do użycia")
    parser.add_argument("--max_tokens", type=int, default=128, 
                      help="Maksymalna liczba tokenów do wygenerowania")
    parser.add_argument("--n_ctx", type=int, default=2048, 
                      help="Rozmiar kontekstu modelu")
    parser.add_argument("--out", default="logits", 
                      help="Prefiks nazwy pliku wyjściowego")
    parser.add_argument("--log", default=None, 
                      help="Ścieżka do pliku logów (domyślnie: tylko konsola)")
    parser.add_argument("--gpu_split", type=float, default=0.5, 
                      help="Stosunek podziału promptów między GPU (0-1.0)")
    parser.add_argument("--chat_format", default="llama-3", 
                      help="Format chatu (np. llama-3, chatml)")
    parser.add_argument("--combine", action="store_true", 
                      help="Czy łączyć wyniki z obu GPU do jednego pliku")
    parser.add_argument("--n_batch", type=int, default=1024, 
                      help="Rozmiar batcha dla modelu")
    parser.add_argument("--f16_kv", action="store_true", 
                      help="Czy używać half-precision dla kluczy/wartości")
    args = parser.parse_args()

    # Konfiguracja loggera
    global logger
    logger = setup_logger(args.log)
    logger.info(f"Rozpoczęcie przetwarzania z parametrami: {vars(args)}")

    # Sprawdzenie, czy CUDA jest dostępne
    if torch.cuda.is_available():
        logger.info(f"CUDA dostępne: {torch.cuda.device_count()} urządzeń")
        for i in range(torch.cuda.device_count()):
            logger.info(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        logger.error("CUDA niedostępne! Wymagana jest obsługa CUDA.")
        return

    # Utwórz katalog dla pliku wyjściowego, jeśli nie istnieje
    out_dir = os.path.dirname(args.out)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir)
        logger.info(f"Utworzono katalog wyjściowy: {out_dir}")

    # Wczytaj prompty
    try:
        all_prompts = load_prompts(args.prompts)
        if not all_prompts:
            logger.error("Brak promptów do przetwarzania")
            return
    except Exception as e:
        logger.error(f"Nie udało się wczytać promptów: {e}")
        return
    
    # Podziel prompty dla dwóch GPU
    prompts_gpu0, prompts_gpu1 = split_prompts(all_prompts, args.gpu_split)
    logger.info(f"Przydzielono {len(prompts_gpu0)} promptów do GPU 0 i {len(prompts_gpu1)} do GPU 1")

    try:
        # Inicjalizacja modelu na GPU 0
        logger.info("Inicjalizacja modelu na GPU 0...")
        llama_gpu0 = Llama(
            model_path=args.model, 
            n_gpu_layers=-1,      # Wszystkie warstwy na GPU
            n_ctx=args.n_ctx, 
            chat_format=args.chat_format,
            logits_all=True,      # Kluczowe dla logitów
            main_gpu=0,           # Przypisz do GPU 0
            verbose=True,
            n_batch=args.n_batch,
            f16_kv=args.f16_kv    # Half-precision dla kluczy/wartości
        )
        
        # Inicjalizacja modelu na GPU 1
        logger.info("Inicjalizacja modelu na GPU 1...")
        llama_gpu1 = Llama(
            model_path=args.model, 
            n_gpu_layers=-1,
            n_ctx=args.n_ctx, 
            chat_format=args.chat_format,
            logits_all=True,
            main_gpu=1,           # Przypisz do GPU 1
            verbose=True,
            n_batch=args.n_batch,
            f16_kv=args.f16_kv
        )
        
        # Uruchom przetwarzanie równoległe na dwóch GPU
        logger.info("Uruchamianie wątków przetwarzania...")
        
        thread_gpu0 = threading.Thread(
            target=process_batch, 
            args=(llama_gpu0, prompts_gpu0, args.temps, args.max_tokens, args.out, 0)
        )
        
        thread_gpu1 = threading.Thread(
            target=process_batch, 
            args=(llama_gpu1, prompts_gpu1, args.temps, args.max_tokens, args.out, 1)
        )
        
        # Uruchom wątki
        thread_gpu0.start()
        thread_gpu1.start()
        
        # Poczekaj na zakończenie obu wątków
        thread_gpu0.join()
        thread_gpu1.join()
        
        logger.info("Przetwarzanie na obu GPU zakończone.")
        
        # Opcjonalnie - połącz wyniki z obu GPU
        if args.combine:
            try:
                logger.info("Łączenie wyników z obu GPU...")
                results_gpu0 = torch.load(f"{args.out}_gpu0.pt")
                results_gpu1 = torch.load(f"{args.out}_gpu1.pt")
                
                # Sprawdź, czy mamy logity w obu plikach
                if "logits" in results_gpu0 and "logits" in results_gpu1:
                    combined_logits = results_gpu0["logits"] + results_gpu1["logits"]
                    combined_meta = results_gpu0["meta"] + results_gpu1["meta"]
                    
                    combined_file = f"{args.out}_combined.pt"
                    torch.save({"logits": combined_logits, "meta": combined_meta}, combined_file)
                    logger.info(f"Połączono wyniki z obu GPU w pliku {combined_file}")
                else:
                    logger.error("Brak logitów w jednym lub obu plikach wynikowych")
            except Exception as e:
                logger.error(f"Błąd podczas łączenia wyników: {e}")
    
    except Exception as e:
        logger.error(f"Wystąpił błąd podczas wykonywania: {e}")

if __name__ == "__main__":
    main()