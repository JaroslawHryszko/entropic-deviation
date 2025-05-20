#!/usr/bin/env python
# generate.py
import argparse, json, os, itertools, torch, numpy as np, tqdm
from llama_cpp import Llama
import ctypes

def load_prompts(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line)["prompt"] if line.strip().startswith("{")
                else line.strip() for line in f]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)          # GGUF path
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--temps", nargs="+", type=float, default=[0.7,1.0,1.3])
    ap.add_argument("--max_tokens", type=int, default=128)
    ap.add_argument("--out", default="logits.pt")
    args = ap.parse_args()

    # Utwórz katalog dla pliku wyjściowego, jeśli nie istnieje
    out_dir = os.path.dirname(args.out)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # Inicjalizacja modelu z logits_all=True - to kluczowy parametr!
    # Wymusza, aby model obliczał logity dla wszystkich tokenów
    llama = Llama(
        model_path=args.model, 
        n_gpu_layers=-1, 
        n_ctx=args.max_tokens+64, 
        chat_format="llama-3",
        logits_all=True  # Kluczowe ustawienie!
    )

    prompts = load_prompts(args.prompts)
    out_logits, meta = [], []
    
    # Nazwa pliku tymczasowego
    temp_out_file = f"{os.path.splitext(args.out)[0]}_temp.pt"
    
    # Licznik kombinacji
    total_combinations = len(list(itertools.product(args.temps, prompts)))
    processed_combinations = 0

    for t, prompt in tqdm.tqdm(list(itertools.product(args.temps, prompts))):
        # Tokenizacja wejścia
        input_tokens = llama.tokenize(prompt.encode('utf-8'))
        
        # Czyszczenie stanu modelu
        llama.reset()

        # Generowanie odpowiedzi z logitami
        # W tej wersji API używamy logprobs=5 (lub dowolnej innej wartości)
        # co wymusza zachowanie logitów w odpowiedzi
        response = llama.create_completion(
            prompt, 
            max_tokens=args.max_tokens, 
            temperature=t,
            logprobs=5  # Ten parametr jest istotny, ponieważ uruchamia obliczanie logprobs z logitów
        )
        
        # Wydrukowanie struktury odpowiedzi do celów diagnostycznych
        processed_combinations += 1
        print(f"Kombinacja {processed_combinations}/{total_combinations}")
        print(f"Prompt: {prompt[:50]}... Temperatura: {t}")
        print(f"Klucze w odpowiedzi: {response.keys()}")
        
        # Teraz bezpośrednio pozyskujemy tensory logitów z modelu
        # To jest trick - musimy uzyskać dostęp do wewnętrznego obiektu _ctx i scores
        if hasattr(llama, "_ctx") and hasattr(llama, "scores"):
            # Konwersja surowych logitów na tensor PyTorch
            # scores zawiera logity dla tokenów
            logits_array = np.array(llama.scores, dtype=np.float32)
            logits_tensor = torch.from_numpy(logits_array)
            
            # Zapisujemy logity wraz z metadanymi
            out_logits.append(logits_tensor)
            meta.append({"prompt": prompt, "temp": t, "seq_len": logits_array.shape[0]})
            
            print(f"Zapisano logity o kształcie: {logits_array.shape}")
            
            # Zapis pośredni do pliku tymczasowego po każdej kombinacji
            try:
                torch.save({"logits": torch.stack(out_logits), "meta": meta}, temp_out_file)
                print(f"Zapisano tymczasowe dane do {temp_out_file} ({len(out_logits)} sekwencji)")
            except Exception as e:
                print(f"Błąd podczas zapisywania pliku tymczasowego: {e}")
        else:
            print("Nie można uzyskać dostępu do wewnętrznych logitów modelu.")
            print("Dostępne atrybuty:", [attr for attr in dir(llama) if not attr.startswith('_')])

    # Zapis końcowy - wszystkie dane razem
    if out_logits:
        try:
            torch.save({"logits": torch.stack(out_logits), "meta": meta}, args.out)
            print(f"Zapisano końcowy plik {args.out} z {len(out_logits)} sekwencjami")
            
            # Usunięcie pliku tymczasowego, jeśli zapis końcowy się powiódł
            if os.path.exists(temp_out_file):
                os.remove(temp_out_file)
                print(f"Usunięto plik tymczasowy {temp_out_file}")
        except Exception as e:
            print(f"Błąd podczas zapisywania końcowego pliku: {e}")
            print(f"Dane tymczasowe pozostają dostępne w {temp_out_file}")
    else:
        print("Nie udało się pozyskać żadnych logitów.")

if __name__ == "__main__":
    main()