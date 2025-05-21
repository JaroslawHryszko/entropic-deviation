#!/bin/bash

set -euo pipefail

# Katalog docelowy
DEST="/mnt/dysk/skompresowane"

# Tworzenie katalogu, jeśli nie istnieje
mkdir -p "$DEST"

# Zlicz pliki do przetworzenia
count=$(ls logits_*.pt 2>/dev/null | wc -l)
if [[ $count -eq 0 ]]; then
  echo "Brak plików logits_*.pt w bieżącym katalogu."
  exit 1
fi

echo "Rozpoczynam kompresję $count plików..."

# Pętla po plikach
for file in logits_*.pt; do
    dest_file="$DEST/${file}.zst"
    echo "🔄 Kompresuję: $file → $dest_file"

    # Kompresuj z użyciem wielu wątków
    if zstd -T0 "$file" -o "$dest_file"; then
        echo "✅ Sukces, usuwam oryginał: $file"
        rm "$file"
    else
        echo "❌ Błąd kompresji: $file – oryginał NIE został usunięty"
    fi
done

echo "✅ Wszystko gotowe – sprawdź $DEST"
