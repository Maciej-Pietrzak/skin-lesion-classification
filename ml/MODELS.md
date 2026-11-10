# Rejestr wersji modelu

Dokumentacja wytrenowanych wersji klasyfikatora zmian skórnych. Same wagi (`ml/models/*.pt`,
`*.onnx`) są gitignorowane, więc to ten plik łączy checkpoint z konfiguracją treningu i wynikami.

## Podsumowanie

| Wersja | Data       | Architektura    | Test balanced acc. | Test accuracy | Test macro F1 | F1 `mel` | Recall `mel` | Status                    |
| ------ | ---------- | --------------- | ------------------ | ------------- | ------------- | -------- | ------------ | ------------------------- |
| v1     | 2026-07-30 | EfficientNet-B0 | 0.665              | 0.738         | 0.601         | 0.482    | 0.558        | bazowy, używany w backendzie |

Główna metryka od v2: **balanced accuracy** (średni recall po klasach, oficjalna metryka ISIC
2018) — według niej trening wybiera checkpoint. Te same liczby w formie do Excela/pandas:
`ml/results/models.csv`.

---

## v1 — model bazowy

### Identyfikacja

| | |
| --- | --- |
| Plik | `ml/models/efficientnet_b0_ham10000_v1.pt` (state_dict, ~16 MB; do 2026-11-10 jako `efficientnet_b0_ham10000.pt`) |
| Data treningu | 2026-07-30 (checkpoint zapisany 21:53) |
| Kod źródłowy | commit `0089e60` („Add ML pipeline: dataset, EfficientNet-B0 transfer learning, Grad-CAM”) |
| Eksport ONNX | `ml/models/efficientnet_b0_ham10000.onnx` + `.onnx.data` (2026-08-30, `ml/src/export_onnx.py`; stara nazwa bez `_v1` — plik `.onnx` ma zapisaną nazwę pliku z wagami, więc nie zmieniamy jej) — tylko do wizualizacji architektury |
| Status | model bazowy, ładowany przez backend do `POST /predict` |

### Dane

- Zbiór: HAM10000 (`kmader/skin-cancer-mnist-ham10000` z Kaggle), licencja CC BY-NC 4.0
- 7 klas: `akiec`, `bcc`, `bkl`, `df`, `mel`, `nv`, `vasc`
- Podział **po `lesion_id`** (wszystkie zdjęcia jednej zmiany trafiają do tego samego zbioru —
  brak wycieku danych), stratyfikowany po `dx`, proporcje 70/15/15, `random_state=42`
  (`ml/src/split_dataset.py` → `ml/data/splits.csv`)
- Liczności: train 7002 / val 1532 / test 1481 zdjęć
- Silna nierównowaga klas: `nv` ~67%, `df` i `vasc` ~1%

### Architektura i preprocessing

- EfficientNet-B0, wagi `IMAGENET1K_V1` (torchvision)
- Głowica: `Dropout(0.2)` + `Linear(1280, 7)` w miejsce oryginalnej 1000-klasowej (`ml/src/model.py`)
- Wejście: resize do 224×224, normalizacja statystykami ImageNet
- Augmentacje (tylko train): poziome i pionowe odbicie, rotacja ±20°,
  `ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1)` (`ml/src/dataloaders.py`)
- Nierównowaga klas: `WeightedRandomSampler` z wagą 1/liczność klasy; funkcja straty to zwykła
  `CrossEntropyLoss` bez wag klas (żeby nie korygować nierównowagi podwójnie)

### Trening

Dwuetapowy (`ml/src/train.py`), optymalizator Adam, batch size 32. Checkpoint wybierany wg
najlepszego **val macro F1**; etap 2 nadpisuje go tylko, jeśli poprawia wynik etapu 1.

| Etap | Co trenowane | lr | Epoki | Najlepszy val macro F1 |
| --- | --- | --- | --- | --- |
| 1 — liniowa sonda | tylko głowica (backbone zamrożony) | 1e-3 | 15 (najlepsza: 10.) | 0.4602 |
| 2 — fine-tuning | cała sieć, start z checkpointu etapu 1 | 1e-5 | 10 | 0.6043 |

Uwaga: dla tego treningu nie ma logu epok — `ml/results/training_log.csv` dodano później
(commit `92d4e51`). Wartości w tabeli pochodzą z notatek z 2026-07-30. Liczba epok to wartości
domyślne z `train.py`; nie da się już sprawdzić dokładnej liczby epok fine-tuningu.

### Wyniki na zbiorze testowym

Wynik `python -m ml.src.evaluate` (uruchomione ponownie 2026-11-10, liczby zgodne z notatkami):

| Klasa | Precision | Recall | F1 | Liczba przykładów |
| --- | --- | --- | --- | --- |
| akiec | 0.382 | 0.630 | 0.475 | 46 |
| bcc | 0.495 | 0.746 | 0.596 | 71 |
| bkl | 0.515 | 0.696 | 0.592 | 168 |
| df | 0.450 | 0.450 | 0.450 | 20 |
| **mel** | **0.424** | **0.558** | **0.482** | 165 |
| nv | 0.957 | 0.784 | 0.862 | 992 |
| vasc | 0.714 | 0.789 | 0.750 | 19 |
| **accuracy** | | | **0.738** | 1481 |
| **macro avg** | 0.563 | 0.665 | **0.601** | 1481 |
| **balanced accuracy** | | | **0.665** | 1481 |
| weighted avg | 0.798 | 0.738 | 0.757 | 1481 |

Val macro F1 (0.604) i test macro F1 (0.601) są niemal identyczne, więc model nie przeuczył się
na zbiorze walidacyjnym.

Macierz pomyłek (wiersze = klasa prawdziwa, kolumny = predykcja):

| prawda \ pred | akiec | bcc | bkl | df | mel | nv | vasc |
| --- | --- | --- | --- | --- | --- | --- | --- |
| akiec | **29** | 7 | 8 | 0 | 2 | 0 | 0 |
| bcc | 10 | **53** | 2 | 2 | 2 | 1 | 1 |
| bkl | 9 | 4 | **117** | 5 | 21 | 12 | 0 |
| df | 5 | 5 | 0 | **9** | 0 | 1 | 0 |
| mel | 16 | 5 | 33 | 0 | **92** | 19 | 0 |
| nv | 7 | 31 | 67 | 4 | 100 | **778** | 5 |
| vasc | 0 | 2 | 0 | 0 | 0 | 2 | **15** |

### Znane słabości

- **Czerniak (`mel`) wypada słabo: F1 0.482, recall 0.558.** Model nie rozpoznaje 73 ze 165
  czerniaków: 33 klasyfikuje jako `bkl`, 19 jako `nv` i 16 jako `akiec`. Dla narzędzia
  przeznaczonego dla personelu medycznego przeoczony czerniak to najgroźniejszy błąd. Szczególnie
  groźne są pomyłki z `nv` i `bkl`, czyli zmianami łagodnymi.
- Niska precyzja `mel` (0.424) wynika głównie z tego, że 100 znamion (`nv`) jest
  klasyfikowanych jako czerniak. To mniej groźne klinicznie (fałszywy alarm), ale obniża
  wiarygodność predykcji.
- Accuracy (73.8%) jest tylko trochę wyższe od modelu, który zawsze odpowiada `nv` (~67%).
  Dlatego główną metryką jest macro F1.
- Fine-tuning się nie zbiegł: val F1 rosło jeszcze w ostatniej epoce.
- Najrzadsze klasy (`df`: 20, `vasc`: 19 przykładów testowych) mają szerokie przedziały
  niepewności, więc ich metryki są mało stabilne.

### Pomysły na v2

- Dłuższy fine-tuning (val F1 wciąż rosło), ewentualnie z harmonogramem lr (np. cosine) i early
  stoppingiem
- Poprawa wykrywania `mel`: większa waga klasy w stracie albo focal loss, próg decyzyjny dla
  `mel` dobrany na walidacji pod wymagany recall
- Wyższa rozdzielczość wejścia lub większy wariant EfficientNet (B1–B3)
- Mocniejsze augmentacje (RandomResizedCrop, kontrolowane zmiany kolorów)
- Dodatkowe metryki w ewaluacji: balanced accuracy, ROC-AUC dla każdej klasy

---

## Jak dodać nową wersję

Każda wersja ma własny plik `ml/models/efficientnet_b0_ham10000_<wersja>.pt`; trening odmawia
startu, jeśli plik o tej nazwie już istnieje, więc wcześniejsze modele nigdy nie są nadpisywane.

1. Wytrenuj model: `python -m ml.src.train --version vN`. Log każdej epoki (z kolumną `version`)
   trafia do `ml/results/training_log.csv`.
2. Oceń go na zbiorze testowym: `python -m ml.src.evaluate --version vN --notes "co się zmieniło"`.
   Wiersz z wynikami trafia do `ml/results/models.csv` (ponowne uruchomienie dla tej samej wersji
   nadpisuje jej wiersz, nie dubluje go).
3. Dodaj wiersz do tabeli [Podsumowanie](#podsumowanie) i sekcję `## vN` w tym pliku: commit
   kodu, zmiany względem poprzedniej wersji, konfiguracja, wyniki, słabości.
4. Jeśli nowa wersja ma zastąpić poprzednią w aplikacji: zmień `MODEL_CHECKPOINT_PATH` w
   `backend/.env` (i domyślną wartość w `backend/app/core/config.py` oraz `.env.example`) oraz
   `DEFAULT_VERSION` w `ml/src/registry.py`. ONNX (`python -m ml.src.export_onnx`) eksportujemy
   tylko dla modelu finalnego.
