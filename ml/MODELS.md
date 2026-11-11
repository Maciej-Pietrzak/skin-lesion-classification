# Rejestr wersji modelu

Dokumentacja wytrenowanych wersji klasyfikatora zmian skórnych. Same wagi (`ml/models/*.pt`,
`*.onnx`) są gitignorowane, więc to ten plik łączy checkpoint z konfiguracją treningu i wynikami.

## Podsumowanie

| Wersja | Data       | Architektura    | Test balanced acc. | Test accuracy | Test macro F1 | F1 `mel` | Recall `mel` | Status                    |
| ------ | ---------- | --------------- | ------------------ | ------------- | ------------- | -------- | ------------ | ------------------------- |
| v1     | 2026-07-30 | EfficientNet-B0 | 0.665              | 0.738         | 0.601         | 0.482    | 0.558        | bazowy                   |
| v2     | 2026-11-10 | EfficientNet-B0 | 0.710              | 0.793         | 0.687         | **0.560** | 0.636        | poprzedni                 |
| v3     | 2026-11-10 | EfficientNet-B0 | **0.752**          | **0.824**     | **0.732**     | 0.552    | 0.612        | używany w backendzie      |
| v4     | 2026-11-11 | EfficientNet-B0 | 0.747              | 0.797         | 0.704         | 0.538    | **0.661**    | eksperyment (waga `mel`)  |

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
| Status | model bazowy; do 2026-11-10 ładowany przez backend do `POST /predict`, potem zastąpiony przez v2 |

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

## v2 — dłuższy fine-tuning z cosine LR i early stoppingiem

### Identyfikacja

| | |
| --- | --- |
| Plik | `ml/models/efficientnet_b0_ham10000_v2.pt` (state_dict, ~16 MB) |
| Data treningu | 2026-11-10, 20:24–21:29 (checkpoint zapisany 21:08, po 6. epoce fine-tuningu) |
| Kod źródłowy | `ml/src/train.py` w commicie, który dodał tę sekcję (podczas treningu zmiany nie były jeszcze w commicie); polecenie: `python -m ml.src.train --version v2` (wartości domyślne) |
| Status | od 2026-11-10 do 2026-11-11 ładowany przez backend do `POST /predict`, potem zastąpiony przez v3 |

### Zmiany względem v1

Dane, architektura, preprocessing, augmentacje i sampler są takie same jak w v1. Zmienił się tylko
trening:

- Checkpoint jest wybierany według **val balanced accuracy**, a nie val macro F1.
- Fine-tuning (etap 2) ma lr **1e-4** zamiast 1e-5. v1 uczył się za wolno: val F1 jeszcze rosło
  w ostatniej epoce.
- Do etapu 2 dodano **cosine annealing** lr (`CosineAnnealingLR`, `T_max=30`). Lr maleje płynnie
  od 1e-4 do 0: duże kroki na początku, drobne poprawki na końcu.
- Etap 2 ma maksymalnie **30 epok** i **early stopping** z `patience=6`.
- Wszystkie generatory losowe mają stały seed (**42**). Powtórzony trening da bardzo podobny
  wynik, ale przez niedeterminizm GPU nie zawsze identyczny co do bitu.

### Trening

Adam, batch size 32. Pełny log epok: `ml/results/training_log.csv` (wiersze z `version=v2`).

| Etap | Co trenowane | lr | Epoki | Najlepsza epoka | Val balanced acc. | Val macro F1 |
| --- | --- | --- | --- | --- | --- | --- |
| 1 — liniowa sonda | tylko głowica | 1e-3 (stały) | 15 | 13. | 0.613 | 0.471 |
| 2 — fine-tuning | cała sieć | 1e-4 → cosine | 12 z 30 | 6. | **0.700** | 0.665 |

Jedna epoka trwała ok. 1,7 min w etapie 1 i ok. 3,3 min w etapie 2.

**Przeuczenie w etapie 2.** Od 7. epoki fine-tuningu train balanced accuracy rosło z 0.93 do 0.97,
a val balanced accuracy spadało z 0.70 do 0.65. Val recall `mel` przestał rosnąć (0.55–0.59).
Trening przerwano ręcznie po 12. epoce. To ten sam moment, w którym zatrzymałby go early stopping
(6 epok bez poprawy). Checkpoint zapisuje się tylko przy poprawie wyniku val, więc plik `v2.pt`
ma wagi z 6. epoki, sprzed przeuczenia.

### Wyniki na zbiorze testowym

Wynik `python -m ml.src.evaluate --version v2` (2026-11-10):

| Klasa | Precision | Recall | F1 | F1 w v1 | Liczba przykładów |
| --- | --- | --- | --- | --- | --- |
| akiec | 0.475 | 0.630 | 0.542 | 0.475 | 46 |
| bcc | 0.609 | 0.746 | 0.671 | 0.596 | 71 |
| bkl | 0.607 | 0.726 | 0.661 | 0.592 | 168 |
| df | 0.812 | 0.650 | 0.722 | 0.450 | 20 |
| **mel** | **0.500** | **0.636** | **0.560** | 0.482 | 165 |
| nv | 0.945 | 0.846 | 0.893 | 0.862 | 992 |
| vasc | 0.778 | 0.737 | 0.757 | 0.750 | 19 |
| **accuracy** | | | **0.793** | 0.738 | 1481 |
| **macro avg** | 0.675 | 0.710 | **0.687** | 0.601 | 1481 |
| **balanced accuracy** | | | **0.710** | 0.665 | 1481 |
| weighted avg | 0.822 | 0.793 | 0.804 | 0.757 | 1481 |

Val balanced accuracy (0.700) i test balanced accuracy (0.710) są niemal identyczne, więc wybrany
checkpoint dobrze uogólnia. F1 poprawił się we wszystkich klasach; `vasc` wyszedł praktycznie bez
zmian.

Macierz pomyłek (wiersze = klasa prawdziwa, kolumny = predykcja):

| prawda \ pred | akiec | bcc | bkl | df | mel | nv | vasc |
| --- | --- | --- | --- | --- | --- | --- | --- |
| akiec | **29** | 5 | 10 | 0 | 1 | 1 | 0 |
| bcc | 6 | **53** | 5 | 0 | 3 | 4 | 0 |
| bkl | 9 | 6 | **122** | 2 | 12 | 17 | 0 |
| df | 3 | 3 | 0 | **13** | 0 | 1 | 0 |
| mel | 11 | 1 | 24 | 0 | **105** | 24 | 0 |
| nv | 3 | 18 | 40 | 1 | 87 | **839** | 4 |
| vasc | 0 | 1 | 0 | 0 | 2 | 2 | **14** |

### Znane słabości

- **Czerniak (`mel`) wciąż jest najsłabszą klasą o znaczeniu klinicznym: F1 0.560, recall 0.636.**
  Model przeocza 60 ze 165 czerniaków (w v1: 73). Klasyfikuje je jako `bkl` (24), `nv` (24) i
  `akiec` (11). Pomyłek `mel` → `nv` jest więcej niż w v1 (24 zamiast 19), a to najgroźniejszy
  błąd: czerniak zostaje uznany za zwykłe znamię.
- 87 znamion (`nv`) jest klasyfikowanych jako czerniak (w v1: 100), stąd precyzja `mel` wynosi
  tylko 0.500.
- Szybkie przeuczenie w fine-tuningu: po 6 epokach model zaczyna zapamiętywać zbiór treningowy.
  Obecne augmentacje i regularyzacja (tylko `Dropout(0.2)`) nie wystarczają przy lr 1e-4.
- Duży skok `df` (F1 0.45 → 0.72) opiera się na 20 przykładach testowych, więc może wynikać
  w dużej części z przypadku.

### Pomysły na v3

- Walka z przeuczeniem: mocniejsze augmentacje (`RandomResizedCrop`, silniejszy `ColorJitter`),
  weight decay (AdamW), większy dropout, mniejsze `patience` (np. 3)
- Poprawa wykrywania `mel`: większa waga klasy w stracie albo focal loss; próg decyzyjny dla `mel`
  dobrany na walidacji pod wymagany recall
- Wyższa rozdzielczość wejścia lub większy wariant EfficientNet (B1–B3)

---

## v3 — mocniejsza augmentacja

### Identyfikacja

| | |
| --- | --- |
| Plik | `ml/models/efficientnet_b0_ham10000_v3.pt` (state_dict, ~16 MB) |
| Data treningu | 2026-11-10, 22:09–23:59 (checkpoint zapisany 23:40, po 17. epoce fine-tuningu) |
| Kod źródłowy | `ml/src/dataloaders.py` w commicie, który dodał tę sekcję; polecenie: `python -m ml.src.train --version v3` (wartości domyślne, jak w v2) |
| Status | ładowany przez backend do `POST /predict` (od 2026-11-11) |

### Zmiany względem v2

Zmieniła się **tylko augmentacja zbioru treningowego** (`build_transforms(train=True)` w
`ml/src/dataloaders.py`). Trening, architektura, sampler, seed oraz przetwarzanie val/test są takie
same jak w v2, więc różnica v2 → v3 pokazuje efekt samej augmentacji.

| | v1, v2 | v3 |
| --- | --- | --- |
| Skalowanie | `Resize(224, 224)` | `RandomResizedCrop(224, scale=(0.6, 1.0), ratio=(0.75, 1.33))` |
| Odbicia | poziome i pionowe | bez zmian |
| Rotacja | ±20° | ±45° |
| Kolor | `ColorJitter(0.1, 0.1, 0.1)` | `ColorJitter(0.2, 0.2, 0.2, hue=0.02)` |

Cel: ograniczyć przeuczenie z v2. Sampler pokazuje rzadkie zdjęcia (`df`, `vasc`) wiele razy w
epoce, a przy słabej augmentacji model je zapamiętywał. `RandomResizedCrop` ma `scale ≥ 0.6`, żeby
zmiana (zwykle na środku kadru) zostawała w obrazie. `hue` jest małe, bo kolor jest cechą
diagnostyczną.

### Trening

Adam, batch size 32, ustawienia jak w v2. Pełny log epok: `ml/results/training_log.csv` (wiersze
z `version=v3`).

| Etap | Co trenowane | lr | Epoki | Najlepsza epoka | Val balanced acc. | Val macro F1 |
| --- | --- | --- | --- | --- | --- | --- |
| 1 — liniowa sonda | tylko głowica | 1e-3 (stały) | 15 | 11. | 0.609 | 0.462 |
| 2 — fine-tuning | cała sieć | 1e-4 → cosine | 23 z 30 | 17. | **0.721** | 0.702 |

Jedna epoka trwała ok. 2–2,5 min w etapie 1 i ok. 3–5 min w etapie 2, dłużej niż w v2, bo
augmentacja na zdjęciach 600×450 jest liczona na CPU.

**Przeuczenie przyszło później niż w v2.** Różnica train−val w balanced accuracy po 2. epoce
fine-tuningu wynosiła 0.12 (v2: 0.16), a po 8. epoce 0.24 (v2: 0.28). Wynik val wahał się w
epokach 7–19 w przedziale 0.66–0.72, ale nie spadał systematycznie i rekordy padały jeszcze w 13.
i 17. epoce. W v2 najlepsza była 6. epoka. Trening zatrzymał early stopping po 23. epoce
(6 epok bez poprawy); tym razem nie trzeba było przerywać ręcznie.

### Wyniki na zbiorze testowym

Wynik `python -m ml.src.evaluate --version v3` (2026-11-11):

| Klasa | Precision | Recall | F1 | F1 w v2 | Liczba przykładów |
| --- | --- | --- | --- | --- | --- |
| akiec | 0.564 | 0.674 | 0.614 | 0.542 | 46 |
| bcc | 0.663 | 0.803 | 0.726 | 0.671 | 71 |
| bkl | 0.711 | 0.631 | 0.669 | 0.661 | 168 |
| df | 0.810 | 0.850 | 0.829 | 0.722 | 20 |
| **mel** | **0.502** | **0.612** | **0.552** | 0.560 | 165 |
| nv | 0.940 | 0.901 | 0.920 | 0.893 | 992 |
| vasc | 0.833 | 0.789 | 0.811 | 0.757 | 19 |
| **accuracy** | | | **0.824** | 0.793 | 1481 |
| **macro avg** | 0.718 | 0.751 | **0.732** | 0.687 | 1481 |
| **balanced accuracy** | | | **0.752** | 0.710 | 1481 |
| weighted avg | 0.837 | 0.824 | 0.829 | 0.804 | 1481 |

Test balanced accuracy (0.752) jest nawet wyższa niż val (0.721), więc wybrany checkpoint nie jest
przeuczony pod zbiór walidacyjny. F1 poprawił się w 6 z 7 klas; wyjątkiem jest `mel`.

Macierz pomyłek (wiersze = klasa prawdziwa, kolumny = predykcja):

| prawda \ pred | akiec | bcc | bkl | df | mel | nv | vasc |
| --- | --- | --- | --- | --- | --- | --- | --- |
| akiec | **31** | 7 | 6 | 0 | 1 | 1 | 0 |
| bcc | 8 | **57** | 2 | 0 | 1 | 2 | 1 |
| bkl | 7 | 2 | **106** | 2 | 33 | 17 | 1 |
| df | 0 | 2 | 0 | **17** | 0 | 1 | 0 |
| mel | 8 | 1 | 21 | 1 | **101** | 33 | 0 |
| nv | 1 | 17 | 14 | 1 | 64 | **894** | 1 |
| vasc | 0 | 0 | 0 | 0 | 1 | 3 | **15** |

### Znane słabości

- **Czerniak (`mel`) nie skorzystał na augmentacji: recall 0.612 (v2: 0.636), F1 0.552 (v2:
  0.560).** Model przeocza 64 ze 165 czerniaków (v2: 60). **Pomyłek `mel` → `nv` jest 33 (v2: 24)**,
  a to najgroźniejszy błąd: czerniak uznany za zwykłe znamię.
- W zamian jest mniej fałszywych alarmów `nv` → `mel`: 64 (v2: 87). Granica między `mel` a `nv`
  przesunęła się w stronę `nv`. Checkpoint jest wybierany według balanced accuracy, czyli średniej
  ze wszystkich klas, więc sama ta metryka nie chroni czerniaka.
- 33 zmiany `bkl` są klasyfikowane jako `mel` (v2: 12), stąd recall `bkl` spadł do 0.631.
- Model nadal się przeucza (train balanced accuracy 0.96–0.97 pod koniec), tylko później niż v2.
  Wynik val mocno skacze między epokami (±0.02–0.03).
- `df` i `vasc` mają po ok. 20 przykładów testowych, więc ich poprawa jest mało pewna.

### Pomysły na v4

- Próg decyzyjny dla `mel` dobrany na walidacji pod wymagany recall (bez ponownego treningu)
- Większa waga `mel` w stracie albo focal loss; ewentualnie wybór checkpointu z uwzględnieniem
  recall `mel`
- Dalsza regularyzacja: AdamW z weight decay, większy dropout
- Wyższa rozdzielczość wejścia lub większy wariant EfficientNet (B1–B3)

---

## v4 — większa waga czerniaka w funkcji straty

### Identyfikacja

| | |
| --- | --- |
| Plik | `ml/models/efficientnet_b0_ham10000_v4.pt` (state_dict, ~16 MB) |
| Data treningu | 2026-11-11, 16:01–17:05 (checkpoint zapisany 16:50, po 7. epoce fine-tuningu) |
| Kod źródłowy | `ml/src/train.py` (`build_criterion()`) w commicie, który dodał tę sekcję; polecenie: `python -m ml.src.train --version v4` (wartości domyślne, `--mel-weight 2.0`) |
| Status | eksperyment; backend zostaje przy v3 |

### Zmiany względem v3

Zmieniła się **tylko funkcja straty**: `CrossEntropyLoss` z wagami klas `[1, 1, 1, 1, 2, 1, 1]`,
czyli błąd na zdjęciu, którego prawdziwa klasa to `mel`, kosztuje 2× więcej (oba etapy treningu).
Augmentacja, sampler, optymalizator, harmonogram lr i seed są jak w v3.

To nie jest korekta nierównowagi klas, bo tę robi już `WeightedRandomSampler`. To celowe
przesunięcie modelu na korzyść czerniaka, bo przeoczony czerniak to najgroźniejszy błąd. Waga
działa według prawdziwej klasy, więc karze przeoczenie (`mel` → `nv`), a nie fałszywy alarm.

Skutek uboczny: `val_loss` jest liczony z tą samą wagą, więc w `training_log.csv` nie da się go
porównać z v1–v3. Metryki (balanced accuracy, F1, mel recall) są porównywalne. Od v4 log ma
kolumnę `mel_weight` (dla v2 i v3 uzupełnioną wartością 1.0).

Pierwsza próba treningu została przerwana w 14. epoce etapu 1, bo w systemie zabrakło pamięci
RAM. Częściowy checkpoint i wiersze logu usunięto, a trening uruchomiono od nowa. Liczby z obu
prób były identyczne co do 4. miejsca po przecinku, więc stały seed daje w pełni powtarzalny
trening.

### Trening

Adam, batch size 32, ustawienia jak w v3. Pełny log epok: `ml/results/training_log.csv` (wiersze
z `version=v4`).

| Etap | Co trenowane | lr | Epoki | Najlepsza epoka | Val balanced acc. | Val macro F1 | Val mel recall |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 — liniowa sonda | tylko głowica | 1e-3 (stały) | 15 | 14. | 0.593 | 0.470 | 0.600 |
| 2 — fine-tuning | cała sieć | 1e-4 → cosine | 13 z 30 | 7. | **0.708** | 0.654 | 0.624 |

Jedna epoka trwała ok. 2,3 min w obu etapach (szybciej niż v3, bo było więcej wolnej pamięci).

**Mel recall na walidacji był w trakcie treningu wyraźnie wyższy niż w v3:** w epokach 1–6
fine-tuningu 0.62–0.73 (v3: 0.42–0.59). Checkpoint jest jednak wybierany według balanced accuracy,
a najlepsza pod tym względem była 7. epoka z mel recall 0.624. Epoki 4. (0.729) i 13. (0.700)
miały lepszy recall czerniaka przy balanced accuracy niższej o 0.01–0.04. Early stopping
zatrzymał trening po 13. epoce, dużo wcześniej niż v3 (23. epoka, rekordy jeszcze w 13. i 17.).

### Wyniki na zbiorze testowym

Wynik `python -m ml.src.evaluate --version v4` (2026-11-11):

| Klasa | Precision | Recall | F1 | F1 w v3 | Liczba przykładów |
| --- | --- | --- | --- | --- | --- |
| akiec | 0.524 | 0.717 | 0.606 | 0.614 | 46 |
| bcc | 0.675 | 0.789 | 0.727 | 0.726 | 71 |
| bkl | 0.660 | 0.625 | 0.642 | 0.669 | 168 |
| df | 0.654 | 0.850 | 0.739 | 0.829 | 20 |
| **mel** | **0.454** | **0.661** | **0.538** | 0.552 | 165 |
| nv | 0.947 | 0.853 | 0.898 | 0.920 | 992 |
| vasc | 0.824 | 0.737 | 0.778 | 0.811 | 19 |
| **accuracy** | | | **0.797** | 0.824 | 1481 |
| **macro avg** | 0.677 | 0.747 | **0.704** | 0.732 | 1481 |
| **balanced accuracy** | | | **0.747** | 0.752 | 1481 |
| weighted avg | 0.828 | 0.797 | 0.808 | 0.829 | 1481 |

Macierz pomyłek (wiersze = klasa prawdziwa, kolumny = predykcja):

| prawda \ pred | akiec | bcc | bkl | df | mel | nv | vasc |
| --- | --- | --- | --- | --- | --- | --- | --- |
| akiec | **33** | 7 | 5 | 0 | 1 | 0 | 0 |
| bcc | 5 | **56** | 5 | 0 | 2 | 2 | 1 |
| bkl | 11 | 3 | **105** | 4 | 30 | 15 | 0 |
| df | 1 | 1 | 0 | **17** | 0 | 1 | 0 |
| mel | 11 | 1 | 17 | 1 | **109** | 26 | 0 |
| nv | 2 | 14 | 27 | 4 | 97 | **846** | 2 |
| vasc | 0 | 1 | 0 | 0 | 1 | 3 | **14** |

Czerniak w porównaniu z poprzednimi wersjami:

| | v2 | v3 | v4 |
| --- | --- | --- | --- |
| Przeoczone `mel` (ze 165) | 60 | 64 | **56** |
| `mel` → `nv` | **24** | 33 | 26 |
| Fałszywe alarmy `nv` → `mel` | 87 | **64** | 97 |

### Wnioski

- Waga `mel` robi to, czego się spodziewano: najwyższy recall czerniaka ze wszystkich wersji
  (0.661) i mniej pomyłek `mel` → `nv` niż w v3 (26 zamiast 33). Kosztem jest 33 fałszywych
  alarmów `nv` → `mel` więcej, a precyzja `mel` spadła do 0.454.
- Ogólnie model nie jest lepszy: balanced accuracy praktycznie bez zmian (0.747 vs 0.752,
  w granicach losowości), macro F1 i accuracy niższe o ok. 0.03. Najbardziej straciły `df`
  (F1 0.83 → 0.74, ale tylko 20 przykładów), `bkl` i `nv`.
- Efekt przypomina obniżenie progu decyzyjnego dla `mel`, tylko wbudowane w trening. Próg da
  podobne przesunięcie bez ponownego treningu i bez wpływu na to, czego model się uczy, a do
  tego można go ustawić dokładnie pod wymagany recall. Dlatego kolejne wersje wracają do
  `mel_weight=1.0`, a recall czerniaka zostanie dostrojony progiem na finalnym modelu.
- Backend zostaje przy v3: według zasady wyboru na walidacji v3 jest lepszy (val balanced
  accuracy 0.721 vs 0.708) i ma lepsze ogólne metryki na teście.
- Wybór checkpointu według balanced accuracy gubi część zysku z wagi `mel` (lepsze dla czerniaka
  epoki 4. i 13. nie zostały wybrane).

### Pomysły na v5

- Większy wariant sieci: EfficientNet-B2 (natywnie 260×260), z ustawieniami v3
  (`--mel-weight 1.0`)
- Przed B2: szybsze ładowanie danych (`num_workers > 0`), bo augmentacja jest liczona na CPU
- Na koniec, dla finalnego modelu: walidacja krzyżowa (5-fold, `StratifiedGroupKFold` po
  `lesion_id`), próg dla `mel` na predykcjach out-of-fold, ensemble modeli z foldów

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
