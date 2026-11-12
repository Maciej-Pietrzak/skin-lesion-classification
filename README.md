# System wspomagania klasyfikacji zmian skórnych

Praca inżynierska — system wspomagający klasyfikację zmian skórnych na obrazach dermatoskopowych
z wykorzystaniem głębokich sieci neuronowych (transfer learning, EfficientNet), udostępniany
personelowi medycznemu poprzez aplikację webową.

> ⚠️ Projekt akademicki — **nie jest wyrobem medycznym** i nie służy do diagnozy. Wynik modelu
> ma charakter wyłącznie pomocniczy.

**Stack:** PyTorch · EfficientNet (torchvision) · Grad-CAM · FastAPI · PostgreSQL · Alembic ·
React · TypeScript · Vite · Tailwind

## Funkcje

- Klasyfikacja zdjęcia dermatoskopowego do 7 klas HAM10000 (`akiec`, `bcc`, `bkl`, `df`, `mel`,
  `nv`, `vasc`) z prawdopodobieństwami dla każdej klasy
- Nakładka **Grad-CAM** pokazująca obszary obrazu, na których model oparł decyzję
- Historia przypadków (zapis w PostgreSQL): lista, szczegóły, usuwanie
- Wersjonowane modele z udokumentowaną konfiguracją i wynikami ([`ml/MODELS.md`](ml/MODELS.md))

## Wyniki modelu

Zbiór testowy HAM10000 (1481 zdjęć, podział po `lesion_id`, bez wycieku danych). Główna metryka:
**balanced accuracy** (średni recall po klasach, oficjalna metryka ISIC 2018).

| Wersja | Architektura    | Balanced acc. | Accuracy | Macro F1 | Recall `mel` | Uwagi                         |
| ------ | --------------- | ------------- | -------- | -------- | ------------ | ----------------------------- |
| v1     | EfficientNet-B0 | 0.665         | 0.738    | 0.601    | 0.558        | model bazowy                  |
| v3     | EfficientNet-B0 | 0.752         | **0.824**| **0.732**| 0.612        | **używany w backendzie**      |
| v4     | EfficientNet-B0 | 0.747         | 0.797    | 0.704    | **0.661**    | większa waga klasy `mel`      |
| v7     | EfficientNet-B0 | **0.756**     | 0.780    | 0.695    | 0.636        | native-resolution crop + TTA  |

Pełna lista wersji (v1–v7), konfiguracje treningu, macierze pomyłek i znane słabości (głównie
rozróżnianie `mel` / `nv`): [`ml/MODELS.md`](ml/MODELS.md), surowe liczby:
[`ml/results/models.csv`](ml/results/models.csv).

## Struktura repozytorium

- `ml/` — przygotowanie danych, trening i ewaluacja modelu klasyfikacyjnego (PyTorch)
- `backend/` — REST API (FastAPI) obsługujące inferencję modelu i historię przypadków (PostgreSQL)
- `frontend/` — aplikacja webowa (React) dla personelu medycznego

## Uruchomienie (backend)

### Wymagania

- Python 3.12
- Docker (baza danych PostgreSQL)
- Wytrenowany checkpoint modelu pod `ml/models/efficientnet_b0_ham10000_v3.pt` — plik jest
  gitignorowany, więc trzeba go samodzielnie wytrenować (patrz sekcja
  [Trening modelu](#trening-modelu-opcjonalnie)) albo skopiować z innego źródła

### Backend + baza danych

Wszystkie polecenia z katalogu głównego repozytorium, chyba że zaznaczono inaczej.

1. Środowisko wirtualne i zależności:

   ```bash
   python -m venv .venv
   .venv\Scripts\activate        # Windows; na Linux/macOS: source .venv/bin/activate
   pip install -r backend/requirements.txt
   ```

2. Baza danych — kontener PostgreSQL zgodny z domyślnymi wartościami w `.env.example`:

   ```bash
   docker run -d --name skin-lesion-db -p 5432:5432 \
     -e POSTGRES_USER=skinlesion -e POSTGRES_PASSWORD=devpassword \
     -e POSTGRES_DB=skin_lesion_db postgres:16
   ```

   Przy kolejnych uruchomieniach wystarczy `docker start skin-lesion-db`.

3. Konfiguracja: skopiuj `backend/.env.example` do `backend/.env` (domyślne wartości pasują do
   kontenera powyżej — do zmiany tylko jeśli masz inną konfigurację lokalną).

4. Migracje bazy danych:

   ```bash
   cd backend
   alembic upgrade head
   cd ..
   ```

5. Start API:

   ```bash
   uvicorn backend.app.main:app --reload
   ```

   - `http://127.0.0.1:8000/docs` — interaktywna dokumentacja Swagger
   - `http://127.0.0.1:8000/ui` — prowizoryczna strona testowa do `POST /predict` (nie docelowy
     frontend)

### Trening modelu (opcjonalnie)

Potrzebne tylko, jeśli nie masz gotowego checkpointu. Dodatkowe zależności (dataset, wykresy):

```bash
pip install -r ml/requirements.txt
```

Następnie: `ml/src/download_dataset.py` (pobiera HAM10000 z Kaggle) → `ml/src/split_dataset.py`
(dzieli po `lesion_id`) → `python -m ml.src.train --version v3 --arch efficientnet_b0 --no-amp`
(dwuetapowy trening: liniowa sonda, potem fine-tuning; bez flag `train.py` trenuje według
ustawień najnowszej wersji, zob. `ml/MODELS.md`; zapisuje checkpoint do
`ml/models/efficientnet_b0_ham10000_v3.pt` i log
epok do `ml/results/training_log.csv`) → `python -m ml.src.evaluate --version v3` (wyniki na
zbiorze testowym trafiają do `ml/results/models.csv`). Każda wersja ma osobny plik i żaden trening
nie nadpisuje wcześniejszego modelu; backend używa wersji wskazanej w `MODEL_CHECKPOINT_PATH`
(`backend/.env`). Opis wszystkich wersji: [`ml/MODELS.md`](ml/MODELS.md).

Wytrenowane wersje modelu (konfiguracja, wyniki na zbiorze testowym, znane słabości) są opisane
w [`ml/MODELS.md`](ml/MODELS.md).

### Frontend

Wymaga uruchomionego backendu (patrz wyżej, domyślnie na `http://127.0.0.1:8000`).

```bash
cd frontend
npm install
npm run dev
```

- `http://localhost:3000/` — aplikacja (port ustawiony na sztywno w `vite.config.ts`, żeby
  pasował do domyślnego `CORS_ORIGINS` backendu)
- Adres backendu można nadpisać zmienną `VITE_API_BASE_URL` (domyślnie `http://127.0.0.1:8000`)

## Status

MVP działa end-to-end (upload → predykcja → Grad-CAM → historia). Trwają eksperymenty nad
poprawą modelu.

- ✅ Dataset (HAM10000, podział po `lesion_id`) i pipeline treningowy — `ml/`
- ✅ Model: 7 udokumentowanych wersji (v1–v7), w backendzie v3 — zob. [Wyniki modelu](#wyniki-modelu)
- ✅ Grad-CAM podłączony do API — każde `POST /predict` zwraca też `gradcam_url`
- ✅ Backend (FastAPI + PostgreSQL): `GET /health`, `POST /predict` (inferencja + Grad-CAM +
  zapis przypadku), `GET/DELETE /cases`, pliki pod `/storage/uploads` i `/storage/gradcam`
- ✅ Frontend (Vite + React + TypeScript + Tailwind) — 3 ekrany MVP: upload + wynik z Grad-CAM
  (`/`), historia przypadków (`/cases`), szczegół przypadku (`/cases/:id`)
- 🔧 Dalsze eksperymenty z modelem (augmentacja LCA, ważona strata, model finalny na train+val)
- ⬜ Testy backendu (`backend/tests/` — na razie same szkielety plików)
- ⬜ Konteneryzacja (Docker Compose)

## Dane i licencja

Model jest trenowany na zbiorze [HAM10000](https://doi.org/10.7910/DVN/DBW86T) (Tschandl i in.,
2018), udostępnionym na licencji **CC BY-NC 4.0**, więc wolno go używać tylko niekomercyjnie.
Zbiór danych ani wagi modeli nie są przechowywane w repozytorium.
