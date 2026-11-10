# System wspomagania klasyfikacji zmian skórnych

Praca inżynierska — system wspomagający klasyfikację zmian skórnych na obrazach dermatoskopowych
z wykorzystaniem głębokich sieci neuronowych (transfer learning, EfficientNet), udostępniany
personelowi medycznemu poprzez aplikację webową.

## Struktura repozytorium

- `ml/` — przygotowanie danych, trening i ewaluacja modelu klasyfikacyjnego (PyTorch)
- `backend/` — REST API (FastAPI) obsługujące inferencję modelu i historię przypadków (PostgreSQL)
- `frontend/` — aplikacja webowa (React) dla personelu medycznego

## Uruchomienie (backend)

### Wymagania

- Python 3.12
- Docker (baza danych PostgreSQL)
- Wytrenowany checkpoint modelu pod `ml/models/efficientnet_b0_ham10000_v1.pt` — plik jest
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
(dzieli po `lesion_id`) → `python -m ml.src.train --version v2` (dwuetapowy trening: liniowa
sonda, potem fine-tuning; zapisuje checkpoint do `ml/models/efficientnet_b0_ham10000_v2.pt` i log
epok do `ml/results/training_log.csv`) → `python -m ml.src.evaluate --version v2` (wyniki na
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

Projekt w fazie wczesnej implementacji.

- ✅ Dataset (HAM10000, podział po `lesion_id`) i model bazowy (EfficientNet-B0, transfer
  learning, dwuetapowy trening) — `ml/`
- ✅ Grad-CAM podłączony do API — każde `POST /predict` zwraca też `gradcam_url` z nakładką
  pokazującą obszary, na których model oparł decyzję
- 🔧 Backend (FastAPI + PostgreSQL) — w budowie: `GET /health`, `POST /predict` (inferencja +
  Grad-CAM + zapis przypadku), `GET/DELETE /cases` (historia przypadków), przesłane zdjęcia i
  nakładki Grad-CAM serwowane pod `/storage/uploads` i `/storage/gradcam`
- ✅ Frontend (Vite + React + TypeScript + Tailwind) — 3 ekrany MVP: upload + wynik z Grad-CAM
  (`/`), historia przypadków (`/cases`), szczegół przypadku (`/cases/:id`)
- ⬜ Konteneryzacja (Docker Compose), testy i dokumentacja

Aktualna roadmapa: dataset → model bazowy → Grad-CAM → backend → frontend → konteneryzacja →
testy i dokumentacja.
