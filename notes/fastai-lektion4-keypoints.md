# Lektion 4 — Die wichtigsten Punkte

## Tokenization
- Immer den **eigenen Tokenizer des Modells** verwenden — nie einen eigenen
- `batched=True` nicht vergessen → 100x schneller
- `truncation=True` bei langen Texten

## Modell & Transfer Learning
- **Vortrainiertes Modell laden** — nie von Null trainieren wenn möglich
- `num_labels=1` → Zahl ausgeben | `num_labels=2+` → Kategorien
- Frühe Schichten einfrieren, nur späte Schichten anpassen

## Training
- **Lernrate klein halten** bei vortrainierten Modellen (`8e-5` statt `1e-3`)
- Batch Size: auf Mac ohne GPU → `16`, mit GPU → `128`
- 3–4 Epochen reichen meistens

## Validation
- **Zeitlicher Split** statt random — sonst täuscht das Ergebnis
- Auf Kaggle: Public Leaderboard ist nicht das echte Ergebnis

## Metrics
- **Accuracy allein lügt** — immer mehrere Metrics prüfen
- Ausreißer nicht einfach löschen — verstehen warum sie da sind
- Metric ≠ echtes Ziel (Goodhart's Law)

## Post-Processing
- Vorhersagen immer auf gültigen Bereich clippen: `np.clip(preds, 0, 1)`

## Goldene Regeln
1. Transfer Learning > von Null trainieren
2. Validation Set > Training Score
3. Einfaches Modell zuerst, dann komplexer
4. Ausreißer untersuchen, nicht ignorieren
