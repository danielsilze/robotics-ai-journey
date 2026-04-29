# Lernrate (Learning Rate)

## Was ist die Lernrate?

Die Lernrate bestimmt **wie stark das Modell auf seinen Fehler reagiert** — wie groß der Schritt bei Gradient Descent ist.

## Die Formel

```
Neues Gewicht = Altes Gewicht - Lernrate × Gradient
```

Beispiel:
```
Gewicht = 0.5
Gradient = 0.3
Lernrate = 0.01

Neues Gewicht = 0.5 - 0.01 × 0.3 = 0.497
```

## Zu groß / Zu klein / Richtig

| Lernrate | Effekt |
|---|---|
| Zu groß (z.B. 1.0) | Gewichte springen wild, Training divergiert |
| Zu klein (z.B. 0.000001) | Training dauert ewig |
| Richtig (z.B. 0.001) | Loss wird stetig kleiner, Training konvergiert |

## Analogie

Du lernst Fahrrad fahren und fällst nach rechts:
- **Zu groß** → Lenker stark nach links → fällst nach links → nie stabil
- **Zu klein** → minimale Korrektur → 1000 Versuche nötig
- **Richtig** → nach paar Versuchen stabil

## Einfluss auf Prozessorleistung

Lernrate hat keinen direkten Einfluss auf die Prozessorleistung — aber indirekt:
- Falsche Lernrate → mehr Epochen → mehr Rechenzeit
- Richtige Lernrate → weniger Epochen → weniger Rechenzeit

## Lernrate-Scheduler (wie bei Claude/ChatGPT)

Große Modelle nutzen keine feste Lernrate — sie verwenden einen Scheduler:

| Phase | Lernrate | Warum |
|---|---|---|
| Anfang | Klein (0.0001) | Modell noch instabil |
| Mitte | Größer (0.001) | Schnell lernen |
| Ende | Wieder klein (0.00001) | Fein abstimmen |

Analogie: Wie ein neuer Job — erst langsam, dann selbstständig, dann präzise.

## In fast.ai

`fine_tune()` macht den Lernrate-Scheduler automatisch.
`lr_find()` findet die optimale Lernrate automatisch.

```python
learn.lr_find()          # optimale Lernrate finden
learn.fine_tune(3)       # trainiert mit automatischer Lernrate
learn.fit(3, lr=0.001)   # manuelle Lernrate setzen
```
