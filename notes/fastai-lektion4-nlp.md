# fast.ai Lektion 4 — NLP komplett

**Kurs:** Practical Deep Learning for Coders  
**Video:** [Lesson 4: Natural Language (NLP)](https://www.youtube.com/watch?v=toUgBQv1BT8)  
**Notebook:** [Getting started with NLP for absolute beginners](https://www.kaggle.com/code/jhoward/getting-started-with-nlp-for-absolute-beginners)

---

## Das Projekt — US Patent Phrase Matching

Kaggle-Wettbewerb: Wie ähnlich sind zwei Patent-Phrasen?

```
anchor:  "semiconductor"
target:  "interchangeable"
score:   0.25  → nicht ähnlich

anchor:  "display panel"
target:  "visual screen"
score:   0.75  → sehr ähnlich
```

Trick: Beide Phrasen zu einem Text zusammenbauen:
```python
df['input'] = 'TEXT1: ' + df.anchor + '; TEXT2: ' + df.target
```
Ähnlichkeitsproblem wird zu Klassifikationsproblem.

---

## Tokenization

Text wird in kleine Stücke (Tokens) zerhackt — Modelle verstehen nur Zahlen.

### Warum Subwords?

| Option | Problem |
|---|---|
| Ganze Wörter | Millionen Einträge, unbekannte Wörter |
| Einzelne Buchstaben | Zu viele Tokens, Kontext geht verloren |
| Subwords (BPE) | Kompromiss — funktioniert für alles |

### BPE (Byte Pair Encoding)

Algorithmus der das Wörterbuch automatisch lernt:

```
Start: ["t","o","k","e","n","i","z","a","t","i","o","n"]
Runde 1: häufigstes Paar zusammenfassen → "to"
Runde 2: nächstes Paar → "en"
...nach 30.000 Runden: optimales Wörterbuch
```

Ergebnis:
```
"tokenization"  →  ["token", "ization"]
"unbekannteswort" →  ["un", "bek", "anntes", "wort"]
"xqzpfw"        →  ["x","q","z","p","f","w"]  ← worst case
```

### Numericalization

Jeder Token bekommt eine feste Nummer:
```
"hello"  →  7592
"world"  →  2088
"[CLS]"  →  101   ← Satzanfang
"[SEP]"  →  102   ← Satzende

"hello world"  →  [101, 7592, 2088, 102]
```

**Was sind [CLS] und [SEP]?**
- `[CLS]` (Classification Token) = wird am Anfang jedes Textes eingefügt. Das Modell packt die "Gesamtbedeutung" des Textes in diesen Token — wird für die finale Klassifikation verwendet.
- `[SEP]` (Separator Token) = markiert das Ende eines Textes oder die Grenze zwischen zwei Texten.
```
Eingabe:  "hello world"
Mit Tokens: [CLS] hello world [SEP]
Als Zahlen: [101, 7592, 2088, 102]
```
Diese Sonderzeichen wurden beim Vortraining des Modells verwendet — deshalb muss man sie auch beim Fine-Tuning verwenden.

### Wichtige Regel

```python
# RICHTIG — Modells eigenen Tokenizer verwenden
tokz = AutoTokenizer.from_pretrained('microsoft/deberta-v3-small')

# FALSCH — eigener Tokenizer passt nicht zum vortrainierten Modell
```

### In Code

```python
def tok_func(x):
    return tokz(x['input'], truncation=True)

tok_ds = ds.map(tok_func, batched=True)
```

---

## Truncation — lange Dokumente

Jedes Modell hat ein maximales Kontextfenster:
```
DeBERTa-v3-small: max. 512 Tokens
GPT-4:            max. 128.000 Tokens
```

```python
tokz("langer text...", truncation=True)
# → alles nach Token 512 wird abgeschnitten
```

Lösungen bei zu langen Texten:
- `truncation=True` → einfach abschneiden
- Überlappende Stücke (stride/overlap)
- Größeres Modell mit mehr Kontext
- Text vorher zusammenfassen

---

## Parallel Processing

```python
# LANGSAM
tok_ds = ds.map(tok_func)

# SCHNELL — 100x schneller
tok_ds = ds.map(tok_func, batched=True)
```

Mit `batched=True` werden 128 Texte gleichzeitig verarbeitet — möglich weil Tokenizer kein Gedächtnis zwischen Texten hat.

---

## ULMFiT

**Universal Language Model Fine-Tuning** — erfunden von Jeremy Howard & Sebastian Ruder, 2018.

Hat Transfer Learning für NLP populär gemacht.

### Die 3 Schritte

```
Schritt 1 — Pretraining auf Wikipedia
  Modell liest 100 Millionen Sätze
  Lernt: Grammatik, Bedeutung, Zusammenhänge
  Aufgabe: nächstes Wort vorhersagen
  "Der Hund ___" → "bellt"

Schritt 2 — Language Model Finetuning (auf eigenen Daten)
  Modell liest deine Texte (z.B. Patent-Dokumente)
  Lernt: Fachsprache, Domäne
  Kein Label nötig — nur roher Text

Schritt 3 — Classifier Finetuning
  Jetzt kommen Labels
  Modell lernt: was bedeutet score 0.0 vs 1.0?
```

### Language Model vs. Classifier

```
Language Model:  "Der Hund ___"  →  nächstes Wort vorhersagen
Classifier:      "Der Hund bellt" →  "positiv / negativ"
```

Erst Language Model trainieren, dann Classifier — Modell muss Sprache verstehen bevor es klassifiziert.

---

## AWD-LSTM

**ASGD Weight-Dropped Long Short-Term Memory**

Architektur hinter ULMFiT. Liest Text Wort für Wort — hat Gedächtnis für langen Kontext.

### LSTM Gedächtnis

```
"Der Mann, der gestern im Laden war, kaufte..."

Wort 1: "Der"    → Gedächtnis: [subject=?]
Wort 2: "Mann"   → Gedächtnis: [subject=Mann]
...
Wort 8: "kaufte" → Gedächtnis: [subject=Mann → er kaufte]
```

### Drei Tore

| Tor | Funktion |
|---|---|
| Forget Gate | Was wird aus Gedächtnis gelöscht? |
| Input Gate | Was wird neu gespeichert? |
| Output Gate | Was wird weitergegeben? |

### AWD = Verbesserungen gegen Overfitting

**ASGD — Averaged Stochastic Gradient Descent**

Normales SGD macht nach jedem Batch einen Schritt und verwirft den alten Wert:
```
Schritt 1: Gewicht = 0.5
Schritt 2: Gewicht = 0.48
Schritt 3: Gewicht = 0.51  ← springt hin und her
Schritt 4: Gewicht = 0.49
```

ASGD speichert alle Zwischenwerte und mittelt sie am Ende:
```
Durchschnitt: (0.5 + 0.48 + 0.51 + 0.49) / 4 = 0.495
```
Ergebnis: Gewichte landen genauer im Optimum statt ständig drum herumzuspringen.

Analogie: Du schätzt die Temperatur draußen. Einmal messen = ungenau. 10x messen und mitteln = viel genauer.

---

**Weight-Dropped (Dropout)**

Während jedem Trainingsschritt werden zufällig X% der Verbindungen zwischen Neuronen auf 0 gesetzt — also deaktiviert:

```
Ohne Dropout:
  Neuron A → Neuron B → Neuron C → Ausgabe
  Modell verlässt sich stark auf Neuron B

Mit Dropout (z.B. 20%):
  Schritt 1: Neuron A → [B deaktiviert] → Neuron C → Ausgabe
  Schritt 2: [A deaktiviert] → Neuron B → Neuron C → Ausgabe
  Schritt 3: Neuron A → Neuron B → [C deaktiviert] → Ausgabe
```

Jedes Mal andere Neuronen aktiv → Modell kann sich nicht auf einzelne verlassen → lernt robustere, verallgemeinerbare Muster.

Beim echten Einsatz (Inference) sind alle Neuronen aktiv — dann nutzt das Modell sein volles Wissen.

Analogie: Fußballtraining wo immer andere Spieler fehlen. Das Team lernt flexibel zu spielen statt auf einzelne Stars angewiesen zu sein.

---

## Transformer & Attention

**"Attention Is All You Need"** — Google, 2017. Basis für GPT, BERT, Claude, ChatGPT.

### Das Problem mit LSTM

LSTM liest sequenziell — bei langen Texten geht früher Kontext verloren.

### Attention — die Lösung

Jedes Wort schaut gleichzeitig auf alle anderen:

```
"Der Bank-Direktor saß an der Bank des Flusses"

"Bank" (zweites) schaut auf:
  "Fluss"         → sehr relevant (score: 0.9)
  "saß"           → relevant     (score: 0.6)
  "Bank-Direktor" → weniger      (score: 0.1)
→ Bedeutung: "Ufer", nicht "Geldinstitut"
```

### Multi-Head Attention

Nicht eine Attention — 8-12 parallel, jede lernt andere Beziehungen:
- Head 1: grammatische Beziehungen
- Head 2: semantische Bedeutung
- Head 3: Koreferenzen (er/sie/es → wer ist gemeint?)

### Transformer vs. AWD-LSTM

| | AWD-LSTM | Transformer |
|---|---|---|
| Liest Text | sequenziell | alles gleichzeitig |
| Geschwindigkeit | langsamer | viel schneller auf GPU |
| Kontext | begrenzt | besser bei langen Texten |
| Heute | überholt | State-of-the-art |

---

## DeBERTa

**Decoding-enhanced BERT with Disentangled Attention** — Microsoft, 2021.

Das konkrete Modell aus Lektion 4.

### Verbesserung gegenüber BERT

BERT behandelt Wortinhalt und Position zusammen. DeBERTa trennt sie:

```
BERT:    Inhalt + Position → zusammen verarbeitet
DeBERTa: Inhalt            → separat
         Position          → separat
→ versteht Satzstruktur besser
```

### Modellwahl

```python
model_nm = 'microsoft/deberta-v3-small'
model = AutoModelForSequenceClassification.from_pretrained(model_nm, num_labels=1)
# num_labels=1  → eine Zahl ausgeben (0.0 bis 1.0) = Regression
# num_labels=2  → zwei Kategorien (z.B. positiv/negativ) = Klassifikation
# num_labels=10 → zehn Kategorien (z.B. Ziffern 0-9)
```

**Was ist `from_pretrained`?**
Lädt ein bereits trainiertes Modell aus der Hugging Face Cloud herunter — mit allen Gewichten die es beim Vortraining gelernt hat. Ohne `from_pretrained` würde das Modell mit zufälligen Gewichten starten.

---

## Transfer Learning

Das übergreifende Konzept — gilt für Bilder UND Text.

```
Vortrainiertes Modell hat gelernt:
  Was sind Wörter / Konzepte?
  Wie hängen sie zusammen?
  Grammatik, Logik, Weltwissen

Finetuning fügt hinzu:
  Was ist spezifisch an dieser Aufgabe?
  Welche Features sind relevant?
```

In Zahlen:
```
Ohne Transfer Learning: 1.000.000 Beispiele, Wochen Training
Mit Transfer Learning:  1.000 Beispiele, Stunden Training → oft besser!
```

---

## Zeiler & Fergus Paper

**"Visualizing and Understanding Convolutional Networks"** — 2013

Erklärt WARUM Fine-Tuning funktioniert. Schichten lernen hierarchisch:

```
Schicht 1 → Grundlagen     (Buchstaben, Wortteile)
Schicht 2 → Muster         (Wörter, Wortarten)
Schicht 3 → Strukturen     (Phrasen, Bedeutungsgruppen)
Schicht 4 → Zusammenhänge  (Satzstruktur, Kontext)
Schicht 5 → Aufgabe        (komplexe Bedeutung)
```

**Konsequenz:** Frühe Schichten = universell → nicht anfassen. Späte Schichten = aufgabenspezifisch → anpassen.

---

## Gradual Unfreezing

```python
# Runde 1: nur letzte Schicht
learn.freeze_to(-1)
learn.fit(1, lr=1e-2)

# Runde 2: letzte 2 Schichten
learn.freeze_to(-2)
learn.fit(1, lr=5e-3)

# Runde 3: alles
learn.unfreeze()
learn.fit(2, lr=1e-3)
```

Späte Schichten lernen zuerst — wenn stabil, werden frühe Schichten vorsichtig angepasst.

---

## Diskriminative Learning Rates

```python
learn.fit_one_cycle(4, slice(1e-5, 1e-3))
#                      ↑ frühe    ↑ späte Schichten
```

**Was bedeutet `slice(1e-5, 1e-3)`?**
- `1e-5` = 0.00001 (wissenschaftliche Notation: 1 × 10⁻⁵)
- `1e-3` = 0.001   (1 × 10⁻³)
- `slice(von, bis)` = fast.ai verteilt die Lernraten automatisch zwischen diesen zwei Werten über alle Schichten

```
Schicht 1 (allgemein)  → 0.00001  (kaum ändern)
Schicht 2              → 0.00003
Schicht 3              → 0.0001
Schicht 4 (spezifisch) → 0.001   (stark anpassen)
```

---

## Batch Size

```python
per_device_train_batch_size=128
```

128 Texte gleichzeitig durch das Modell — einen gemeinsamen Gradienten berechnen — Gewichte einmal updaten.

**Was ist ein Gradient?**
Der Gradient sagt dem Modell: "In welche Richtung müssen die Gewichte geändert werden, damit der Fehler kleiner wird?"
```
Fehler groß → Gradient groß → Gewichte stark anpassen
Fehler klein → Gradient klein → Gewichte kaum anpassen
```
Stell dir vor du bist auf einem Hügel und willst ins Tal — der Gradient zeigt dir die steilste Richtung nach unten.

| Batch Size | Effekt |
|---|---|
| Klein (8-32) | noisiger Gradient, manchmal besser generalisiert |
| Groß (128-512) | stabiler, schneller pro Epoche |
| Zu groß | RAM-Fehler |

Auf Mac ohne GPU: `per_device_train_batch_size=16`

---

## Loss Function vs. Metric

| | Loss Function | Metric |
|---|---|---|
| Zweck | Modell trainieren | Ergebnis messen |
| Muss differenzierbar sein | JA | Nein |
| Beispiel | MSE, Cross Entropy | Pearson r, Accuracy |

**Was bedeutet "differenzierbar"?**
Differenzierbar = man kann einen Gradienten berechnen = Gradient Descent kann damit arbeiten.
```
MSE (x-y)²  → glatte Kurve → Gradient berechenbar → ✅ für Training
Accuracy    → springt von 0 auf 1 → kein glatter Gradient → ❌ für Training
```
Nicht differenzierbare Metrics können trotzdem gut messen — nur nicht zum Trainieren verwendet werden.

Training: Modell minimiert Loss. Evaluation: Wir messen Metric.

---

## Pearson Correlation (r)

Misst: Folgen Vorhersagen dem gleichen Trend wie echte Werte?

```
r = 1.0  → perfekt
r = 0.0  → kein Zusammenhang
r = -1.0 → perfekt umgekehrt
```

Empfindlich auf Ausreißer — aber **nicht löschen**: Ausreißer enthalten oft wichtige Information.

---

## Overfitting vs. Underfitting

```
Underfitting:  Training schlecht,    Validation schlecht  → Modell zu simpel
Overfitting:   Training sehr gut,    Validation schlecht  → Modell lernt auswendig
Richtig:       Training gut,         Validation gut
```

---

## Validation Set

```
FALSCH — random split:
  Modell sieht ähnliche Daten → scheinbar gute Ergebnisse

RICHTIG — zeitlicher Split:
  Train: Januar–September | Validation: Oktober–Dezember
  → echte Generalisierung
```

Kaggle hat zwei Test-Sets:
- **Public Leaderboard** → während Wettbewerb sichtbar
- **Private Leaderboard** → echtes Ergebnis am Ende

---

## Post-Processing

Modell-Output kann außerhalb [0,1] liegen:

```python
preds = trainer.predict(tok_ds['test']).predictions
# [[ 0.234], [-0.021], [1.043], ...]  ← negativ und über 1.0 möglich

preds = np.clip(preds, 0, 1)
# [0.234, 0.0, 1.0, ...]  ← bereinigt
```

---

## Modellauswahl

```python
# Task bestimmt die Klasse:
AutoModelForSequenceClassification   # Text klassifizieren / vergleichen
AutoModelForQuestionAnswering        # Fragen beantworten
AutoModelForCausalLM                 # Text generieren

# Größe:
deberta-v3-small   # schnell, Experimente
deberta-v3-base    # Standard
deberta-v3-large   # besser, langsamer
```

---

## Kompletter Code

```python
from pathlib import Path
import pandas as pd
import numpy as np
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer

# 1. Daten laden
path = Path('../input/us-patent-phrase-to-phrase-matching')
df = pd.read_csv(path/'train.csv')

# 2. Texte zusammenbauen
df['input'] = 'TEXT1: ' + df.context + '; TEXT2: ' + df.anchor + '; SIMILAR: ' + df.target

# 3. Dataset + Split
ds = Dataset.from_pandas(df).rename_column('score', 'label')
dds = ds.train_test_split(0.25, seed=42)
# seed=42: Zufallszahl für die Aufteilung — gleiche Zahl = gleiche Aufteilung bei jedem Durchlauf
# 42 ist eine Konvention unter Programmierern (aus "Per Anhalter durch die Galaxis")
# Wichtig für Reproduzierbarkeit: andere sollen dein Experiment nachstellen können

# 4. Tokenizer
model_nm = 'microsoft/deberta-v3-small'
tokz = AutoTokenizer.from_pretrained(model_nm)

def tok_func(x):
    return tokz(x['input'], truncation=True)

tok_ds = dds.map(tok_func, batched=True)

# 5. Modell laden
model = AutoModelForSequenceClassification.from_pretrained(model_nm, num_labels=1)

# 6. Training
args = TrainingArguments(
    'outputs',               # Ordner wo Checkpoints gespeichert werden
    learning_rate=8e-5,      # 0.00008 — klein weil Modell schon vortrainiert
    per_device_train_batch_size=128,  # 128 Texte gleichzeitig
    num_train_epochs=4,      # 4x durch alle Daten
    evaluation_strategy='epoch'  # nach jeder Epoche Validation messen
)

trainer = Trainer(
    model, args,
    train_dataset=tok_ds['train'],  # Daten zum Lernen
    eval_dataset=tok_ds['test'],    # Daten zum Prüfen (werden nicht gelernt)
    tokenizer=tokz
)
trainer.train()

# 7. Vorhersagen + Post-Processing
preds = trainer.predict(tok_ds['test']).predictions.astype(float)
preds = np.clip(preds, 0, 1)
```

---

## Gesamtüberblick

```
CSV laden
    ↓
Texte zusammenbauen (anchor + target)
    ↓
Subword Tokenization (BPE — 30.000 Tokens)
    ↓
Numericalization (Token → Zahl)
    ↓
truncation=True (max 512 Tokens)
    ↓
batched=True (128 Texte gleichzeitig)
    ↓
DeBERTa-v3-small laden
    ↓
Transfer Learning / ULMFiT Prinzip
    ↓
Fine-Tuning (Gradual Unfreezing + Disk. Lernraten)
    ↓
Loss: MSE | Metric: Pearson r
    ↓
Validation Set (zeitlich)
    ↓
Post-Processing (clip auf 0-1)
    ↓
Kaggle einreichen
```

---

## Das Problem mit Metrics — Why Metrics Are Dangerous in AI

### "You get what you measure"

Wenn ein Modell auf eine Metric optimiert wird, findet es Wege die Metric zu maximieren — nicht das eigentliche Ziel.

**Beispiel 1 — Accuracy täuscht:**
```
Dataset: 95% gesunde Patienten, 5% krank
Modell sagt immer "gesund"
→ Accuracy: 95%  ← sieht gut aus!
→ Realität: findet keinen einzigen Kranken
```

**Beispiel 2 — YouTube Watch Time:**
```
Ziel:     Nutzer sollen tolle Videos schauen
Metric:   Watch Time maximieren
Ergebnis: Extreme, aufwühlende Videos werden empfohlen
          → Menschen schauen länger, werden aber radikalisiert
```

**Beispiel 3 — Klausuren-Chatbot:**
```
Ziel:     Schüler sollen lernen
Metric:   Klausur-Score maximieren
Ergebnis: Chatbot schreibt Klausuren für Schüler
          → Score hoch, Lernen null
```

### Pearson r — konkretes Problem aus Lektion 4

```
Echte Werte:   [0.0, 0.25, 0.5, 0.75, 99.0]  ← ein Ausreißer
Vorhersagen:   [0.1, 0.2,  0.6, 0.8,   1.0]

→ r wird schlecht obwohl fast alles korrekt vorhergesagt
```

Ein einziger Ausreißer kann die gesamte Metric verzerren.  
Jeremys Regel: **Ausreißer nicht löschen** — verstehen warum sie da sind.

### Das tiefere AI-Problem

```
Modell optimiert auf:  "Nutzer klickt"
Echtes Ziel:           "Nutzer findet hilfreiche Information"

Modell optimiert auf:  "Text klingt überzeugend"
Echtes Ziel:           "Text ist wahr"
```

Metrics messen Zahlen — aber Zahlen sind nie das eigentliche Ziel.  
Das ist eines der fundamentalsten Probleme in AI (auch bekannt als **Goodhart's Law**):

> *"When a measure becomes a target, it ceases to be a good measure."*

---

*Erstellt: 29. April 2026 | Quelle: fast.ai Lektion 4*
