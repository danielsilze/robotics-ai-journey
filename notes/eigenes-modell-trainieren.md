# Eigenes Modell trainieren — Schritt für Schritt

## Bilder klassifizieren (einfachste Option)

### Schritt 1 — Daten sammeln

```python
from fastai.vision.all import *
from duckduckgo_search import DDGS
from fastdownload import download_url
from time import sleep

# Kategorien definieren
categories = ('golden retriever', 'labrador', 'poodle')

# Bilder herunterladen
path = Path('mein_datensatz')
for cat in categories:
    dest = path/cat
    dest.mkdir(exist_ok=True, parents=True)
    
    with DDGS() as ddgs:
        results = ddgs.images(cat, max_results=50)
        for r in results:
            download_url(r['image'], dest/f'{cat}_{r["image"][-10:]}', show_progress=False)
            sleep(0.1)
```

### Schritt 2 — Kaputte Bilder entfernen

```python
failed = verify_images(get_image_files(path))
failed.map(Path.unlink)
print(f'{len(failed)} kaputte Bilder gelöscht')
```

### Schritt 3 — DataLoader erstellen

```python
dls = DataBlock(
    blocks=(ImageBlock, CategoryBlock),
    get_items=get_image_files,
    splitter=RandomSplitter(valid_pct=0.2, seed=42),
    get_y=parent_label,
    item_tfms=[Resize(192, method='squish')]
).dataloaders(path, bs=32)

# Daten anschauen
dls.show_batch()
```

### Schritt 4 — Modell erstellen

```python
learn = vision_learner(dls, resnet18, metrics=accuracy)
```

### Schritt 5 — Optimale Lernrate finden

```python
learn.lr_find()
# Grafik anschauen → wo fällt Loss am steilsten?
# Den Wert eine Zehnerpotenz vor dem Minimum nehmen
```

### Schritt 6 — Trainieren

```python
learn.fine_tune(3)
# 3 = Epochen, für Anfang reicht das
```

### Schritt 7 — Ergebnis anschauen

```python
learn.show_results()          # Vorhersagen visualisieren
interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix() # Welche Klassen werden verwechselt?
interp.plot_top_losses(5)      # Die 5 schlechtesten Vorhersagen
```

### Schritt 8 — Modell speichern

```python
learn.export('mein_modell.pkl')
```

### Schritt 9 — Modell benutzen

```python
learn_inf = load_learner('mein_modell.pkl')
pred, idx, probs = learn_inf.predict('testbild.jpg')
print(f'Vorhersage: {pred}, Sicherheit: {probs[idx]:.2f}')
```

---

## Text klassifizieren (Hugging Face)

### Schritt 1 — CSV vorbereiten

```
text,label
"Das ist super!",positiv
"Schreckliches Produkt",negativ
```

### Schritt 2 — Daten laden + tokenisieren

```python
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer

df = pd.read_csv('meine_daten.csv')
model_nm = 'microsoft/deberta-v3-small'
tokz = AutoTokenizer.from_pretrained(model_nm)

ds = Dataset.from_pandas(df)
tok_ds = ds.map(lambda x: tokz(x['text']), batched=True)
tok_ds = tok_ds.rename_columns({'label': 'labels'})
dds = tok_ds.train_test_split(0.2, seed=42)
```

### Schritt 3 — Modell + Training

```python
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer

model = AutoModelForSequenceClassification.from_pretrained(model_nm, num_labels=2)
# num_labels=2 für zwei Klassen (positiv/negativ)

args = TrainingArguments('outputs',
    learning_rate=8e-5,
    per_device_train_batch_size=16,  # klein für Mac
    num_train_epochs=3,
    evaluation_strategy='epoch')

trainer = Trainer(model, args,
    train_dataset=dds['train'],
    eval_dataset=dds['test'],
    tokenizer=tokz)

trainer.train()
```

### Schritt 4 — Vorhersage

```python
from transformers import pipeline

classifier = pipeline('text-classification', model='outputs/checkpoint-xxx')
classifier("Das ist fantastisch!")
# [{'label': 'positiv', 'score': 0.98}]
```

---

## Auf Kaggle trainieren (kostenlose GPU)

1. kaggle.com → "Create Notebook"
2. Oben rechts → Accelerator → **GPU T4 x2**
3. Code einfügen + ausführen
4. 30h GPU pro Woche kostenlos

---

## Checkliste vor dem Training

- [ ] Mindestens 50-100 Bilder pro Klasse (mehr = besser)
- [ ] Kaputte Bilder entfernt
- [ ] Validation Split gesetzt (20%)
- [ ] Vortrainiertes Modell gewählt (resnet18 für Anfang)
- [ ] lr_find() ausgeführt
- [ ] Nach Training: Confusion Matrix angeschaut

---

*Erstellt: 29. April 2026*
