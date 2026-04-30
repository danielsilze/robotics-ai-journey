# fast.ai Lektion 5 — Linear Model & Neural Net from Scratch

**Notebook:** `jhoward/linear-model-and-neural-net-from-scratch`
**Datensatz:** Titanic (Kaggle)
**Ziel:** Vorhersagen wer überlebt (1) oder stirbt (0)

---

## Das große Bild

Jeremy baut in diesem Notebook ein Neural Net **komplett von Hand** — ohne fertige Architekturen, ohne Optimizer-Libraries. Nur PyTorch und Mathe.

**Warum?** Damit man versteht was wirklich passiert — kein "schwarze Box" Denken mehr.

Der komplette Ablauf:
```
Daten laden → bereinigen → normalisieren
→ Koeffizienten initialisieren
→ Vorhersage berechnen
→ Loss berechnen
→ Gradienten berechnen (backward)
→ Koeffizienten anpassen
→ wiederholen → Loss wird kleiner
```

---

## 1. Daten bereinigen (Cleaning)

### Fehlende Werte füllen
```python
modes = df.mode().iloc[0]   # häufigster Wert jeder Spalte
df.fillna(modes, inplace=True)
```

**Warum:** Man kann nicht mit fehlenden Werten (`NaN`) rechnen. Man füllt sie mit dem häufigsten Wert (Mode) — das ist eine einfache aber effektive Methode.

### Log für Geldbeträge
```python
df['LogFare'] = np.log(df['Fare'] + 1)
```

**Warum:** `Fare` hat extreme Ausreißer (z.B. $500 neben $5). Das dominiert das Training.
`log()` drückt große Zahlen zusammen:
```
log(5)   = 1.6
log(50)  = 3.9
log(500) = 6.2
```
Statt Faktor 100 Unterschied → nur Faktor 4. Viel besser für das Modell.

**Warum +1:** `log(0)` ist unendlich — deshalb `+1` damit Null-Werte funktionieren.

### Kategorien → Zahlen (Dummy Variables)
```python
df = pd.get_dummies(df, columns=["Sex","Pclass","Embarked"])
```

**Warum:** Man kann `"male"` oder `"female"` nicht mit einer Zahl multiplizieren.
`get_dummies` erstellt neue Spalten:
```
Sex_male   = 1 wenn männlich, sonst 0
Sex_female = 1 wenn weiblich, sonst 0
Pclass_1   = 1 wenn 1. Klasse, sonst 0
Pclass_2   = 1 wenn 2. Klasse, sonst 0
...
```

---

## 2. Daten als PyTorch Tensoren

```python
t_dep = tensor(df.Survived)   # Ziel: überlebt? (0 oder 1)

indep_cols = ['Age', 'SibSp', 'Parch', 'LogFare',
              'Sex_male', 'Sex_female', 'Pclass_1', ...]

t_indep = tensor(df[indep_cols].values, dtype=torch.float)
```

**Dependent Variable (`t_dep`):** Was wir vorhersagen wollen (Survived)
**Independent Variables (`t_indep`):** Was wir zur Vorhersage nutzen (Alter, Geschlecht, Klasse...)

---

## 3. Normalisierung

```python
vals, indices = t_indep.max(dim=0)
t_indep = t_indep / vals
```

**Warum:** `Age` (Werte 0–80) würde `Sex_male` (Werte 0–1) komplett dominieren.
Nach Normalisierung: alle Spalten zwischen 0 und 1.

**Broadcasting:** `t_indep / vals` teilt eine Matrix durch einen Vektor — PyTorch macht das automatisch für jede Zeile.

---

## 4. Lineares Modell

### Koeffizienten initialisieren
```python
coeffs = torch.rand(n_coeff) - 0.5
```

Zufällige Startwerte zwischen -0.5 und 0.5 — nur ein Ausgangspunkt.

### Vorhersage berechnen
```python
preds = (t_indep * coeffs).sum(axis=1)
```

Jede Spalte × ihr Koeffizient → alles zusammenzählen → eine Zahl pro Passagier.

**Kompakter mit Matrix-Multiplikation (`@`):**
```python
preds = t_indep @ coeffs   # identisch, aber schneller
```

`@` ist der PyTorch/NumPy Operator für Matrix-Multiplikation.

### Loss berechnen
```python
loss = torch.abs(preds - t_dep).mean()
```

Wie weit ist unsere Vorhersage von der richtigen Antwort entfernt?
- Vorhersage 0.8, Richtig 1 → Fehler = 0.2
- Vorhersage 0.3, Richtig 1 → Fehler = 0.7

---

## 5. Gradient Descent (Herzstück)

```python
coeffs.requires_grad_()      # PyTorch soll Gradienten tracken

loss = calc_loss(coeffs, t_indep, t_dep)
loss.backward()              # Gradienten berechnen

with torch.no_grad():
    coeffs.sub_(coeffs.grad * lr)   # Koeffizienten anpassen
    coeffs.grad.zero_()             # Gradienten zurücksetzen!
```

**Was passiert Schritt für Schritt:**
1. `requires_grad_()` — PyTorch merkt sich jeden Rechenschritt
2. `loss.backward()` — berechnet für jeden Koeffizienten: "wie sehr hat er zum Fehler beigetragen?"
3. `coeffs.sub_(grad * lr)` — bewegt jeden Koeffizienten ein kleines Stück in die richtige Richtung
4. `grad.zero_()` — **wichtig:** Gradienten werden addiert, nicht ersetzt — muss man manuell zurücksetzen

**Learning Rate (lr):** Wie groß ist der Schritt?
- Zu groß → springt über das Minimum
- Zu klein → Training dauert ewig
- Typisch: 0.01 bis 0.2

**`_` am Ende bedeutet in-place:** `sub_()` ändert die Variable direkt, statt eine neue zu erstellen.

---

## 6. Training Loop

```python
def one_epoch(coeffs, lr):
    loss = calc_loss(coeffs, trn_indep, trn_dep)
    loss.backward()
    with torch.no_grad():
        coeffs.sub_(coeffs.grad * lr)
        coeffs.grad.zero_()

def train_model(epochs=30, lr=0.01):
    coeffs = init_coeffs()
    for i in range(epochs):
        one_epoch(coeffs, lr=lr)
    return coeffs

coeffs = train_model(18, lr=0.2)
```

**Epoch:** Ein kompletter Durchlauf durch alle Trainingsdaten.
30 Epochen = 30 Mal alle Daten gesehen + Koeffizienten angepasst.

### Validation Set
```python
trn_split, val_split = RandomSplitter(seed=42)(df)
trn_indep, val_indep = t_indep[trn_split], t_indep[val_split]
```

80% Training, 20% Validation — damit wir sehen ob das Modell wirklich lernt oder nur auswendig lernt.

---

## 7. Sigmoid

**Problem ohne Sigmoid:**
```
Vorhersagen: -0.3, 0.7, 1.4, 2.1  ← negativ und >1 macht keinen Sinn
```

**Mit Sigmoid:**
```python
def calc_preds(coeffs, indeps):
    return torch.sigmoid(indeps @ coeffs)
```

Sigmoid drückt alles in 0–1:
```
-0.3 → 0.43  (43% Überlebenschance)
 0.7 → 0.67  (67% Überlebenschance)
 1.4 → 0.80  (80% Überlebenschance)
```

**Formel:** `sigmoid(x) = 1 / (1 + e^(-x))`

Ergibt eine S-Kurve — daher der Name (Sigma = S).

---

## 8. Accuracy messen

```python
preds = calc_preds(coeffs, val_indep)
results = val_dep.bool() == (preds > 0.5)
accuracy = results.float().mean()
```

- Vorhersage > 0.5 → überlebt
- Vorhersage ≤ 0.5 → nicht überlebt
- Accuracy = Anteil der korrekt vorhergesagten Passagiere

Ergebnis: ~82% Accuracy — nicht schlecht für ein selbst gebautes Modell!

---

## 9. Neural Network (eine versteckte Schicht)

Lineares Modell: eine Schicht
Neural Net: **mehrere Schichten**

```python
def init_coeffs(n_hidden=20):
    layer1 = (torch.rand(n_coeff, n_hidden) - 0.5) / n_hidden
    layer2 = torch.rand(n_hidden, 1) - 0.3
    const = torch.rand(1)[0]
    return layer1.requires_grad_(), layer2.requires_grad_(), const.requires_grad_()

def calc_preds(coeffs, indeps):
    l1, l2, const = coeffs
    res = F.relu(indeps @ l1)    # Schicht 1 + ReLU
    res = res @ l2 + const       # Schicht 2
    return torch.sigmoid(res)
```

**Was passiert:**
1. Input (12 Spalten) → Schicht 1 → 20 versteckte Neuronen
2. ReLU — Aktivierungsfunktion (negatives → 0)
3. 20 Neuronen → Schicht 2 → 1 Output (Überlebenschance)
4. Sigmoid → 0–1

**ReLU:** `max(0, x)` — einfachste Aktivierungsfunktion
- Positiv → bleibt
- Negativ → wird 0

**Warum ReLU?** Ohne Aktivierungsfunktion wäre zwei Schichten dasselbe wie eine — nur andere Zahlen.

---

## 10. Deep Learning (mehrere Schichten)

```python
def init_coeffs():
    hiddens = [10, 10]   # zwei versteckte Schichten mit je 10 Neuronen
    sizes = [n_coeff] + hiddens + [1]
    layers = [(torch.rand(sizes[i], sizes[i+1]) - 0.3) / sizes[i+1] * 4
              for i in range(len(sizes)-1)]
    consts = [(torch.rand(1)[0] - 0.5) * 0.1
              for i in range(len(sizes)-1)]
    return [l.requires_grad_() for l in layers], [c.requires_grad_() for c in consts]

def calc_preds(coeffs, indeps):
    layers, consts = coeffs
    res = indeps
    for i, l in enumerate(layers):
        res = res @ l + consts[i]
        if i != len(layers)-1:
            res = F.relu(res)   # ReLU außer in der letzten Schicht
    return torch.sigmoid(res)
```

**"Deep" Learning = mehr als eine versteckte Schicht**

Architektur hier: `12 → 10 → 10 → 1`
- 12 Input-Features
- Erste versteckte Schicht: 10 Neuronen
- Zweite versteckte Schicht: 10 Neuronen
- Output: 1 Zahl (Überlebenschance)

---

## 11. Kaggle Submission

```python
tst_df = pd.read_csv(path/'test.csv')

# Gleiche Vorverarbeitung wie Training
tst_df.fillna(modes, inplace=True)
tst_df['LogFare'] = np.log(tst_df['Fare'] + 1)
tst_df = pd.get_dummies(tst_df, columns=["Sex","Pclass","Embarked"])
tst_indep = tensor(tst_df[indep_cols].values, dtype=torch.float) / vals

# Vorhersagen
tst_df['Survived'] = (calc_preds(tst_indep, coeffs) > 0.5).int()

# CSV speichern
sub_df = tst_df[['PassengerId','Survived']]
sub_df.to_csv('sub.csv', index=False)
```

---

## Die wichtigsten Erkenntnisse

### 1. Ein Neural Net ist nur Mathe
```
Input × Gewichte → Schicht 1
Schicht 1 × Gewichte → Schicht 2
...
Schicht N → Sigmoid → Wahrscheinlichkeit
```

Alles ist Matrix-Multiplikation. Keine Magie.

### 2. Gradient Descent ist universell
Egal ob lineares Modell, Neural Net oder GPT-4 — alle lernen durch:
- Loss berechnen
- `backward()` aufrufen
- Gewichte anpassen
- Wiederholen

### 3. Initialisierung ist kritisch
Jeremy warnt: "Die kleinste Änderung der Initialisierung kann das Training komplett kaputt machen."
Deshalb gibt es in der Praxis Methoden wie **Xavier Initialization** — fast.ai macht das automatisch.

### 4. Tiefere Netze ≠ immer besser
Auf kleinen Datensätzen (Titanic hat nur 891 Zeilen) ist ein lineares Modell oft genauso gut wie ein Deep Net. Neural Nets brauchen **viele Daten**.

### 5. `@` ist Matrix-Multiplikation
```python
result = matrix @ vector   # schneller als (matrix * vector).sum(axis=1)
```

---

## Begriffe Glossar

| Begriff | Bedeutung |
|---------|-----------|
| **Epoch** | Ein kompletter Durchlauf durch alle Trainingsdaten |
| **Loss** | Wie falsch das Modell ist (niedriger = besser) |
| **Gradient** | Richtung in die man Gewichte anpassen muss |
| **Learning Rate** | Schrittgröße bei der Anpassung |
| **Sigmoid** | Drückt Zahlen in 0–1 Bereich |
| **ReLU** | Aktivierungsfunktion: max(0, x) |
| **Dummy Variable** | Kategorie als 0/1 Spalte |
| **Normalisierung** | Alle Spalten auf gleiche Skala bringen |
| **Broadcasting** | Matrix ÷ Vektor automatisch für jede Zeile |
| **in-place** | Methoden mit `_` ändern Variable direkt (sub_, zero_) |
| **requires_grad** | PyTorch soll Gradienten für diese Variable tracken |
| **backward()** | Berechnet alle Gradienten |
| **Validation Set** | Daten die nicht trainiert wurden — zum fairen Testen |

---

*Erstellt: 30. April 2026 — basierend auf Jeremy Howards Notebook*
