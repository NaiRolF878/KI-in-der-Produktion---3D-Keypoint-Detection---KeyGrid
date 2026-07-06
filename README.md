# 3D Keypoint Detection mit Key-Grid

Studienprojekt im Fach **Künstliche Intelligenz in der Produktion**
Hochschule Karlsruhe, Fakultät Maschinenbau und Mechatronik
Sommersemester 2026

**Gruppe 02**

---

## 1. Aufgabenstellung

Im Rahmen der Vorlesung sollte ein 3D Keypoint Detection Verfahren auf Basis von
Punktwolken umgesetzt werden. Die konkreten Vorgaben aus der Aufgabenstellung:

- **Datensatz:** [KeypointNet](https://github.com/qq456cvb/KeypointNet)
- **Referenzmodelle** (als Ausgangspunkt genannt): UKPGAN, SkeletonMerger
- **Ziel:** Erkennen von Keypoints an Objekten, die durch Punktwolken repräsentiert
  sind. Keypoints dienen nachgelagert u. a. für Pose Detection, Visual Servoing
  oder Object Classification.
- Mindestens drei State-of-the-Art-Modelle sollten recherchiert und evaluiert
  werden; bei ausreichend Zeit sollten zwei Modelle verglichen werden.
- Bewertungskriterien: Vorgehen/Fortschritt gegenüber State of the Art
  (Kreativität), Umsetzung und Validierung, sowie Präsentation der Ergebnisse.
- Die Aufgabe ist als Gruppenarbeit (max. 4 Personen) angelegt und zählt als
  Prüfungsvorleistung (20 % der Endnote).

---

## 2. Modellauswahl: Warum Key-Grid?

Die Aufgabenstellung nannte UKPGAN und SkeletonMerger als **Beispielmodelle**
(siehe Abschnitt 1) – diese waren nicht verpflichtend vorgegeben, sondern
sollten zur Orientierung dienen. Im Rahmen der eigenen Recherche nach
State-of-the-Art-Modellen für 3D Keypoint Detection auf Punktwolken wurden
diese Beispielmodelle sowie zwei weitere recherchierte Kandidaten
gemeinsam betrachtet:

| Modell | Venue | Kernidee |
|---|---|---|
| UKPGAN | CVPR 2022 | Autoencoder + GAN, SE(3)-invariante, rotationsinvariante lokale Features |
| SkeletonMerger | CVPR 2021 | Skelett aus allen Keypoint-Paaren, Composite Chamfer Distance |
| **Back to 3D (B2-3D)** | CVPR 2024 | Rückprojektion von 2D-Foundation-Model-Features (z. B. DINO) auf 3D-Shapes, Few-Shot-fähig |
| A Fast and Lightweight 3D Keypoint Detector | IJCV 2025 | leichtgewichtige Saliency-Map aus geometrischen und semantischen Cues |
| **Key-Grid** | NeurIPS 2024 | Grid-Heatmap-Repräsentation des Keypoint-Skeletts statt roher Koordinaten |

Aus diesen fünf Kandidaten haben wir uns für **Key-Grid** entschieden.

**Begründung:**

- Aktuellster Stand der Technik unter allen fünf Kandidaten (NeurIPS 2024),
  bei vergleichbarer Aktualität zu Back to 3D (CVPR 2024); deutlich neuer
  als UKPGAN (2022) und SkeletonMerger (2021).
- Im Gegensatz zu Back to 3D, das auf vortrainierten 2D-Foundation-Model-
  Features (DINO) aufbaut und für Few-Shot-Szenarien mit wenig Labeling
  ausgelegt ist, arbeitet Key-Grid vollständig unsupervised und benötigt
  keine zusätzlichen 2D-Bild-Daten oder Foundation-Models – das passte besser
  zu unserem Trainings-Setup (reine 3D-Punktwolken aus KeypointNet).
- Konzeptionell baut Key-Grid auf der SkeletonMerger-Idee auf (Skelett durch
  Verbindung aller Keypoint-Paare), erweitert diese aber um eine
  **Grid-Heatmap-Repräsentation**: Statt die Keypoint-Koordinaten direkt zur
  Rekonstruktion zu nutzen, wird die Information in ein dichtes,
  geometrisches Feature-Feld eingebettet.
- Im Unterschied zu UKPGAN (zusätzlicher GAN-Diskriminator, C++-Build mit
  PCL/Pybind11 nötig) hat Key-Grid eine geringere Einstiegshürde und ist
  rein in Python/PyTorch umgesetzt.
- Dadurch bleiben die erkannten Keypoints auch bei **deformierbaren Objekten**
  (z. B. Kleidung) semantisch konsistent – ein Problem, an dem ältere,
  rein koordinatenbasierte Modelle laut Paper scheitern.
- Key-Grid erreicht laut Paper State-of-the-Art-Ergebnisse bei semantischer
  Konsistenz und Positionsgenauigkeit auf dem KeypointNet-Benchmark und ist
  zudem robust gegenüber Rauschen und Downsampling.


Vollständige Quellenangaben zu allen vier in diesem Kapitel genannten
Modellen (UKPGAN, SkeletonMerger, Back to 3D, Key-Grid)
finden sich gesammelt im **Quellenverzeichnis, Kapitel 10**.

---

## 3. Architektur (Kurzüberblick)

Key-Grid ist ein Autoencoder mit PointNet++ als Backbone:

1. **Encoder (PointNet++):** verarbeitet die Eingabe-Punktwolke hierarchisch
   über mehrere Set-Abstraction-Stufen.
2. **Keypoint-Vorhersage:** jeder Keypoint wird als gewichtete Summe aller
   Eingabepunkte berechnet (Point-Score-Matrix + Softmax).
3. **Skeleton:** alle Keypoint-Paare werden verbunden (vollständiger Graph);
   ein zweiter Encoder-Kopf sagt zusätzlich Kantengewichte vorher.
4. **Grid Heatmap:** ein 3D-Gitter über den Objektraum erhält pro Zelle einen
   Wert basierend auf dem Abstand zur nächsten Skelett-Linie.
5. **Decoder:** rekonstruiert die Punktwolke aus der Heatmap sowie über
   Skip-Verbindungen aus den Encoder-Zwischenstufen (coarse-to-fine).

### 3.1 Wie die Heatmap konkret in den Decoder einfließt

Wichtig für das Verständnis: In den Decoder fließt **nicht nur eine**
Information ein, sondern an jedem Combination-Schritt (✚) **drei**
gleichzeitig, die dort zusammengeführt werden:

1. die **durchlaufende Information aus der vorherigen Decoder-Stufe**
   (die bereits teil-rekonstruierte, gröbere Form),
2. eine **kopierte Encoder-Zwischenstufe** (über die Skip-Verbindung), und
3. eine **aus der Grid Heatmap gezogene Stichprobe**.

(Details und die genaue Funktionsweise dieser drei Quellen folgen in
Abschnitt 3.1.2 weiter unten.)

Das Architekturbild zeigt das vereinfacht – tatsächlich gibt es **drei**
solcher Kombinationsschritte im Decoder (die erste, gröbste Stufe wird
direkt vom Encoder übernommen, ohne eigenen Combination-Schritt):

| Decoder-Stufe | Kopierte Encoder-Stufe | Größe (Punkte) |
|---|---|---|
| Stufe 1 | PointNet++ Stufe „(16, 1024)" | 16 |
| Stufe 2 | PointNet++ Stufe „(64, 512)" | 64 |
| Stufe 3 | PointNet++ Stufe „(256, 256)" | 256 |
| Stufe 4 | PointNet++ Stufe „(1024, 96)" | 1024 |

Damit die **Combination** (durchlaufende Info + Kopie + Heatmap-Stichprobe)
überhaupt
funktioniert, muss die Anzahl der aus der Heatmap gezogenen Punkte **exakt**
zur Punktanzahl der jeweils kopierten Encoder-Stufe passen – die
Matrizen müssen also gleich groß sein. Die zugrunde liegende
Abstand-zum-Skelett-Formel bleibt dabei über alle Stufen hinweg gleich
(„coarse-to-fine"); es werden lediglich unterschiedlich viele Punkte aus
diesem einen Feld gezogen: 64 für die gröbste Decoder-Stufe, dann 256, dann
1024 für die feinste.

#### 3.1.1 Was in den PointNet++-Blöcken wirklich steckt: Positionen *und* Features

Im Architekturbild sind die farbigen PointNet++-Blöcke (z. B. „64, 512")
nur mit ihrer **Feature**-Information beschriftet. Tatsächlich gehört zu
jedem Block immer ein **Paar** aus zwei Dingen:

```
Block = (Positionen [wo],  Features [was])
```

Die Zahl „64" steht für 64 ausgewählte Punkte, „512" für die
Feature-Tiefe pro Punkt – aber zu diesen 64 Punkten gehören unsichtbar
auch ihre **64 Positionen** im 3D-Raum. Das Diagramm zeigt sie nicht
explizit, weil sie sonst jeden Block doppelt darstellen müssten.

**Woher die Positionen kommen:** Beim Downsampling per Farthest Point
Sampling (siehe Abschnitt zu PointNet++) wählt der Encoder gezielt z. B.
64 möglichst weit verteilte Punkte aus den ursprünglichen Punktwolken-
Koordinaten aus. Diese 64 Punkte haben also echte, aus der
Original-Punktwolke stammende Positionen – nicht irgendwelche
berechneten Werte. Wenn der Decoder später die 64er-Ebene rekonstruiert,
nutzt er **dieselben** 64 Positionen, die der Encoder ursprünglich
ausgewählt hat. Sie werden also mitgeführt und an die passende
Decoder-Stufe weitergereicht – quasi eine **zweite, unsichtbare
Skip-Connection**, nur für Positionen statt für Features.

**Jede Stufe hat ihr eigenes, separates Shared MLP.** Zwischen den
Blöcken passiert die eigentliche Verarbeitung – und jede der vier
Set-Abstraction-Stufen (96→256→512→1024) verwendet dabei **ein eigenes**
Shared MLP mit eigenen, unabhängig trainierten Gewichten:

```
Stufe 1 (1024 → 96):   eigener Gewichtssatz A
Stufe 2 (256 → 256):   eigener Gewichtssatz B
Stufe 3 (64 → 512):    eigener Gewichtssatz C
Stufe 4 (16 → 1024):   eigener Gewichtssatz D
```

Wichtig dabei: "Shared" bezieht sich nur auf die Wiederverwendung
**innerhalb** einer Stufe (dasselbe MLP wird auf jeden Punkt dieser
Stufe angewendet, daher permutationsinvariant) – **nicht** auf eine
gemeinsame Nutzung **über alle Stufen hinweg**. Stufe 1 "weiß" nichts
von den Gewichten aus Stufe 2, und umgekehrt. Auf der Decoder-Seite gilt
dasselbe: Jede der drei MLP-Blöcke nach den Combination-Schritten
(✚ → MLP) hat ebenfalls eigene, unabhängige Gewichte, die separat
trainiert werden.

#### 3.1.2 Die drei Quellen, die im Decoder zusammenfließen

An jeder **Combination**-Stelle (✚) treffen nicht zwei, sondern
konzeptionell **drei** Informationsquellen aufeinander:

```
interpolierte Stufe [grobe Form]  ∥  Skip-Kopie [Detail]  ∥  Grid Heatmap [wo die Keypoints sind]  →  MLP
```

Die Grid Heatmap ist dabei der entscheidende dritte Baustein: Sie wird
aus den Keypoint-Positionen berechnet (siehe Architektur-Überblick oben)
und ist damit die Stelle, an der die Keypoint-Information zum ersten Mal
in die Rekonstruktion einfließt.

**Warum das die Keypoints "zwingt", gut zu sein:** Dadurch wird die
Keypoint-Position zu einem echten Baustein der Rekonstruktion – mit
folgender Wirkkette:

1. Keypoints schlecht platziert → Heatmap zeigt auf die falschen Stellen
   → Rekonstruktion wird schlechter → Loss steigt
2. Das Training minimiert den Loss → es schiebt die Keypoints an
   Stellen, die der Rekonstruktion am meisten helfen
3. Welche Stellen helfen am meisten? Die geometrisch markanten,
   charakteristischen Stellen des Objekts (z. B. Ecken, Spitzen, Gelenke)
4. → genau das sind die Eigenschaften, die gute Keypoints ausmachen

Die Heatmap fungiert damit als die "Leine", die die Keypoints an die
Rekonstruktion koppelt – ohne sie gäbe es keinen Druck, semantisch
sinnvolle Positionen zu finden.

**Analogie:** Drei transparente Folien übereinander auf einem Projektor –
Folie A zeigt die grobe Form (interpoliert), Folie B die scharfen Details
(Skip-Kopie), Folie C leuchtende Markierungen an den Keypoint-Stellen
(Heatmap). Das Übereinanderlegen entspricht der Concatenation; das MLP
ist der Betrachter, der alle drei Folien gleichzeitig liest und daraus
ein einziges, stimmiges Bild formt. Damit man die Folien überhaupt
stapeln kann, müssen alle drei dasselbe Format (dieselbe Punktanzahl)
haben – genau das leistet der in Abschnitt 3.1 beschriebene
Sampling-Schritt, der die Heatmap auf die passende Punktzahl bringt.

### 3.2 Sigma und weitere Hyperparameter der Grid Heatmap

Die Breite der Grid Heatmap wird über einen Sigma-Parameter gesteuert
(Gauß-artiger Abfall: $\exp(-d^2/\sigma^2)$, je größer der Abstand $d$
zum Skelett, desto schwächer der Heatmap-Wert).

**Fundstelle im Paper:** Gleichung 5, Abschnitt 3.2, am Ende von Seite 4 /
Anfang von Seite 5.

**Fundstelle im Code:** `merger_net.py`, in der kommentierten Version kurz
vor `def forward`, Zeile 159:

```python
heatmaps = torch.exp(- squared_dist / 2.5e-3)
```

Sigma ist hier **nicht als benannte Variable** ausgelagert, sondern als
fester Wert `2.5e-3` direkt im Code hinterlegt.

**Wichtige Klarstellung – Sigma wirkt nicht auf den Farthest-Point-Loss:**
Sigma steuert ausschließlich die Breite der Heatmap und damit den
**Rekonstruktions-Pfad**. Der Farthest-Point-Loss (der die Keypoints zu
den FPS-Ankern zieht) enthält kein Sigma und bleibt davon unberührt –
ein Verändern von Sigma ändert die Anziehungskraft der Anker nicht
direkt.

**Die direkten Hebel gegen den Anker-Zug** (bereits als Trainings-Flags
in `train.py` vorhanden):

| Argument | Paper-Symbol | Wirkung |
|---|---|---|
| `--lambda_init_points` | $\alpha_{far}$ | Stärke des FPS-Anker-Zugs. Kleiner → Keypoints werden schwächer zu den Ankern gezogen → mehr Freiheit für strukturrelevante Lagen. **Wichtigster Hebel.** |
| `--lambda_chamfer` | $\alpha_{sim}$ | Gewicht der Rekonstruktion. Relativ größer → Rekonstruktion darf Keypoints stärker dorthin ziehen, wo sie Struktur braucht. |
| `--keynumber` | $J$ | Anzahl der FPS-Anker (aktuell 12). Beeinflusst, wie viele "Außenziele" es überhaupt gibt. |

Das Verhältnis `lambda_chamfer / lambda_init_points` ist die naheliegendste
Stellschraube für Fragen rund um Keypoint-Kollaps bzw. -Verteilung.

**Wo Sigma trotzdem indirekt wirken kann:** Ein plausibler, aber
indirekter und nicht zu 100 % gesicherter Effekt:

- *Kleineres Sigma* (schärfere Heatmap) → die Heatmap "leuchtet" nur nah
  an den Skelett-Segmenten → eine kollabierte/spärliche Skelettstruktur
  ließe große Teile des Objekts "dunkel" → die Rekonstruktion wäre
  schlechter, außer die Keypoints verteilen sich → mehr Druck zu
  distinkten Keypoints.
- *Größeres Sigma* (breite Heatmap) → schon wenige Keypoints decken
  alles ab → kollaps-toleranter.

Dieser Effekt wird jedoch durch die Skip-Connections gedämpft: Der
Decoder bekommt Encoder-Features direkt, kann also auch bei "dunkler"
Heatmap noch passabel rekonstruieren. Sigma ist daher ein **sekundärer**,
kein verlässlicher Hebel für die Keypoint-Verteilung.

### 3.3 Wie die Loss-Funktionen konkret wirken

**Chamfer-Distance-Loss:** Da Punktwolken ungeordnet sind, lassen sich
Original- und rekonstruierte Punktwolke nicht Punkt-für-Punkt vergleichen.
Stattdessen wird für **jeden** Punkt der Original-Punktwolke der **nächste**
Punkt in der Rekonstruktion gesucht und der quadrierte Abstand dazwischen
berechnet – und das **in beide Richtungen** (Original→Rekonstruktion und
Rekonstruktion→Original):

```
L_chamfer = Σ min‖x−y‖²  (für jedes x im Original, y in Rekonstruktion)
          + Σ min‖y−x‖²  (für jedes y in Rekonstruktion, x im Original)
```

Beide Richtungen sind notwendig: Die erste Richtung bestraft, wenn ein Teil
der Originalform in der Rekonstruktion fehlt; die zweite Richtung bestraft,
wenn der Decoder Punkte „erfindet", die im Original nicht existieren. Nur
über die Summe beider Richtungen werden beide Fehlerarten erfasst.

**Farthest-Point-Loss:** Dieser Loss wirkt **nicht** auf die Rekonstruktion,
sondern direkt auf die Positionen der k vorhergesagten Keypoints. Er
bestraft Konfigurationen, bei denen Keypoints zu nah beieinander liegen, und
erzwingt dadurch eine gleichmäßige räumliche Verteilung über das Objekt.
Ohne diesen Regularisierungsterm könnte der Chamfer-Loss allein dazu führen,
dass mehrere Keypoints auf einen für die Rekonstruktion „günstigen" Bereich
kollabieren.

**Gesamt-Loss:**

```
L = λ_chamfer · L_chamfer + λ_fps · L_fps
```

Der Gewichtungsfaktor λ_fps steuert, wie stark die räumliche Verteilung
(Farthest-Point-Loss) gegenüber der reinen Rekonstruktionsgüte
(Chamfer-Loss) gewichtet wird.

Im Training wurden beide Gewichte auf den Standardwert `1.0` gesetzt: `--lambda_init_points 1.0` und `--lambda_chamfer 1.0`.

Weitere Details und Diagramme siehe Präsentationsfolien in
`docs/Praesentation.pdf`. **ToDo: Pfad anpassen**

---

## 4. Installation

### 4.1 Voraussetzungen

| Komponente | Version / Hinweis |
|---|---|
| Betriebssystem | Windows 11 |
| Python | 3.10 |
| PyTorch | 2.5.1 |
| CUDA | 11.8 |
| GPU | CUDA-fähig zwingend erforderlich (genutzt: NVIDIA GeForce RTX 2070) |
| Umgebung | Conda-Umgebung `keygrid` |

### 4.2 Schritt-für-Schritt-Anleitung

```bash
# 1. Conda-Umgebung anlegen
conda create -n keygrid python=3.10
conda activate keygrid

# 2. PyTorch mit passender CUDA-Version installieren
conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia

# 3. Key-Grid Repository klonen
git clone https://github.com/JackHck/Key-Grid.git
cd Key-Grid

# 4. Abhängigkeiten installieren
# Hinweis: Es existiert KEINE requirements.txt im Original-Repo,
# Pakete mussten manuell installiert werden:
pip install scikit-learn open3d pyvista seaborn einops wandb matplotlib
pip install --extra-index-url https://miropsota.github.io/torch_packages_builder pytorch3d==0.7.8+pt2.5.1cu118

# 5. Datensatz herunterladen (KeypointNet)
# Repos klonen:
git clone https://github.com/qq456cvb/KeypointNet.git
# Anschließend pcds/ und annotations/ manuell über den Google Drive / OneDrive
# Link im KeypointNet README herunterladen und in den KeypointNet-Ordner legen.
```

### 4.3 Verzeichnisstruktur

Da im Projekt **drei Objektkategorien** trainiert wurden (Flugzeuge,
Schiffe, Motorräder), ist die Datensatz- und Checkpoint-Struktur
entsprechend pro Kategorie organisiert. Eine vierte Kategorie (Cap)
ist als Zukunftsidee geplant, aber noch nicht umgesetzt (Begründung
siehe Kapitel 10.1).

```
Key-Grid/
├── merger/
│   ├── merger_net.py
│   ├── composed_chamfer.py
│   ├── data_flower.py
│   └── pointnetpp/
├── model/
│   ├── airplane.pt              ← k=8, 100 Epochen
│   ├── airplane_v2.pt           ← k=8, 150 Epochen, chamfer=10
│   ├── airplane_v3_k10.pt       ← k=10, 150 Epochen, chamfer=10  ← bestes Modell
│   ├── airplane_v4_k12.pt       ← k=12, 150 Epochen, chamfer=10
│   ├── motorbike.pt             ← k=8, 100 Epochen
│   └── vessel.pt                ← k=8, 100 Epochen
├── results/
│   ├── airplane_v2_keypoints.npy
│   ├── airplane_v3_k10_keypoints.npy
│   ├── airplane_v4_k12_keypoints.npy
│   └── visualizations_*/
├── train.py                           ← angepasst für KeypointNet
├── predict_keypoint.py                ← angepasst für KeypointNet
├── vision.py                          ← angepasst für KeypointNet
├── visualizations.py                  ← pyvista-Fix eingebaut
├── predict_keypoint_annotated.py      ← neu: Vorhersage in Annotationsreihenfolge
├── evaluate_airplane_v2.py            ← neu: Test Evaluierung k=8
├── evaluate_airplane_v3_k10.py        ← neu: Test Evaluierung k=10
├── evaluate_airplane_v4_k12.py        ← neu: Test Evaluierung k=12 + Vergleich
├── evaluate_ood.py                    ← neu: Out-of-Distribution Test
└── compare_training.py                ← neu: Key-Grid vs Skeleton Merger

KeypointNet/
├── pcds/
│   ├── 02691156/   ← Airplane  (1022 Modelle)
│   ├── 03790512/   ← Motorbike
│   └── 04530566/   ← Vessel    (910 Modelle)
└── annotations/
    ├── airplane.json
    ├── motorcycle.json
    ├── vessel.json
    └── ...

SkeletonMerger/
├── merger/
├── model/
│   ├── sm_airplane.pt
│   ├── sm_motorbike.pt
│   └── sm_vessel.pt
├── train.py         ← angepasst für KeypointNet
└── sm_predict.py    ← neu: Vorhersage für KeypointNet
```

---

## 5. Änderungen am Originalcode

Der ursprüngliche Key-Grid-Quickstart ist auf den **ClothesNet**-Datensatz
ausgelegt (Trainingsbeispiel: Chair-Kategorie aus ShapeNetCoreV2, Vorhersage
auf ClothesNet). Für unsere Aufgabenstellung war jedoch eine Evaluation gegen
den **KeypointNet**-Datensatz gefordert (mIoU-Berechnung gegen menschlich
annotierte Ground-Truth-Keypoints). Daher waren folgende Anpassungen
notwendig:

| Datei | Was wurde geändert | Warum |
|---|---|---|
| `train.py` | Datenladen von `.txt` (ClothesNet) auf `.pcd` (KeypointNet) via `open3d`; `--dataset-root` und `--checkpoint-path` als Argumente; 70/15/15 Datensplit; Validation nach jeder Epoche; lokales JSON-Logging + optionales wandb | Original war auf ClothesNet hardcodiert, kein Split, kein Logging |
| `predict_keypoint.py` | Datenladen von `.txt` auf `.pcd`; batched Verarbeitung; Ausgabe als `.npy` | Original lud eine einzelne `.txt` Datei |
| `vision.py` | Datenladen von `.txt` auf `.pcd` Dateien und `.npy` Keypoints; Pfade als Konstanten konfigurierbar | Original war auf ClothesNet-Pfade hardcodiert |
| `visualizations.py` | `plotter.camera_parallel_projection = True` → `plotter.camera.parallel_projection = True`; `plotter.view_xy()` → `plotter.close()` | Inkompatibilität mit neuerer pyvista-Version; Speicherleck bei 1000+ Renderings |
| `SkeletonMerger/train.py` | Analog zu Key-Grid: `.pcd` Laden, 70/15/15 Split, Validation | Original war auf `.h5` Dateien ausgelegt |
| `sm_predict.py` (neu) | Vorhersage für Skeleton Merger auf KeypointNet `.pcd` Dateien | Original benötigte `.h5` Dateien |
| `predict_keypoint_annotated.py` (neu) | Vorhersage in Reihenfolge der KeypointNet Annotationen; speichert `model_id` für PCK-Vergleich mit Ground Truth | Für Ground-Truth-Vergleich muss Reihenfolge mit `annotations/airplane.json` übereinstimmen |
| `evaluate_airplane_v2/v3/v4.py` (neu) | Test-Evaluierung: Chamfer Distance, Keypoint Spread, Konsistenz, Surface Coverage; Plot-Ausgabe | Im Original nicht vorhanden |
| `evaluate_ood.py` (neu) | Out-of-Distribution Test: Airplane-Modell auf Mug-Daten | Im Original nicht vorhanden |
| `compare_training.py` (neu) | Paralleles Training Key-Grid vs Skeleton Merger mit Metrik-Vergleich und Plots | Im Original nicht vorhanden |

> **Hinweis für die Doku-Pflege:** Bitte jede Code-Änderung hier ergänzen,
> sobald sie gemacht wird – nicht erst am Ende rekonstruieren. Am besten
> direkt beim Commit eine Zeile in dieser Tabelle ergänzen.

---

## 6. Aufgetretene Probleme & Lösungen

| Problem | Ursache | Lösung |
|---|---|---|
| `pip install` schlägt fehl mit `dataclass() got an unexpected keyword argument 'slots'` | pip selbst war in der Conda-Umgebung kaputt durch Python-Versions-Mismatch | Umgebung neu erstellen mit Python 3.10; PyTorch über `conda install` statt `pip` |
| `pytorch3d` Installation schlägt fehl mit `unexpected data class slots` | Versions-Mismatch zwischen pytorch3d-Build und installierter PyTorch-Version | Vorkompiliertes Paket von `miropsota.github.io/torch_packages_builder` für exakte Kombination `pt2.5.1cu118` nutzen |
| `ModuleNotFoundError: No module named 'einops'` | `einops` nicht im Original-Repo als Abhängigkeit dokumentiert | `pip install einops` |
| `pyvista.core.errors.PyVistaAttributeError: Attribute 'camera_parallel_projection' does not exist` | Inkompatibilität mit neuerer pyvista-Version | `plotter.camera_parallel_projection = True` → `plotter.camera.parallel_projection = True` |
| Speicherfehler nach ~160 Renderings (`Unable to allocate ... bytes`) | pyvista Plotter wurde nie geschlossen, RAM lief voll | `plotter.view_xy()` → `plotter.close()` nach jedem Screenshot |
| Keypoint-Kollaps in frühen Epochen | Zufällige Gewichtsinitialisierung, Farthest-Point-Loss noch nicht eingeschwungen | Über Epochen 0–9 zunächst nur Farthest-Point-Loss aktiv, Chamfer-Loss erst ab Epoche 10 zugeschaltet (`--chamfer 10`) |
| `TypeError: Object of type Tensor is not JSON serializable` | Loss-Werte wurden als PyTorch Tensor statt als Python float in JSON geschrieben | `float(train_loss)` bei allen Loss-Werten vor dem JSON-Schreiben |
| `training_log.json` wird beim nächsten Training überschrieben | Der Log-Pfad in `train.py` ist fest an den Checkpoint-Ordner gekoppelt und trägt keinen modellspezifischen Namen | Vor jedem neuen Training den `--checkpoint-path` anpassen (z. B. `model/airplane_v3_k10.pt`) – der Log landet dann automatisch unter `model/training_log.json` im gleichen Ordner. Alternativ den Log nach dem Training manuell umbenennen, z. B. `model/training_log_airplane_v3_k10.json` |

---

## 7. Trainingskonfiguration

Trainiert wurden drei Objektkategorien aus dem KeypointNet-Datensatz:
**Flugzeuge (Airplane), Schiffe (Vessel) und Motorräder (Motorcycle)**.
Eine vierte Kategorie (Cap / Mützen) ist als Zukunftsidee geplant,
wurde im Rahmen dieser Arbeit aber noch nicht umgesetzt (Begründung
siehe Kapitel 10.1).
Die Hyperparameter wurden dabei über alle trainierten Kategorien
konstant gehalten.

**Wichtig:** Eine vollständige **Evaluation** (mIoU, DAS, Precision,
Recall, Chamfer-Distance-Verteilung) wurde ausschließlich für
**Airplane** durchgeführt. Vessel und Motorcycle wurden zwar trainiert,
aber noch nicht gegen die KeypointNet-Ground-Truth ausgewertet.

| Parameter | Airplane | Airplane V2 | Airplane V3 | Airplane V4 | Motorbike | Vessel |
|---|---|---|---|---|---|---|
| Checkpoint | `airplane.pt` | `airplane_v2.pt` | `airplane_v3_k10.pt` | `airplane_v4_k12.pt` | `motorbike.pt` | `vessel.pt` |
| Keypoints (k) | 8 | 8 | 10 | 12 | 8 | 8 |
| Epochen | 100 | 150 | 150 | 150 | 100 | 100 |
| Batch-Size | 8 | 16 | 16 | 16 | 16 | 16 |
| Chamfer-Aktivierung | Epoche 20 | Epoche 10 | Epoche 10 | Epoche 10 | Epoche 20 | Epoche 20 |
| Datensplit | 70/15/15 | 70/15/15 | 70/15/15 | 70/15/15 | 70/15/15 | 70/15/15 |
| lambda_init_points | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| lambda_chamfer | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| Evaluiert | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ |

Trainingsbefehl (je Modell):

```bash
# Airplane (original, k=8)
python train.py --dataset-root C:\Users\xrstu\KeypointNet\pcds\02691156 --checkpoint-path model/airplane.pt --epochs 100 -b 8

# Airplane V2 (k=8, verbesserte Parameter)
python train.py --dataset-root C:\Users\xrstu\KeypointNet\pcds\02691156 --checkpoint-path model/airplane_v2.pt --epochs 150 --chamfer 10 -b 16

# Airplane V3 (k=10, Paper-Empfehlung) ← bestes Modell
python train.py --dataset-root C:\Users\xrstu\KeypointNet\pcds\02691156 --checkpoint-path model/airplane_v3_k10.pt --epochs 150 --chamfer 10 -b 16 --n-keypoint 10

# Airplane V4 (k=12)
python train.py --dataset-root C:\Users\xrstu\KeypointNet\pcds\02691156 --checkpoint-path model/airplane_v4_k12.pt --epochs 150 --chamfer 10 -b 16 --n-keypoint 12

# Motorbike
python train.py --dataset-root C:\Users\xrstu\KeypointNet\pcds\03790512 --checkpoint-path model/motorbike.pt --epochs 100 -b 16

# Vessel
python train.py --dataset-root C:\Users\xrstu\KeypointNet\pcds\04530566 --checkpoint-path model/vessel.pt --epochs 100 -b 16
```

### 7.1 Zweiphasiges Training: Loss-Aktivierung

Das Training läuft nicht von Anfang an mit beiden Loss-Termen gleichzeitig,
sondern in **zwei Phasen**:

| Phase | Epochen | Aktive Loss-Terme | Zweck |
|---|---|---|---|
| **Phase 1** | 0 – 9 | nur Farthest-Point-Loss | Keypoints werden zunächst rein räumlich über das Objekt verteilt, ohne Rücksicht auf Rekonstruktionsgüte |
| **Phase 2** | 10 – 149 | Farthest-Point-Loss + Chamfer-Distance-Loss | Sobald die Keypoints räumlich sinnvoll verteilt sind, wird zusätzlich die Rekonstruktion trainiert |

Gesteuert wird dieses Verhalten über das Trainings-Flag **`--chamfer 10`**:
Die Chamfer-Distance-Loss wird erst ab der angegebenen Epoche (hier: 10,
0-indexiert) aktiviert, davor läuft das Training ausschließlich mit dem
Farthest-Point-Loss. Phase 1 umfasst damit 10 Epochen (0–9), Phase 2 die
verbleibenden 140 Epochen (10–149) bis zum Trainingsende.

**Warum dieses Vorgehen sinnvoll ist:** Würde der Chamfer-Loss von Anfang an
mitlaufen, müsste der Decoder versuchen, aus noch zufällig verteilten,
unbrauchbaren Keypoints zu rekonstruieren – das liefert ein verrauschtes,
wenig hilfreiches Lernsignal. Indem zunächst nur der Farthest-Point-Loss
aktiv ist, bekommen die Keypoints eine sinnvolle räumliche Grundordnung,
bevor der Decoder überhaupt einbezogen wird.

**Sichtbar in den eigenen Trainingsergebnissen:** Die Loss-Kurve aus den
Ergebnissen zeigt diesen Übergang deutlich – die Chamfer-Komponente liegt
bis Epoche 10 konstant bei null und springt danach sprunghaft auf einen
Wert von ca. 0,0175, bevor sie über die folgenden Epochen kontinuierlich
sinkt. Dieser Sprung ist **kein Trainingsproblem**, sondern die direkte
Folge der bewussten Phasenumschaltung – das Modell sieht zum ersten Mal
überhaupt ein Rekonstruktionssignal.

**ToDo**: Trainingskurven-Grafik (Gesamt-Loss Train/Val sowie
Chamfer-Komponente, analog zur Präsentation) direkt als Bild hier
einbinden, z. B.:

```markdown
![Loss-Kurve Train/Val mit Phasenübergang](results/loss_curve_airplane.png)
```

---

## 8. Ergebnisse (Kurzüberblick)

- Mittlere Chamfer Distance (Airplane, k=10, In-Distribution): **0,0006**
- Bestes Validierungsergebnis bei Epoche 138
- Modellvergleich nach Keypoint-Anzahl (k = 8 / 10 / 12): mIoU 43 / 59 / 47,
  DAS 72 / 72 / 75 — k = 10 als bester Kompromiss identifiziert
- Out-of-Distribution-Test (Modell trainiert auf Airplane, getestet auf Mug):
  deutlich höhere Chamfer Distance (0,035) – siehe Diskussion
  „Problemstellungen"

Ausführliche Ergebnisse, Diagramme und Diskussion: siehe
`docs/Praesentation.pdf`, Kapitel 4.
**ToDo**: Pfad anpassen, ggf. Ergebnis-Tabellen/Plots direkt in `results/` ablegen.

---

## 9. Reproduktion der Ergebnisse

Am Beispiel von **Airplane V3 (k=10)** – dem besten Modell:

### Schritt 1 – Training

```bash
conda activate keygrid
cd Key-Grid

python train.py \
  --dataset-root C:\Users\xrstu\KeypointNet\pcds\02691156 \
  --checkpoint-path model/airplane_v3_k10.pt \
  --epochs 150 \
  --chamfer 10 \
  -b 16 \
  --n-keypoint 10
```

Das Modell wird nach jeder Epoche unter `model/airplane_v3_k10.pt` gespeichert.
Das Training-Log (Loss pro Epoche) liegt unter `model/training_log.json`.

> ⚠️ **Wichtig:** Das `training_log.json` wird bei jedem neuen Training **überschrieben**, sofern der `--checkpoint-path` auf denselben Ordner zeigt. Vor dem Start eines neuen Trainings das bestehende Log umbenennen, z. B.:
> ```bash
> rename model\training_log.json training_log_airplane_v2.json
> ```

### Schritt 2 – Keypoints vorhersagen

```bash
python predict_keypoint.py \
  --checkpoint-path model/airplane_v3_k10.pt \
  --output results/airplane_v3_k10_keypoints.npy \
  --n-keypoint 10
```

### Schritt 3 – Visualisierung der erkannten Keypoints

In `vision.py` die folgenden Konstanten setzen:

```python
DATASET_ROOT  = r'C:\Users\xrstu\KeypointNet\pcds\02691156'
KEYPOINTS_FILE = r'results\airplane_v3_k10_keypoints.npy'
OUTPUT_DIR    = r'results\visualizations_airplane_v3_k10'
N_KEYPOINTS   = 10
MAX_POINTS    = 2048
```

Dann ausführen:

```bash
python vision.py
```

Die PNG-Bilder mit eingezeichneten Keypoints erscheinen unter `results/visualizations_airplane_v3_k10/png/`.

### Schritt 4 – Evaluation

```bash
python evaluate_airplane_v3_k10.py
```

Ergebnisse (Chamfer Distance, Keypoint Spread, Konsistenz, Surface Coverage) werden
unter `results/evaluation_airplane_v3_k10/` als Plot und JSON gespeichert.

### Schritt 5 – Out-of-Distribution Test (optional)

```bash
python evaluate_ood.py \
  --model model/airplane_v3_k10.pt \
  --n-keypoint 10
```

### Ohne erneutes Training (Checkpoint direkt laden)

Der Checkpoint des besten Modells liegt unter `model/airplane_v3_k10.pt`.
Ab Schritt 2 kann direkt mit diesem Checkpoint weitergemacht werden, ohne 150 Epochen neu zu trainieren.

---

## 10. Ausblick / Zukünftige Arbeiten

### 10.1 Weitere Datensätze und direkter Modellvergleich

Die eigenen Ergebnisse wurden bisher ausschließlich für die Kategorie
**Airplane** vollständig evaluiert (mIoU, DAS, Precision, Recall,
Chamfer-Distance). Vessel und Motorcycle wurden zwar trainiert, aber
noch nicht gegen die KeypointNet-Ground-Truth ausgewertet. Cap ist als
Zukunftsidee geplant — bewusst als letztes, weil die Punktwolken von
Mützen auf Grund ihrer einfachen, annähernd rotationssymmetrischen Form
keine klar definierbaren, geometrisch markanten Stellen besitzen. Das
macht es für das Modell deutlich schwerer, semantisch sinnvolle Keypoints
zu finden — und damit auch schwerer einzuschätzen, ob schlechte Ergebnisse
auf das Modell oder auf die inhärente Mehrdeutigkeit der Kategorie
zurückzuführen sind. Cap eignet sich daher besonders gut als
**Stresstest**: Wenn Key-Grid auch hier konsistente Keypoints findet,
wäre das ein starkes Argument für die Robustheit der Heatmap-Repräsentation.
Folgende Erweiterungen wären sinnvolle nächste Schritte:

- **Deformierbare Objekte (ClothesNet):** Das Key-Grid-Paper zeigt in
  Folie 3 ("Vorteil einer Heatmap") einen visuellen Vergleich mit KD,
  SM und SC3K auf T-Shirt- und Hosen-Kategorien — diese Ergebnisse
  stammen direkt aus dem Paper, nicht aus eigenem Training. Sinnvolle
  Erweiterung: Key-Grid auf denselben ClothesNet-Kategorien selbst
  trainieren, um einen echten, selbst erzeugten Vergleich zu den
  Baseline-Modellen zu erhalten und den Vorteil der Heatmap-Repräsentation
  gegenüber älteren Ansätzen quantitativ zu belegen.
- **Robustheitstests (Rauschen, Downsampling, Rotation):** SC3K
  (ICCV 2023) testet seine Robustheit explizit gegenüber diesen drei
  Störungen und verwendet dafür denselben KeypointNet-Datensatz.
  Key-Grid auf denselben Störungsbedingungen zu evaluieren würde
  direkt vergleichbare Robustheitswerte liefern.
- **Systematischer Out-of-Distribution-Test:** Bisher wurde nur ein
  OOD-Fall getestet (Mug, Folie 13, Chamfer Distance 0,035 vs. 0,0006
  bei Airplane). Um eine verlässlichere Aussage über Generalisierung
  zu treffen, wären mehrere OOD-Kategorien (z. B. Chair, Table, Guitar)
  mit dem auf Airplane trainierten Modell sinnvoll.

### 10.2 Hyperparameter-Variationen

Folgende Parameter wurden im Rahmen dieser Arbeit konstant gehalten und
bieten sich für systematische Experimente an:

**Loss-Gewichtung:**
Laut Paper sind `lambda_init_points` ($\alpha_{far}$) und
`lambda_chamfer` ($\alpha_{sim}$) beide standardmäßig auf 1 gesetzt —
**das ist auch im Code so umgesetzt** (`train.py`, Default-Werte der
entsprechenden Argparse-Parameter).
Das Verhältnis `lambda_chamfer / lambda_init_points` ist die
naheliegendste Stellschraube für die Keypoint-Verteilung: Ein höheres
Verhältnis gibt dem Rekonstruktionssignal mehr Gewicht gegenüber dem
FPS-Anker-Zug, was die Keypoints tendenziell näher an geometrisch
markante Stellen treibt. Die Ablation im Paper zeigt bereits, dass
ohne `lambda_init_points` die Keypoints kollabieren und ohne
`lambda_chamfer` die Rekonstruktion leidet — eine feingranulare
Variation zwischen diesen Extremen wurde jedoch nicht publiziert.

**Sigma der Grid Heatmap:**
Der Sigma-Wert (`2.5e-3` in `merger_net.py`) ist im Code hardcoded
und wurde bisher nicht variiert. Kleineres Sigma erzeugt eine schärfere
Heatmap und erhöht potenziell den Druck zu distinkten, verteilten
Keypoints; größeres Sigma macht das Training stabiler, aber kollaps-
toleranter. Da Sigma kein direktes Trainings-Flag ist, müsste es direkt
im Code angepasst werden (siehe Abschnitt 3.2 für Details).

**Keynumber / Anzahl der FPS-Anker:**
Aktuell 12 (`--keynumber`). Weniger Anker → weniger Außenziele, die
Keypoints werden schwächer "auseinandergezogen"; mehr Anker → mehr
Verteilungsdruck, aber auch mehr Freiheitsgrade im Training.

**Keypoint-Anzahl (k):**
In dieser Arbeit wurde k = 8, 10 und 12 getestet (Folie 13). k = 10
erzielte den besten Kompromiss zwischen mIoU und Chamfer Distance.
Weitere Werte (z. B. k = 6, 15, 20) könnten zeigen, wie robust dieser
Befund ist und ab wann zu wenige oder zu viele Keypoints die Qualität
deutlich degradieren.

### 10.3 Umgang mit symmetrischen Objekten

Viele Objektkategorien aus dem KeypointNet-Datensatz (z. B. Airplanes,
Motorcycles) besitzen eine Spiegelsymmetrie. Das bringt eine
konzeptionelle Schwierigkeit mit sich: Der Chamfer-Distance-Loss
vergleicht nur **Mengen** von Punkten, nicht deren Identität — eine um
die Symmetrieachse gespiegelte Rekonstruktion ergibt denselben
Chamfer-Loss wie das Original. Das Modell hat daher **kein direktes
Lernsignal**, das eindeutig zwischen linker und rechter Seite
unterscheidet.

In der Literatur (z. B. KeypointDeformer) ist bekannt, dass
FPS-basierte Regularisierung tendenziell der Objektsymmetrie folgt —
semantisch werden linke und rechte Seite nicht explizit unterschieden,
sondern nur konsistent behandelt. Die beobachtete Streuung einzelner
Keypoint-Cluster in der Keypoint-Verteilung (Folie 10) könnte daher
nicht nur Positionsungenauigkeit, sondern auch **Seitenverwechslungen**
(linker statt rechter Flügel) bei einzelnen Instanzen widerspiegeln.

**Hinweis:** Ob und wie Key-Grid dieses Problem explizit adressiert,
konnte im Paper nicht bestätigt werden — dieser Abschnitt basiert auf
literaturgestützter Argumentation (KeypointDeformer, SC3K), nicht auf
einer direkten Paper-Aussage von Key-Grid selbst.

---

## 11. Quellen

- Hai, C. et al. „Key-Grid: Unsupervised 3D Keypoints Detection using Grid
  Heatmap Features", NeurIPS 2024.
  [arXiv:2410.02237](https://arxiv.org/abs/2410.02237)
- Wimmer, T. et al. „Back to 3D: Few-Shot 3D Keypoint Detection with
  Back-Projected 2D Features", CVPR 2024.
  [arXiv:2311.18113](https://arxiv.org/abs/2311.18113)
- You, Y. et al. „KeypointNet: A Large-Scale 3D Keypoint Dataset Aggregated
  From Numerous Human Annotations", CVPR 2020.
  [arXiv:2002.12687](https://arxiv.org/abs/2002.12687)
- Shi, R. et al. „Skeleton Merger: an Unsupervised Aligned Keypoint
  Detector", CVPR 2021.
- You, Y. et al. „UKPGAN: A General Self-Supervised Keypoint Detector",
  CVPR 2022.
- Qi, C. R. et al. „PointNet++: Deep Hierarchical Feature Learning on
  Point Sets in a Metric Space", NeurIPS 2017.

---

## 12. Lizenz / Hinweis

> Vollständige Lizenztexte und der detaillierte Status aller genutzten
> Drittanbieter-Komponenten (KeypointNet, Key-Grid, sowie die von Key-Grid
> übernommenen Abhängigkeiten Skeleton Merger und SC3K) finden sich in
> [`THIRD_PARTY_LICENSES.md`](./THIRD_PARTY_LICENSES.md).

- **KeypointNet-Datensatz:** bestätigt unter **MIT-Lizenz** veröffentlicht
  (siehe `LICENSE.md` im [Original-Repository](https://github.com/qq456cvb/KeypointNet)).
  Eine Nutzung und Weiterverarbeitung der Annotationen/Punktwolken in diesem
  Projekt ist damit lizenzrechtlich unproblematisch, sofern die Lizenz beim
  Repository mit angegeben wird.
- **Key-Grid-Code ([JackHck/Key-Grid](https://github.com/JackHck/Key-Grid)):**
  Im Rahmen der Recherche für dieses Dokument konnte **keine explizite
  LICENSE-Datei im Repository verifiziert werden** — das bedeutet nicht
  zwingend, dass keine existiert, sondern nur, dass sie sich nicht
  eindeutig über die durchgeführte Recherche bestätigen ließ.
  **ToDo: Vor der Abgabe direkt im Repository nachsehen** (Datei `LICENSE`
  oder `LICENSE.md` im Root-Verzeichnis), und falls vorhanden hier den
  genauen Lizenztyp eintragen. Ohne erkennbare Lizenz gilt Code
  standardmäßig als urheberrechtlich geschützt ("all rights reserved");
  in diesem Fall sollte in Rücksprache mit dem Betreuer geklärt werden,
  ob die Nutzung für die Studienarbeit dennoch zulässig ist (i. d. R. ja,
  da Key-Grid selbst "We are committed to releasing the code" im Paper
  ankündigt, aber explizit prüfen lohnt sich).
- **Eigener Code (Anpassungen, siehe Abschnitt 5):** ToDo: Lizenz für die
  eigenen Modifikationen festlegen, sofern das Repository öffentlich
  gestellt wird (z. B. ebenfalls MIT, in Anlehnung an die genutzten
  Abhängigkeiten).
