# Third-Party Licenses

Dieses Repository nutzt bzw. baut auf folgenden externen Code- und
Daten-Repositorien auf. Dieses Dokument fasst deren Lizenzstatus zusammen,
um den Anforderungen der jeweiligen Lizenzen (insb. Nennung des
Original-Copyrights bei MIT-lizenziertem Code) nachzukommen.

Diese Datei legt **keine eigene Lizenz für unseren Code** fest — sie
dokumentiert ausschließlich die Lizenzbedingungen der genutzten
Drittanbieter-Komponenten.

---

## 1. KeypointNet (Datensatz)

- **Repository:** https://github.com/qq456cvb/KeypointNet
- **Lizenz:** MIT License — explizit im Repository bestätigt: *"KeypointNet
  is released under the MIT license — see LICENSE.md."*
- **Genutzt für:** Punktwolken (`pcds/`) und menschlich annotierte
  Ground-Truth-Keypoints (`annotations/*.json`) als Trainings- und
  Evaluationsgrundlage.

```
MIT License

Copyright (c) 2021 Yang You

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

> Wortlaut 1:1 aus `LICENSE.md` im KeypointNet-Repository übernommen
> (https://github.com/qq456cvb/KeypointNet/blob/master/LICENSE.md),
> bestätigt durch direkte Einsicht der Datei.

---

## 2. Key-Grid (Hauptmodell, in diesem Projekt trainiert)

- **Repository:** https://github.com/JackHck/Key-Grid
- **Lizenz:** **Keine explizite LICENSE-Datei im Repository vorhanden**
  (Stand der Überprüfung: Datei-Listing des Repositories enthält nur
  `image/`, `merger/`, `README.md`, `train.py`, `vision.py`,
  `predict_keypoint.py`, `visualizations.py` – keine `LICENSE`-Datei; auch
  im "Resources"-Bereich der Repo-Übersichtsseite fehlt ein
  "License"-Eintrag, der bei vorhandener Lizenzdatei automatisch von
  GitHub angezeigt würde).
- **Rechtliche Einordnung:** Ohne erkennbare Lizenz gilt der Code nach
  Standard-Urheberrecht als **"all rights reserved"** – eine Weitergabe
  oder Veröffentlichung des (angepassten) Codes ist damit formal nicht
  durch eine offene Lizenz gedeckt.
- **Kontext:** Der Code wird im Rahmen einer **nicht-kommerziellen
  Studienarbeit** zu Lehrzwecken genutzt und angepasst (siehe Abschnitt 5
  der README für die konkreten Änderungen). Die Autoren haben den Code
  öffentlich zum Paper bereitgestellt, ohne formale Lizenzbedingungen zu
  spezifizieren.

> **Ich muss noch klären:** Vor einer öffentlichen Bereitstellung dieses Repositories
> (über den internen Studienkontext hinaus) empfiehlt sich Rücksprache mit
> dem Betreuer und/oder eine Anfrage bei den Key-Grid-Autoren, ob/wie eine
> Weitergabe des angepassten Codes zulässig ist.

**Wichtige Abgrenzung:** Die zugehörige Projekt-Webseite
(https://jackhck.github.io/keygrid.github.io/) steht unter einer
**Creative Commons Attribution-ShareAlike 4.0 International License**
(CC BY-SA 4.0). Das ist jedoch ein häufig wiederverwendetes
Webseiten-Template (sog. "Nerfies-Template", erkennbar am verlinkten
Quell-Repository `nerfies/nerfies.github.io`) und bezieht sich
ausschließlich auf das **HTML/CSS-Layout der Projekt-Webseite** (Abstract,
Bilder, Video) – **nicht** auf die eigentliche Python/PyTorch-
Implementierung im Code-Repository `github.com/JackHck/Key-Grid`, die wir
tatsächlich nutzen. Diese beiden Lizenzfragen sind unabhängig voneinander
zu betrachten; die Code-Lizenz bleibt wie oben beschrieben ungeklärt.

---

## 3. Von Key-Grid übernommene Abhängigkeiten

Key-Grid selbst gibt im eigenen README an: *"This code inherits some codes
from Skeleton Merger, SC3K."* Das heißt, über Key-Grid sind indirekt auch
Code-Teile dieser beiden Repositorien in diesem Projekt enthalten.

### 3.1 Skeleton Merger

- **Repository:** https://github.com/eliphatfs/SkeletonMerger
- **Lizenz:** **MIT License — bestätigt vorhanden** (Datei `LICENSE` im
  Repository-Root, von GitHub auch offiziell als "MIT license" im
  "Resources"-Bereich der Repo-Seite ausgewiesen).

```
MIT License

Copyright (c) 2021 北海若

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

> Wortlaut 1:1 aus der Original-Datei übernommen (Stand der Überprüfung).
> Der Name „北海若" ist der im Repository hinterlegte Name/Pseudonym des
> Copyright-Inhabers und wurde unverändert übernommen.

### 3.2 SC3K

- **Repository:** https://github.com/IIT-PAVIS/SC3K
- **Lizenz:** **Keine LICENSE-Datei im Repository vorhanden** (Datei-Listing
  enthält nur `config/`, `dataset/`, `images/`, `README.md` sowie
  Python-Skripte; im "Resources"-Bereich der Repo-Seite fehlt ein
  "License"-Eintrag — im direkten Vergleich zu Skeleton Merger, wo dieser
  Eintrag bestätigt erscheint).
- **Rechtliche Einordnung:** wie bei Key-Grid gilt ohne erkennbare Lizenz
  Standard-Urheberrecht ("all rights reserved").

---

## 4. Zusammenfassung

| Komponente | Lizenz | Status |
|---|---|---|
| KeypointNet (Datensatz) | MIT | ✅ vollständig bestätigt (Copyright: Yang You, 2021) |
| Key-Grid (Hauptcode) | keine erkennbare Lizenz | ⚠️ bestätigt ungeklärt |
| Skeleton Merger (indirekt über Key-Grid) | MIT | ✅ vollständig bestätigt |
| SC3K (indirekt über Key-Grid) | keine erkennbare Lizenz | ⚠️ bestätigt ungeklärt |

**Empfehlung:** Vor Veröffentlichung dieses Repositories außerhalb des
internen Studienkontexts die beiden offenen Punkte (⚠️ Key-Grid, SC3K)
klären – am einfachsten durch eine kurze Nachricht an die jeweiligen
Autoren über GitHub Issues.
