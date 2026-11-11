# Gjenkjenning av håndskrevne sifre med CNN

Et konvolusjonelt nevralt nettverk (CNN) trent i PyTorch på [MNIST-datasettet](http://yann.lecun.com/exdb/mnist/), med en nettside der du kan tegne et tall og få det gjenkjent.

**[Prøv det her](https://runesor.github.io/cnn-digit-recognition/)**: tegn et siffer med mus, finger eller Apple Pencil, og trykk «Gjett».

Modellen kjører direkte i nettleseren med [ONNX Runtime Web](https://onnxruntime.ai/docs/get-started/with-javascript/web.html), så nettsiden trenger ingen server.

## Resultat

**99,3 % nøyaktighet** på de 10 000 [testbildene i MNIST](https://huggingface.co/datasets/ylecun/mnist/viewer/mnist/test) etter 10 epoker.

## Modellen

```
Inndata: 1 × 28 × 28 (gråtonebilde)
  │
  ├─ Conv2d(1 → 10, 5×5) → MaxPool(2) → ReLU        → 10 × 12 × 12
  ├─ Conv2d(10 → 20, 5×5) → Dropout2d → MaxPool(2) → ReLU   → 20 × 4 × 4
  ├─ Flatten                                          → 320
  ├─ Linear(320 → 50) → ReLU → Dropout
  └─ Linear(50 → 10)                                  → ett tall per siffer 0–9
```

Treningsoppsett: Adam-optimalisering (læringsrate 0,001), cross-entropy-tap og batcher på 100 bilder.

## Slik fungerer nettsiden

1. Du tegner med hvit strek på svart bakgrunn, slik MNIST-bildene ser ut.
2. Tegningen klippes ut, skaleres til omtrent 20×20 piksler og sentreres i et 28×28-bilde, som i MNIST.
3. Bildet sendes til modellen (`docs/mnist_cnn.onnx`). Normalisering og softmax er bygget inn i ONNX-modellen, så den gir sannsynligheter tilbake direkte.
4. Siffer, sikkerhet og et søylediagram for alle ti sifrene vises på siden.

## Innhold

| Fil / mappe | Beskrivelse |
|---|---|
| `01_mnist_cnn.ipynb` | Jupyter-notatbok: laster data, definerer og trener modellen, og eksporterer til ONNX |
| `mnist_cnn.pt` | Trente vekter (PyTorch `state_dict`) |
| `docs/` | Nettsiden som publiseres med GitHub Pages (`index.html` og `mnist_cnn.onnx`) |

## Kjøre selv

Krever Python 3.10 eller nyere.

```bash
git clone https://github.com/RuneSor/cnn-digit-recognition.git
cd cnn-digit-recognition

python3 -m venv .venv
source .venv/bin/activate
pip install torch torchvision matplotlib jupyterlab onnx onnxscript onnxruntime
```

**Trene modellen:** Åpne `01_mnist_cnn.ipynb` i Jupyter eller VS Code og kjør cellene ovenfra og ned. MNIST lastes ned automatisk til `data/`. Eksportcellen lagrer en ny `docs/mnist_cnn.onnx`.

**Teste nettsiden lokalt:**

```bash
python -m http.server -d docs
```

Åpne http://localhost:8000. Siden må hentes via en server. Åpner du `index.html` direkte som fil, får ikke nettleseren lastet modellen.

## Teknologi

- [PyTorch](https://pytorch.org/) og [torchvision](https://pytorch.org/vision/): trening og MNIST-data
- [ONNX](https://onnx.ai/): modellformat for eksport
- [ONNX Runtime Web](https://onnxruntime.ai/): kjører modellen i nettleseren
- GitHub Pages: publisering
