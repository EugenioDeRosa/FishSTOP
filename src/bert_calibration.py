"""
bert_calibration.py - Inferenza calibrata per il classificatore BERT phishing.

Perche' esiste: il softmax grezzo di un transformer fine-tuned tende ad
essere overconfident (Guo et al., 2017, "On Calibration of Modern Neural
Networks"), soprattutto nelle epoche finali dove la loss di training
continua a scendere mentre quella di validazione ristagna o risale (visto
empiricamente durante il training di questo modello). Questo modulo applica:

  1. Temperature scaling: divide i logit per una temperatura T stimata sul
     validation set PRIMA del softmax. T=1.0 e' un no-op (softmax grezzo),
     usato come fallback quando non e' ancora disponibile una calibrazione.
  2. Una soglia decisionale derivata da una curva ROC/PR sul validation set
     (invece del 50% implicito), con una banda di incertezza intorno ad
     essa, salvate insieme al modello in un file calibration.json.

Il file calibration.json viene prodotto dal notebook di training
(notebooks/Phishing_detection.ipynb, sezione "Calibrazione") e va caricato
insieme al modello (vedi src/views/backend.py::get_calibration).

Import identico sia lato training (Colab, per calcolare T e la soglia) sia
lato inferenza (app Streamlit), cosi' la logica di decisione non puo' piu'
divergere tra le due fasi come era successo in passato per il preprocessing
del testo (bert-base vs distilbert, normalizzazione diversa tra notebook e
app).
"""

import torch

# Fallback usati SOLO se calibration.json non e' ancora stato pubblicato
# insieme al modello. Riproducono ESATTAMENTE il comportamento legacy:
# softmax grezzo, "phishing" sopra il 65%, "legitimate" sotto il 35%,
# "uncertain" nella banda 35-65 (soglia 0.50 +/- banda 0.15 per lato).
DEFAULT_TEMPERATURE = 1.0
DEFAULT_THRESHOLD = 0.50
DEFAULT_BAND = 0.30  # larghezza TOTALE della banda di incertezza attorno alla soglia


def calibrated_probabilities(logits: torch.Tensor, temperature: float = DEFAULT_TEMPERATURE) -> torch.Tensor:
    """
    Applica temperature scaling + softmax. Con temperature=1.0 e' identico
    al softmax grezzo usato in precedenza (nessuna regressione se
    calibration.json non e' disponibile).
    """
    t = max(float(temperature), 1e-6)
    return torch.softmax(logits / t, dim=1)


def classify(prob_phishing: float, threshold: float = DEFAULT_THRESHOLD, band: float = DEFAULT_BAND) -> str:
    """
    Decide 'phishing' / 'legitimate' / 'uncertain' a partire dalla
    probabilita' di phishing calibrata (0-1, non percentuale).

    threshold: punto centrale di decisione stimato sul validation set
        (es. il punto che massimizza F1 sulla curva PR calcolata su
        probabilita' GIA' calibrate). Con i default legacy vale 0.50.
    band: larghezza totale della banda di incertezza centrata sulla
        soglia. Con i default legacy (threshold=0.50, band=0.30) si
        riproduce esattamente la banda 35%-65% usata in precedenza.
    """
    half = band / 2
    if prob_phishing >= threshold + half:
        return "phishing"
    if prob_phishing <= threshold - half:
        return "legitimate"
    return "uncertain"
