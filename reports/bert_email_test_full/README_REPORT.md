# Valutazione batch DistilBERT su `email_test`

Data analisi: 20 luglio 2026.

## Risultato

- File `.eml` trovati: **8.614**
- File validi (parsing e inferenza completati): **8.614**
- Errori di parsing: **0**
- Errori di inferenza/post-processing: **0**
- Classificati `phishing`: **8.594 / 8.614 (99,7678%)**
- Non classificati `phishing` (falsi negativi sul corpus atteso tutto phishing): **20 / 8.614 (0,2322%)**
  - `legitimate`: **14**
  - `uncertain`: **6**

Le percentuali di rilevamento e falso negativo usano come denominatore tutte le 8.614 email: parsing e inferenza sono riusciti sull'intero corpus.

## Modello e decisione

L'applicazione carica da Hugging Face `eugenioderodev/fishstop-bert`. La revisione risolta durante il test è:

`67d2a3b4de475ecfb127f0bb076df210306edeaf`

Il repository Hugging Face non pubblica attualmente `calibration.json`, quindi il runtime applica i fallback presenti in `src/bert_calibration.py`:

- temperature scaling: **T = 1,0** (softmax grezzo)
- etichetta positiva: **ID 1**, semanticamente `malicious (phishing/spam)`
- soglia centrale: **0,50**
- banda totale di incertezza: **0,30**
- `legitimate`: probabilità malicious **<= 0,35**
- `uncertain`: probabilità malicious **tra 0,35 e 0,65**
- `phishing`: probabilità malicious **>= 0,65**

La classe positiva del modello è quindi più ampia del solo phishing: include phishing, scam e spam.

## Preprocessing e inferenza replicati

Il batch usa lo stesso percorso dell'app:

1. `EmlSOCAnalyzer.analyze()` estrae subject e corpo, pulisce l'HTML, seleziona il contenuto utile ed elimina quote/code conversazionali quando riconosciute.
2. `prepare_bert_input()` concatena subject e `body_for_ai`, normalizza Unicode NFKC, rimuove controlli, converte in minuscolo e compatta gli spazi.
3. Le email lunghe sono divise in un massimo di 8 finestre sovrapposte da 512 token, stride 128.
4. Viene scelta la finestra col margine malicious più alto.
5. Softmax/temperatura e classificazione a tre esiti sono applicate con la logica runtime.

L'inferenza raggruppata è stata confrontata con `predict_email_logits()` dell'app su email a uno e più blocchi: probabilità identiche, differenza assoluta `0,0`.

## Distribuzione probabilità malicious

- Media: **0,9979383150**
- Deviazione standard: **0,0433494277**
- Minimo: **0,0000014452**
- P1: **0,9999948740**
- Mediana: **0,9999996424**
- P99: **0,9999997616**
- Massimo: **0,9999997616**

Istogramma:

- 0–10%: **14**
- 30–40%: **6**
- 90–100%: **8.594**

La distribuzione è fortemente polarizzata: tutte le email non classificate phishing ricadono o sotto il 10% oppure attorno al 38,3855%; non ci sono risultati tra 40% e 90%.

## Correzione degli errori di parsing

I cinque errori iniziali provenivano da URL HTML malformati come `https://[an_21]@bit.ly/...`. `urllib.parse` interpretava il placeholder tra parentesi quadre come un host IPv6 e sollevava `ValueError`, interrompendo l'intera analisi.

L'estrattore URL ora:

- recupera in sicurezza la destinazione reale dopo il placeholder (`https://bit.ly/...`);
- ignora URL con parentesi invalide quando la destinazione non è recuperabile;
- non permette a un singolo URL malformato di interrompere il parsing dell'email.

I cinque file sono stati ritentati dopo la correzione:

- `sample-2509.eml`: `phishing`, probabilità malicious **99,999881%**
- `sample-2534.eml`: `phishing`, probabilità malicious **99,999821%**
- `sample-2535.eml`: `phishing`, probabilità malicious **99,999940%**
- `sample-2538.eml`: `phishing`, probabilità malicious **99,999678%**
- `sample-2559.eml`: `phishing`, probabilità malicious **99,999952%**

Il modello DistilBERT non è stato modificato.

## File prodotti

- `email_results.csv`: una riga per ciascuno degli 8.614 `.eml`, inclusi stato, errore, metadati, numero di chunk, logit, probabilità, classificazione e flag falso negativo.
- `summary.json`: riepilogo strutturato e metadati di riproducibilità.
- `summary.txt`: riepilogo testuale compatto.
- `scripts/batch_bert_email_eval.py`: runner riproducibile con checkpoint, gestione errori e `--resume`.

Comando usato:

```powershell
.\.venv\Scripts\python.exe scripts\batch_bert_email_eval.py "C:\Users\DEROSAEU\Downloads\email_test" --output-dir reports\bert_email_test_full --batch-size 32 --flush-chunks 128 --checkpoint-every 100
```
