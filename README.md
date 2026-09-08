# FishSTOP

La repository continua a contenere il progetto Streamlit e il motore di analisi
email originale. La pagina pubblica eseguita da Streamlit (`streamlit_app.py`)
è ora una landing page per scaricare l'app desktop FishSTOP, sviluppata con
Tauri.

## Download dell'app desktop

La landing legge automaticamente l'ultima release pubblica della repository
[fishstop-desktop-email-security](https://github.com/EugenioDeRosa/fishstop-desktop-email-security).
Quando una release contiene un file `.dmg` per macOS o `.msi`/`.exe` per
Windows, il pulsante di download viene aggiornato senza modificare Streamlit.

## Pagine pubbliche OAuth

La landing include le pagine pubbliche utilizzate dal consenso OAuth:

- [Privacy Policy](https://fishstop-eml.streamlit.app/?page=privacy)
- [Terms of Service](https://fishstop-eml.streamlit.app/?page=terms)

Entrambe descrivono l'accesso read-only alla casella e il confine di
elaborazione locale dell'app desktop; i relativi collegamenti sono disponibili
anche nel footer della home page.

## Avvio locale della pagina

```bash
python -m pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Codice di analisi conservato

Il codice storico dell'analizzatore resta in `src/`, con test, dataset e script
di supporto. Non viene eseguito dall'entry point pubblico, ma è conservato per
lo sviluppo e la documentazione della tesi.
