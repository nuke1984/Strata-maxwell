# Routed-expert TP: consolidamento e test, blocco 1–3

## Perimetro

Solo i routed expert assegnati alle quattro M10. Shared expert, attenzione,
router, stati/KV e MTP rimangono sulla 3060. Nessuna conversione FP16 dei pesi.
Spec **4**, contesto **256000**, KV **k8v4**, embedding in VRAM invariati.

Questo blocco rende verificabile e più sicura l'implementazione TP esistente;
**non crea ancora un backend Meta autonomo né ottimizza le prestazioni**.

## Baseline fissata prima delle modifiche

Server verificato con 391, 9999 e otto richieste concorrenti. Binario iniziale:
`cd0ce43dceec9b44911355628c3bb7ddebd30762e9bb8c18d184a0b84766f433`.
Snapshot privato di sorgenti, binario, profilo e configurazione senza credenziali:
`/tmp/opencode/tp-routed-before/`. Backup persistente del binario:
`engine/strata.bak-before-tp-audit-20261019`.

Cache 3060 **1874**; complemento M10 **22702** (5676/5676/5675/5675 nel percorso
a esperti interi). Nessun cambiamento alla selezione del profilo.

Baseline 3×3, temp 0, limite 512 token, tempi decode del motore:

| Prompt | Tok/s individuali | Media |
|---|---|---:|
| story | 38,6 / 39,6 / 39,5 | 39,23 |
| fib | 45,1 / 47,3 / 45,0 | 45,80 |
| code | 45,8 / 45,8 / 45,1 | 45,57 |

Media **43,53 tok/s**. Una risposta code ha esaurito il budget nel ragionamento;
questa è una misura di decode, non una valutazione della qualità della risposta.

## Correzioni

- Risultati dei fill paralleli in byte indipendenti, non `vector<bool>`; thread
  RAII uniti prima della distruzione dei risultati e con eccezioni catturate.
- Snapshot del blob sorgente di proprietà di ciascun thread TP, tramite
  `copy_blob` e accesso serializzato all'API della sorgente: la conversione non
  dipende più dalla durata di un puntatore a staging riciclabile. Il caricamento
  a esperti interi conserva il percorso precedente.
- Un'unica funzione per gli indici del ring: produttore e consumatori usano
  `step & 63`, anche oltre il passo 63.
- Verifica delle geometrie IQ prima del lancio e, nel server, prima del load.
- Guardie anticipate per le modalità senza wiring TP completo. In particolare,
  `--spec-split` è rifiutato: correggere gli indici non basta, perché due gruppi
  in volo non possono condividere senza protocollo un solo buffer parziale/rank.
- Controllo esplicito dei quattro rank e degli ID device distinti, separati dalla
  GPU principale; controllo sm_50 prima della lettura dei pesi.
- Il gate numerico rifiuta anche risultati non finiti.
- Commenti corretti: la rotazione bilancia i tagli, ma non garantisce identici
  byte per rank con formati e distribuzioni per layer differenti.

## Test riproducibili

```bash
cmake --build build --target strata maxwell_tp_plan_test maxwell_tp_chain_test -j4
build/maxwell_tp_plan_test
python3 tests/core/maxwell_tp_cli_test.py build/strata
STRATA_MW_GPUFLAG=2 build/maxwell_tp_chain_test /path/model-00001-of-00002.gguf /path/native-pack
cmake --build build --target maxwell_tp_publish_test -j4
build/maxwell_tp_publish_test
```

Il test GPU si esegue con serving fermo: **non è un benchmark prestazionale**.

### Planner e controlli anticipati

`maxwell_tp_plan_test`: copertura dei 640 canali senza buchi/duplicati su tutti
i 48 layer, somma byte degli shard, rotazione, geometrie valide/invalide, 10000
indici del ring e parsing device. Passato.

Il test CLI verifica **23 casi**, tutti passati, di rifiuto prima del caricamento dei pesi delle combinazioni
non supportate: generazione diretta, batching, pipeline, layer split, spec split,
assenza del token graph, flag GPU invalidi, v1 down, geometrie incompatibili,
device duplicati/malformati e bracci diagnostici incompatibili.

### Catena su quattro M10

`maxwell_tp_chain_test` usa **un routed expert reale per ciascuno dei 48 layer**,
caricato dal loader di produzione su tutti i quattro rank. Non pretende di
verificare numericamente ciascuno dei 24576 esperti del modello.

- Intermedi SwiGLU, codici quantizzati, scale e correzioni degli shard confrontati
  bit per bit con il kernel Maxwell senza sharding.
- Routing di dieci ID distinti/token; un esperto posseduto dal fixture, ripetuto
  fra token e collocato a rank di routing variabile, con righe non possedute.
- Finestre decode da 1, 2, 3, 4 token su tutti i layer: **304 servizi**, di cui
  **12 vuoti**, inclusi 100 replay dello stesso ingresso con controllo bitwise.
- Ring locale con 96 passi, riutilizzato oltre il confine 64, e contatore dei rank
  monotono; è uno stress sequenziale, **non supporto a due gruppi in volo**.
- Trasporto e flag di produzione, riduzione GPU confrontata bit per bit con la
  somma CPU nello stesso ordine di rank; righe non possedute mai sovrascritte.
- Prefill su layer 0/1/8, T=1/4/63/64, scatter con offset e accumulo dei parziali:
  **12 casi**. Il loop dei sotto-batch oltre 64 token richiede il test serving.
- Uscita finale confrontata con l'esperto intero, soglia relL1 **1e-5**, con
  controllo dei valori finiti. Non si richiede uguaglianza bitwise con il down
  intero: la partizione cambia l'associazione delle somme FP32.

| Configurazione del test | Esito | Peggior relL1 TP vs intero |
|---|---|---:|
| GPUFLAG=2, geometria default | PASS | 1,741e-7 |
| GPUFLAG=1, geometria default | PASS | 1,741e-7 |
| GPUFLAG=2, IQ_SPLIT=2 | PASS | 1,680e-7 |
| GPUFLAG=2, IQ_T256=1 (split 4) | PASS | 1,741e-7 |

Compute Sanitizer memcheck, GPUFLAG=2, intera suite: **ERROR SUMMARY: 0 errors**.
Questi tempi sotto strumento non sono usati per valutare le prestazioni.

### Pubblicazione asincrona delle sequenze

`maxwell_tp_publish_test` avvia il consumer mentre il main flag è ancora basso,
poi pubblica expected e main, mantenendo intenzionalmente vecchie le quattro
flag dei rank. Il consumer non deve completare finché non vengono pubblicate le
nuove flag. Tre consumer reali: wait_mw, copy_rows_from_mapped_after_mw,
sum_tp_rows_after_mw. **1000 iterazioni ciascuno: zero completamenti anticipati**.
È uno stress osservato, non una prova formale di tutti gli interleaving possibili.

Nota sul banco: il primo tentativo aveva upload pageable sul default stream e
kernel sullo stream non-blocking, senza ordinamento esplicito. Il confronto degli
intermedi lo ha rilevato. Gli upload del riferimento sono stati ordinati sullo
stesso stream prima delle misure finali: non era una correzione dell'aritmetica TP.

## Verifica del caricamento e del serving TP completo

Configurazione locale `strata-iq2xs-tp-routed-test.json` (gitignored: contiene
credenziali), derivata dalla configurazione embedding VRAM con le sole aggiunte
`--maxwell-split tensor` e `STRATA_MW_GPUFLAG=2`.

- Caricamento completo: **22702 shard/rank**, quattro round-trip gate PASSED.
- GiB/card riportati: **7,61 / 7,63 / 7,63 / 7,61**; zero sovrapposizioni con la
  cache 3060 da 1874 esperti, invariata.
- Fill parallelo: circa **493,9 s**; server pronto dopo **535,3 s**. Non sono
  confronti controllati delle prestazioni di startup.
- Log: quattro GPU-raised doorbell mode 2 e TP armed, attivazione effettiva verificata.
- `/health` loaded/ok a **256000**.
- **391, 9999, otto richieste concorrenti 391: tutti corretti**.
- Prompt aggiuntivo **1509 token senza prefisso riusato → 391**, 71 token generati,
  36,608 s HTTP. Esercita il prefill oltre il limite del singolo sotto-batch M10.
- Non è stato ingerito un prompt completo da 256k token.

### Punto prestazionale della TP corretta

Stesso protocollo 3×3 della baseline, senza profiler o sanitizer:

| Prompt | Tok/s individuali TP | Media |
|---|---|---:|
| story | 31,1 / 32,0 / 31,4 | 31,50 |
| fib | 36,5 / 35,5 / 36,1 | 36,03 |
| code | 37,3 / 35,7 / 35,9 | 36,30 |

Media TP attuale **34,61 tok/s**. Le tre risposte code hanno raggiunto il limite
nel ragionamento. Le sequenze generate non sono identiche alla baseline: questa
è una misura della configurazione serving, non un confronto a routing fissato.

Il binario della TP è
`c35277a9da8241d9341d0f8a80c5a4333fa0723275a65d897963ab9aedff2f4c`.
Il percorso a esperti interi è stato poi ripristinato e rimisurato con
**lo stesso binario** (vedi chiusura del blocco). TP richiede GPUFLAG=2 mentre il serving normale
non lo arma: il confronto include il protocollo E1 richiesto dalla TP, non isola
il solo sharding. Non è una campagna ABBA con intervalli di confidenza.

Memoria dopo lo screening TP: 3060 **11407 MiB usati / 503 liberi**; M10
**7913/7932/7935/7916 MiB usati**, liberi **208/190/187/206 MiB**.

Questi numeri sono il punto di partenza dell'implementazione corrente, **non un
verdetto sulle prestazioni del futuro tier Meta o su tutte le TP MoE**.

## Osservazioni per i blocchi successivi, non ottimizzazioni già realizzate

- La rotazione per layer distribuisce i tagli nel modello, ma a ciascun layer
  due rank calcolano 192 canali e due 128: non bilancia il singolo passo.
- Nel TP tutti i rank possiedono gli stessi incarichi: il vecchio ordinamento
  per numero di incarichi li considera uguali, pur con shard di ampiezza diversa.
- Down conserva scratch/staging intermedi full-width e produce un parziale
  per incarico; il costo va misurato separatamente da quello del dot.
- Il risultato TP viene prima scritto nelle righe host, poi ricopiato sulla
  3060. La riduzione e la consegna non sono ancora un'unica operazione interna
  al tier Meta.
- Il prossimo confronto controllato dovrà fissare anche routing/numero di
  token nei microtest, riportare l'accettazione speculativa e distinguere la
  configurazione E1 dall'effetto del solo partizionamento.

Nessun guadagno viene dedotto da queste sole osservazioni statiche.

## Ritorno al serving normale e chiusura del blocco

Server ripristinato con `strata-iq2xs-embed-vram.json`, pronto dopo **456,9 s**:
embedding VRAM attivo, **nessuna TP e nessun GPUFLAG** nel processo. Stesso binario
usato per B TP (SHA256 `c35277a9…dff2f4c`). Distribuzione confermata:
1874 esperti sulla 3060, 5676/5676/5675/5675 interi sulle M10, zero overlap.

Quattro gate M10 PASSED, health loaded/ok a 256000, **391 / 9999 / 391×8 ✓**.
Ripetuto anche il prompt da **1509 token senza riuso → 391**, 69 token generati,
23,079 s HTTP. Memoria finale libera: 3060 **509 MiB**, M10 **190/192/203/205 MiB**.

| Configurazione | Story | Fib | Code | Media tok/s |
|---|---:|---:|---:|---:|
| Normale prima, binario precedente | 39,23 | 45,80 | 45,57 | **43,53** |
| TP corretta, GPUFLAG=2 | 31,50 | 36,03 | 36,30 | **34,61** |
| Normale dopo, stesso binario della TP | 38,93 | 44,47 | 45,97 | **43,12** |

A dopo, osservazioni individuali: story **37,8/39,5/39,5**, fib **43,9/44,8/44,7**,
code **46,6/45,5/45,8**. Il primo code ha esaurito il budget nel ragionamento.
Il normale dopo differisce dal prima di circa **−0,94%**, entro la dispersione
osservata; non è una prova statistica di equivalenza. La TP corrente è **−19,74%**
rispetto al normale sullo stesso binario in questo screening, con i limiti di
protocollo/routing già dichiarati. Non viene promossa in produzione.

**Completati i blocchi 1–3:** stato riproducibile, test della catena esistente,
correzioni e supporto circoscritto. Non ancora realizzati: tier Meta autonomo,
confronti ABBA con E1 fissato, profilo normalizzato a routing/token identici,
fusioni e ottimizzazioni del percorso. Serviranno i blocchi successivi del piano.
Non sono stati modificati gli shared expert o la selezione degli esperti del tier.

## Supporto dichiarato di questa versione

Il ramo TP è limitato al serving seriale con un verifier, quattro M10 sm_50 e
GPUFLAG 1 o 2. Non sono supportati batching, layer split, pipeline windows,
spec split, device-plan e i bracci zero-copy/local-down/rows-f16. Il rifiuto è
esplicito e anticipato; i percorsi normali restano disponibili senza TP.

Build e verifiche su CUDA; HIP/SYCL non testati su questa macchina.
Nessun commit/push. Il lavoro precedente nel tree è conservato.
