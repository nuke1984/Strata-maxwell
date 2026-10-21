# Maxwell Meta autonomo: routed expert M10

## Perimetro e architettura

Solo il complemento routed della cache 3060: la selezione degli esperti non viene
cambiata. Shared expert, attenzione, router, KV/stati e MTP restano sulla 3060;
embedding VRAM, spec 4, 256k e k8v4 restano invariati. Nessun peso espanso FP16.

`MaxwellMetaTier` (`include/strata/core/maxwell_meta.hpp`,
`src/core/maxwell_meta.cpp`) è il singolo endpoint del tier:

- Possiede privatamente quattro backend fisici, assignment, loader parallelo,
  snapshot dei blob sorgente, flag, contatori, mappe e buffer parziali.
- `owns` e `held_pairs` descrivono un unico insieme di routed expert.
- `decode` prepara e avvia l'intero lavoro TP, pubblicando internamente la sequenza.
- `record_consume` registra completamento, riduzione e consegna nel grafo della
  GPU principale. Il verifier non orchestra i rank del percorso Meta.
- `prefill` esegue e ricompone il sotto-batch, controllando i limiti dello scatter.
- Una consegna GPU pubblica un ticket di consumo. Un secondo invio che potrebbe
  sovrascrivere buffer ancora in uso è rifiutato; gli errori sono latched.
- `abort` rilascia i consumer in attesa. Il rilascio fatale del verifier usa
  questo endpoint, senza conoscere le flag dei rank.

L'oggetto va aperto prima della cattura e deve vivere più a lungo dei grafi che
lo utilizzano; open/close sono operazioni quiescenti. Non supporta pipeline,
batching, layer split o due gruppi spec-split in volo. Non è il backend Meta
generico dell'intero modello GGML: è specializzato per il blocco routed M10.

## Integrazione

Il percorso `--maxwell-split tensor` usa il nuovo Meta. Dispatch e prefill
ricevono un solo puntatore al tier. La precedente TP è conservata esclusivamente
come riferimento same-binary con **`STRATA_MW_META_LEGACY=1`**. Le strutture a
quattro rank rimangono nel percorso legacy/whole-expert, non nell'orchestrazione
del nuovo endpoint.

Il percorso predefinito a esperti interi rimane distinto e non attiva il Meta.
Il collegamento prefill del Meta è obbligatorio: un errore non produce un ripiego
silenzioso che sposta il calcolo degli esperti sulla 3060 o sulla CPU.

## Ottimizzazioni separate

- **`STRATA_MW_META_DIRECT=1`**: il kernel di riduzione consegna direttamente
  le righe complete nella VRAM della 3060. Somma FP32 nel medesimo ordine
  `((p0+p1)+p2)+p3`; zeri per gli hit della GPU principale e copia delle eventuali
  righe CPU/fallback. Elimina la scrittura dei risultati ricomposti in RAM e
  la successiva rilettura verso GPU.
- **`STRATA_MW_META_HEAVY_FIRST=1`**: ordina gli avvii per dimensione dello shard,
  senza modificare placement o ordine della somma. Ordinamento su array fisso,
  senza allocazioni per layer. È sperimentale, non selezionato come default.
- **`STRATA_MW_META_WIRE16=1`** (2026-10-20, pomeriggio): i parziali viaggiano in
  fp16 (metà byte sul filo); la riduzione li riconverte esattamente (uint2 per
  float4 d'uscita). Cambia i bit d'uscita in modo limitato: gate L1 5e-3 nel test.
- **Consegna a blocchi**: il D2H dei parziali è spezzato in 4 tratti, ciascuno in
  attesa solo dei propri down, così il filo si nasconde sotto la coda dei down
  rimanenti (le copy engine non contendono con i kernel - misurato nella campagna
  LOCAL_DOWN). Gli intermedi (faq/fasc/fahx) sono offsettati per tratto: il
  kernel li indicizza per job locale. Il primo tentativo senza offset leggeva le
  attivazioni sbagliate nei tratti successivi (il test di catena lo ha catturato
  subito, relL1 ~1,0).
- **Doppi parziali + ticket rilassato**: la sottomissione l+1 parte mentre il
  consume del layer l è ancora in volo (un massimo; il secondo viene rifiutato).
  Atti packati una volta e condivisi dai quattro rank (prima ogni rank ripeteva
  lo stesso packing per-token).

Il trasporto di riferimento del nuovo Meta usa uno staging di ritorno proprio:
non modifica il buffer fallback const del chiamante. Il ticket viene pubblicato
dopo la copia finale. Nel percorso diretto il ticket viene pubblicato dall'ultimo
blocco, dopo le scritture di tutte le righe. Non sono usate atomiche GPU su host
condivise fra rank: il contatore di arrivo risiede sulla GPU principale.

## Correttezza

Il test `maxwell_tp_chain_test --meta` estende il fixture a **otto esperti reali
per ognuno dei 48 layer** (384 esperti), con più esperti nella stessa finestra,
ID distinti per token e riuso fra token. Confronta il nuovo endpoint contro la
TP precedente, con gli stessi dati e gli stessi incarichi.

Sono coperti hit GPU0, righe CPU/fallback, finestre 1/2/3/4, 304 servizi decode
(12 vuoti, 100 replay), prefill 1/4/63/64, scatter con offset, indipendenza della
parità dal ring esterno, rifiuto di sovrapposizioni, abort con main flag basso,
bounds dello scatter, errore della sorgente e rilascio delle risorse.

I test delle quattro combinazioni direct/heavy e di GPUFLAG=1 hanno
prodotto risultati bit-identici alla TP legacy, con peggior relL1 circa
**1,482e-7** rispetto al down intero. Gli intermedi quantizzati sono bit-identici.
La suite finale controlla anche che il fallback del chiamante non venga scritto.
Compute Sanitizer memcheck sulla suite finale, sia reference sia direct:
**ERROR SUMMARY: 0 errors** in entrambi i casi.

`maxwell_tp_publish_test` include i due trasporti Meta: cinque consumer ×1000
pubblicazioni asincrone, **zero completamenti anticipati osservati**, inclusi i
ticket dei due trasporti Meta. Non è una prova formale di tutti gli interleaving.

## Benchmark a routing fissato

`maxwell_tp_chain_test --bench-meta` confronta la TP legacy con il Meta nello
stesso processo. Dataset: 48 layer, 4 token, 10 ID/token, 8 esperti M10 distinti
per layer riusati sui quattro token (**32 incarichi/rank/layer**), altri ID
trattati come hit GPU0. Attivazioni quantizzate preparate prima del timer.

Un grafo consumer di 48 passi per backend, con piccoli marker D2H al posto del
lavoro attention/router della 3060. La CPU attende in memoria, senza query/sync
CUDA nella finestra dei rank. Timer include dispatch, trasporto, calcolo,
riduzione, marker e consegna; esclude tokenizzazione, attenzione, shared expert,
quantizzazione host iniziale e MTP. Non è una misura di tok/s del server.

Burn alternato di almeno due secondi; sette blocchi **A–B–B–A**, A=legacy TP,
B=Meta configurato. Delta sul rapporto fra le medie A e B di ogni blocco; CI95
Student-t sui sette blocchi. È un intervallo condizionato al dataset e al processo,
non una replica fra avvii o una garanzia end-to-end.

La diversità del fixture (otto esperti/layer) non replica quella del routing di
produzione. Il confronto finale deve includere il server e l'accettazione MTP.
Risultati finali dopo il consolidamento dello staging privato, dati individuali
e stdout completi in [fixed-routing.json](fixed-routing.json):

| Candidato Meta | Legacy µs/layer | Meta µs/layer | Guadagno di throughput tier | CI95 sui blocchi |
|---|---:|---:|---:|---:|
| Riferimento | 1066,82 | 1050,70 | +1,53% | +1,47..+1,59% |
| Direct | 1074,09 | 1015,72 | **+5,75%** | **+5,66..+5,84%** |
| Heavy-first | 1069,01 | 1064,92 | +0,38% | +0,26..+0,51% |
| Direct + heavy-first | 1072,06 | 1025,60 | +4,53% | +4,45..+4,61% |

Selezionato per il test nel server **direct=1, heavy-first=0**. I controlli
legacy sono misurati dentro ciascun esperimento ABBA, non ricavati da un'altra
giornata. Le configurazioni sono eseguite in processi distinti; la scelta tra
candidati è uno screening, non un confronto head-to-head in un unico ABBA.

### Banco a routing fissato dopo le ottimizzazioni (2026-10-20, pomeriggio)

Stessa metodologia ABBA (7 blocchi, CI95 sui blocchi), legacy TP come braccio A:

| Candidato Meta | Legacy µs/layer | Meta µs/layer | Guadagno | CI95 |
|---|---:|---:|---:|---:|
| direct (fp32, blocchi, atti condivisi) | 1063,4 | 1008,9 | +5,41% | +4,48..+6,33 |
| direct+wire16 | 1062,2 | 899,4 | +18,09% | +17,87..+18,32 |
| heavy+wire16 | 1053,0 | 890,8 | +18,21% | +17,82..+18,59 |
| **direct+heavy+wire16** | 1061,2 | **858,5** | **+23,61%** | +23,32..+23,89 |

Nota sul bersagio 3,29x (dal test denso Ornith): il tempo di tier both16
(858 µs/L) contro una singola M10 che facesse tutti i 32 job del fixture
(kernel ×4 ≈ 1.920 µs, più la consegna) è ~2,8-3,0x end-to-end sul tier; il
compute per card resta 4x e il residuo è consegna. Il banco misura TP-vs-TP:
il confronto onesto con il whole-expert resta il serving. Il test denso
ottiene 3,29x perché il suo intero token è sulla riga; qui solo la frazione
routed passa dalle M10, e il filo è condiviso con gli atti in ingresso.

## Serving: Meta ottimizzato (direct+heavy+wire16)

Config `strata-iq2xs-meta-opt.json`, pronto dopo **467,0 s**. Correttezza:
**391 / 9999 / 391×8 ✓**; prefill 1509 → 391 (36,5 s HTTP). **TTFT** su prompt
unico da 1509 token: **47,95 s** — invariato rispetto al Meta direct: la fase
ottimizzata è la consegna di decode; il prefill usa ancora l'accumulo
rank-ordered su host (la sua parallelizzazione è lavoro futuro).

Il 3×3 decode viene riportato al completamento del benchmark in corso.

### Decode 3×3, Meta ottimizzato — e il verdetto sulla catena

| Prompt | Tok/s individuali | Media |
|---|---|---:|
| story | 31,0 / 32,1 / 32,2 | 31,77 |
| fib | 37,0 / 37,6 / 37,6 | 37,40 |
| code | 36,9 / 37,3 / 37,3 | 37,17 |

Media **35,44 tok/s** — **invariata** rispetto al Meta direct (35,51) nonostante
il tier sia +23,61% più veloce nel banco a routing fissato.

**Questo è il dato decisivo della sessione.** Il guadagno sul tier è reale e
misurato, ma il collo del serving non è (più) il tier M10: è la **catena seriale
per layer della 3060** (ring ~1.089 µs/L = lavoro attention/grafo + attese),
che il miglioramento della consegna non tocca. Nel test denso Ornith la TP dà
3,29x perché **l'intero modello** gira sulle M10: non esiste una 3060 la cui
catena seriale domini. Qui la frazione routed è ~36% della finestra e vive
sotto il tetto della 3060.

| Configurazione (stesso binario) | Media tok/s |
|---|---:|
| Esperti interi | 42,98 / 42,77 |
| TP legacy | 34,52 |
| Meta direct | 35,51 |
| Meta ottimizzato (wire16+blocchi+doppi) | 35,44 |

LaTP rimane **~−17%** rispetto al whole-expert in serving. Le prossime leve
end-to-end non sono nel tier: sono la sovrapposizione della catena M10 con il
lavoro indipendente della 3060 (riordino del grafo, la storica "Fase 4") e la
riduzione del ring stesso. Serving normale ripristinato a fine misura.

La baseline normale iniziale del 20 ottobre, prima di intervenire sul server,
è **42,98 tok/s** (3×3, story38,70/fib45,00/code45,23). Il dato microbench non
viene trasformato in un guadagno previsto dei tok/s del modello.

## Serving: riferimento sul medesimo binario

Binario `b6be3ea6973b869346aef4a97e8605c17d7bb293d6d1777344c965598eac1613`,
config privata `strata-iq2xs-meta-legacy-reference.json`: GPUFLAG=2,
META_LEGACY=1. Server pronto dopo **464,7 s**; cache 3060 **1874**, ognuno dei
quattro rank **22702** shard, zero overlap, quattro gate PASSED.
391/9999/391×8 e health a 256000 tutti corretti.

3×3 standard, temp0/512, senza profiler:

| Prompt | Tok/s individuali | Media |
|---|---|---:|
| story | 30,9 / 31,1 / 31,1 | 31,03 |
| fib | 36,7 / 36,7 / 37,1 | 36,83 |
| code | 33,6 / 36,2 / 37,3 | 35,70 |

Media **34,52 tok/s**. Le tre code terminano il budget nel ragionamento.
Osservazioni complete in [serving-legacy.json](serving-legacy.json).
Il caricamento Meta direct segue questa misura; non è ancora un risultato finale.

## Serving: Meta direct

Stesso binario e GPUFLAG=2; configurazione privata `strata-iq2xs-meta-direct.json`,
direct=1, heavy-first=0. Pronto dopo **502,7 s**, fill circa **462,7 s**.
I tempi di startup non sono un A/B controllato della velocità di caricamento.

- Log verificato: **ONE routed-expert tier** e **ONE Maxwell Meta** nel prefill.
- **22702 shard/rank**, cache 3060 **1874**, zero overlap.
- GiB/card **7,612 / 7,630 / 7,632 / 7,614**, quattro round-trip gate PASSED.
- Health a 256000, **391 / 9999 / 391×8 tutti corretti**.
- Prefill **1509 token, zero riuso, risposta 391**, 69 generati, 36,426 s HTTP.
  Non è una prova di ingestione di un intero prompt da 256k token.

### Decode 3×3, Meta direct

| Prompt | Tok/s individuali | Media |
|---|---|---:|
| story | 31,2 / 32,2 / 32,1 | 31,83 |
| fib | 37,1 / 37,6 / 37,9 | 37,53 |
| code | 36,9 / 37,4 / 37,2 | 37,17 |

Media Meta direct **35,51 tok/s** (osservazioni complete in
[serving-direct.json](serving-direct.json)). Le tre code terminano il budget
nel ragionamento. **Correzione:** in una comunicazione intermedia la media era
stata indicata come 35,96 (+4,15%): ricalcolata dalle osservazioni archiviate è
**35,51, +2,86%** rispetto alla TP legacy nello stesso screening. Fa fede questo
documento.

**Quadro complessivo dello stesso giorno, stesso binario:**

| Configurazione | Media tok/s |
|---|---:|
| Esperti interi (baseline iniziale) | 42,98 |
| TP legacy (GPUFLAG2, META_LEGACY1) | 34,52 |
| Meta direct (GPUFLAG2, DIRECT1) | **35,51** |

Il Micro Meta con consegna diretta è la configurazione TP più veloce misurata e
mantiene identico l'insieme dei routed expert. Rimane **sotto il percorso a
esperti interi** in questo screening. Le nove richieste per braccio non sono una
campagna ABBA con intervalli di confidenza: il +2,86% è la differenza delle
medie di due screening sequenziali, con routing e testi generati diversi.

Il serving normale a esperti interi è stato ripristinato a fine misura; Meta e
TP restano opt-in documentati (`--maxwell-split tensor`, con
`STRATA_MW_META_LEGACY=1` per il riferimento). Nessuna promozione a default.

## Chiusura: serving normale ripristinato e verificato

Config `strata-iq2xs-embed-vram.json` (whole-expert, embedding VRAM, nessun
GPUFLAG/Meta nel processo), stesso binario `b6be3ea6…eac1613`.
**391 / 9999 / 391×8 tutti corretti.** 3×3 finale: story **38,83**, fib **45,30**,
code **44,17**, media **42,77 tok/s** — in linea con la baseline iniziale del
giorno (42,98; −0,5%, entro la dispersione osservata). Osservazioni complete in
`/tmp/opencode/meta-normal-after-20261020.json` (archivio locale) e in
[checks.json](checks.json). Una code ha esaurito il budget nel ragionamento.

Il lavoro di questa sessione è documentato in `WORKLOG.md` (voci 2026-10-19 e
2026-10-20) e nelle cartelle `bench/results/2026-10-19-*` e
`bench/results/2026-10-20-maxwell-meta/`. Nessun commit o push eseguito.

## Riproduzione

```bash
cmake --build build --target strata maxwell_tp_chain_test maxwell_tp_publish_test -j4
STRATA_MW_GPUFLAG=2 STRATA_MW_META_DIRECT=1 build/maxwell_tp_chain_test /path/shard1.gguf /path/pack --meta
STRATA_MW_GPUFLAG=2 STRATA_MW_META_DIRECT=1 build/maxwell_tp_chain_test /path/shard1.gguf /path/pack --bench-meta
build/maxwell_tp_publish_test
```

Serving fermo durante test e benchmark. Solo CUDA verificato su questa macchina;
HIP/SYCL non costruiti. Nessun commit o push.

## Chiusura finale (binario con le ottimizzazioni Meta in tree)

Il binario finale `dd2852bc…4761d35` contiene wire16, consegna a blocchi,
doppi parziali e atti condivisi — **inerti nel percorso whole-expert** (tutto
sotto `--maxwell-split tensor`). Serving normale ripristinato e verificato:

- **391 / 9999 / 391×8 ✓**, prefill 1509 → 391 (22,8 s HTTP).
- 3×3: story **39,27**, fib **45,00**, code **45,37**, media **43,21 tok/s** —
  in linea con 42,77/42,98 delle misure precedenti (percorso non toccato).

Backup del binario pre-ottimizzazioni: `engine/strata.bak-before-meta-20261020`
(`b6be3ea6…eac1613`). Nessun commit/push. Le prossime leve end-to-end sono
fuori dal tier M10 e richiedono il riordino del grafo della 3060.
