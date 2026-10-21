# Diagnosi: perché la TP routed è più lenta degli esperti interi

**Data:** 20 ottobre 2026. **Binario:** `b6be3ea6…eac1613` per tutte le braccia.
**Metodo:** due run serving strumentate (`STRATA_DECODE_TIMING=1`,
`STRATA_MW_LAYERPROF=1`), stesso carico (3 prompt standard + TTFT su prompt
unico da 1.509 token), stesso giorno. I valori profilati non vanno confrontati
con le medie non profilate delle tabelle precedenti: confronto solo braccio
contro braccio.

## A1 — Accettanza speculativa: NON è la causa

 Dai log delle tre configurazioni già misurate (19-20 richieste ciascuna):

| Configurazione | depth 1 | depth 2 | depth 3 |
|---|---:|---:|---:|
| whole-expert | 85% | 81% | 82% |
| TP legacy | 83% | 85% | 82% |
| Meta direct | 85% | 82% | 88% |

Accettanza identica (83-85%): il divario di tok/s è tempo di esecuzione reale.

## A2/B — Budget per finestra e per layer (decode)

| Fase (ms/finestra) | whole | Meta direct | Δ |
|---|---:|---:|---:|
| **ms/finestra** | **57,66** | **69,70** | **+12,0 (+21%)** |
| verify | 54,13 | 66,20 | +12,1 |
| · GPU-reach wait | 20,39 | 53,32 | +32,9 |
| · host (pool: plan+actq+jobs+run) | 31,81 | 10,47 | −21,3 |
| · — di cui actq | 5,87 | 8,66 | +2,8 |
| · — di cui run (sync M10) | 24,31 | 0,01 | −24,3 |
| draft | 3,50 | 3,47 | ≈0 |

| vlayer (µs/layer) | whole | Meta direct |
|---|---:|---:|
| ring (attesa grafo 3060) | 418,7 | **1.089,0** |
| pool (dispatch host) | 682,5 | 205,9 |
| post | 39,5 | 46,0 |
| **somma** | **1.140,7** | **1.340,9** (+200) |

La somma per layer (+200 µs × 48 ≈ +9,6 ms/finestra) chiude il conto dei
+12 ms/finestra osservati.

## B1/B2 — Le M10 non sono più lente in TP

Per card, mediana dei report decode:

| Per card | whole (jobs/L ~7) | Meta (jobs/L ~24,4) |
|---|---:|---:|
| distinct/L | ~5 | 17,2 |
| MB/L letti | ~6,7 | ~5,9 |
| **kernel totale µs/L** | **473–655** | **472–489** |
| — gup | 355–375 | 337–346 |
| — down | 98–287 | 135–137 |
| sync-wait | 42–407 | 0 |

In TP ogni card fa ~3,5× più job, ma ogni job è uno shard: **i byte letti e il
tempo di kernel per card sono gli stessi**. Il floor VMAD (~50-59 µs/MB) è
rispettato in entrambe le modalità. La TP non perde sui kernel.

## Dove si perdono i ~200 µs/layer: la struttura di consegna

1. **D2H dei 4 blocchi di parziali**: 4 × 24,7 job × 10 KB ≈ **1,01 MB/layer**
   sulla uplink condivisa da ~3,3 GB/s ≈ **~300 µs/layer** di nuova fase seriale
   (down → D2H → flag, in stream per card). Il percorso whole scrive le righe
   mapped **dentro** il kernel down (nessuna fase aggiuntiva; ~247 KB totali
   già contati nel kernel).
2. **actq più pesante**: 183,7 vs 123,4 µs/L (+60): staging di atti+tabelle per
   24,7 job × 4 card contro ~7 job × 4.
3. **Riduzione + protocollo flag sulla 3060**: la somma legge 4 aree mapped
   (~1 MB via PCIe della 3060) e la catena [flag → reduce] è sul cammino critico
   del layer.
4. **Perdita di sovrapposizione**: nel percorso whole l'attesa delle M10 vive
   nel sync host (run = 24,3 ms/finestra) che si sovrappone al lavoro della
   3060; in Meta l'attesa si sposta nel grafo (ring 419→1.089) dove la catena
   M10 [actq 184 + kernel 480 + D2H ~300 + flag] ≈ ~970 µs/layer è meno
   sovrapposta alle ~419 µs di lavoro indipendente della 3060.

Nota di equità: il whole-expert qui gira senza GPUFLAG; storicamente E1m2
sul whole era −4%. Il confronto Meta-vs-whole attuale è quello di produzione.

## D — Latenza (TTFT, prompt unico da 1.509 token)

| | whole | Meta direct |
|---|---:|---:|
| TTFT | **22,7–29,6 s** | **45,7–49,8 s** (+60%) |

Prefill per card (µs/L): in Meta i down scalano per ordine di servizio
(**1,44 / 3,94 / 6,47 / 8,94 ms/L** per rank) e ogni rank sincronizza e
accumula su host in ordine: la serializzazione per ordine di visita del
vecchio percorso è tornata nel prefill TP (accumulate rank-ordered), più i 4×
D2H. Nel whole il pattern esiste (1,4/3,7/5,2/6,4) ma su 1/4 dei byte di
consegna per job totale.

## Conclusione

**La TP routed non perde nel calcolo delle M10** (kernel e byte per card
uguali; accettanza identica). Perde perché la ricomposizione a 4 vie sposta
~1 MB/layer sulla uplink condivisa come nuova fase seriale (down→D2H→flag→reduce),
rende più pesante lo staging per layer e riduce la sovrapposizione fra la
catena M10 e il lavoro della 3060; nel prefill l'accumulo rank-ordered
serializza le card e raddoppia il TTFT.

Le leve, in ordine di atteso impatto (da misurare prima di promuoverle):
pre-riduzione a coppie di rank (2 D2H invece di 4, ~−150 µs/L), parziali fp16
sul filo (dimezza i byte D2H), fusione reduce+consegna già fatta (direct),
riordino/parallelizzazione dell'accumulate di prefill.

## Limiti

Due run profilate per braccio, non ABBA; mediane su ~20 report per card; TTFT
su 2-3 prompt unici; stima D2H da byte misurati e banda nota, non da timer
diretto sul copy engine; confronto col whole senza GPUFLAG (nota di equità
sopra). Nessuna di queste limitazioni cambia la direzione del risultato.
