# Split-window + tier persistente: +22% end-to-end, il verdetto "7% più lento" era stantio

**Data:** 21 ottobre 2026. **Binario:** `d4f80481` (persist in-order; il percorso default è inerte
senza env/flag). Protocollo 3×3+TTFT identico a tutte le campagne; baseline del 20-10: **43,06 tok/s**,
TTFT 22,0 s.

## La tabella

| Braccio | tok/s (media) | TTFT | vs baseline |
|---|---:|---:|---:|
| baseline (produzione) | 43,06 | 22,0 s | — |
| persist da solo (`STRATA_MW_PERSIST=1`+`GPUFLAG=2`) | 42,08 | 39,1 s | −2,3% |
| **`--spec-split` da solo** | **52,44** | 22,5 s | **+21,8%** |
| **composto: split + persist** | **52,52** | 22,6 s | **+22,0%** |
| composto profilato (LAYERPROF+DECODE_TIMING) | 52,04 | — | +20,9% |

## Il fatto nuovo: spec-split VINCE

Il commento in `generate.cpp` ("the overlap study: exact, ~7% slower") è **stantio**: su questo
binario la split-window P6 misura **+21,8% da sola**. Dallo studio a oggi sono arrivati i kernel
gup delle campagne vittoria-1/2 (−36% kernel: i kernel dimezzati dallo split ora costano poco) e
i dispatch P6.x packed. La pipeline dei due gruppi (il mixer di B non dipende dagli esperti di A)
riempie l'attesa M10 con lavoro 3060 legale: **ring 415 → 407 µs/layer ma con pool 688 → 54**
(composto) — la finestra scende da 54,3 a **41,1 ms @ 2,20 tok/fin = 18,7 ms/token (−29%)**.

## Il persist: bug trovato e corretto (protocollo in-order), poi parità

Primo tentativo del composto: **hang** alla prima finestra G=2 (issue #29, multi-wait in attesa
della flag tier). Bisect: split+GPUFLAG=2 senza persist gira a 51,3 tok/s → il bug era del
persist. Causa: con la finestra splittata l'host può legittimamente correre **due servizi avanti**
(il ring di B(l) scatta prima del multi-wait di A(l)); il kernel serviva "il doorbell più recente"
e **saltava il servizio in mezzo** — righe mai calcolate, flag mai alzata, hang. Il persist da
solo non corre mai avanti, per questo funzionava.

Correzione: doorbell **relativo per lancio** (1, 2, …), il kernel serve **strettamente seen+1**,
e anche i servizi **vuoti** passano dal doorbell (hdr con njobs 0: ogni servizio contato è un
servizio alzato dal kernel; il raise host degli empty è disinnescato sotto persist — due
scrittori avrebbero potuto regredire la parola monotona). Dopo la correzione: composto 52,52,
TTFT 22,6 s (la regressione 39 s del persist-solo era della stessa famiglia ed è scomparsa),
risposte corrette (2+2=4, Paris, 17 primo ✓).

## Lettura

- La catena per-layer NON è un muro assoluto: la pipeline intra-finestra (P6) la spezza e vale
  −29% ms/token; il verdetto contrario era un artefatto dei costi di dispatch dell'epoca.
- Il persist contribuisce ora ~zero sul composto (52,52 vs 52,44): il dispatch cheap è già
  pagato dallo split; resta utile come base (dispatch 2 chiamate/servizio, veleno impossibile)
  e l'in-order lo rende componibile con tutto.
- Verifica di correttezza: risposte deterministiche corrette; lunghezze di generazione variabili
  run-to-run (in-famiglia con la base k8v4/cache, documentato in 2026-10-19-embedding-vram).

## Raccomandazione

**`--spec-split` da solo** è la configurazione promuovibile: +21,8%, TTFT invariato, un flag,zero
nuovi meccanismi. Il composto è equivalente (il persist è opt-in e ora componibile). Non è stato
modificato alcun default: la configurazione di produzione resta come trovata; aggiungere il flag
è una scelta dell'utente.

## File

- `bench-split.json` — spec-split da solo (52,44)
- `bench-splitpersist.json` — il composto (52,52)
- `bench-splitpersistprof.json` — composto profilato (52,04; fasi: ring 407, pool 54 µs/L)
- `bench-persist.json` — persist da solo (42,08, protocollo pre-in-order)
