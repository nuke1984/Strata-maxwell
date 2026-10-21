# Il tier M10 persistente (STRATA_MW_PERSIST): costruito, corretto, PARITÀ

**Data:** 20 ottobre 2026. **Binario:** `4aa79887` (opt-in `STRATA_MW_PERSIST=1`; il percorso
default è inerte senza l'env e il gate parity del kernel rifattorizzato passa 0 failure).

## Il disegno

Un kernel residente per M10 serve l'intero decode (avvio al primo `begin()`, halt su prefill/
close: Maxwell non ha preemption). Per servizio, l'host fa **due chiamate CUDA** — un H2D packed
`[hdr 128 B][atti][job table]` e una `cuStreamWriteValue32` che scrive il doorbell **di device**
dopo la copia — e il kernel: polla il doorbell in L2 (blocco 0, broadcast su memoria device),
grid-barrier, gira gli STESSI corpi gup e down dei kernel batched (estratti in funzioni device,
bit-identici per costruzione, gate `maxwell_iq_parity` 0 failure), barrier, scrive le righe
mapped, alza la flag E1 esattamente come la coda del down batched. Il halt è una **store host su
parola mapped** (uno stream-write non può raggiungere un kernel che possiede il suo stream:
accodarsibbe dietro di lui — deadlock trovato e vissuto). Griglia = occupancy propria del kernel
(20 blocchi su 5 SM; bug vissuto: la query degli SM sul device 0 = la 3060 dava 112 blocchi e
il barrier deadlockava).

Le chiamate per servizio cadono mentre **tutte** le M10 sono ferme al doorbell (il grafo non
spedisce il layer n+1 prima che le flag di n siano salite): la finestra dei down in cui viveva
il "veleno" delle chiamate host non esiste più, per costruzione.

## Le tre iterazioni (misurate, stesso protocollo 3×3+TTFT)

| Versione | Dispatch | tok/s (media) | Note |
|---|---|---:|---|
| legacy (baseline del mattino) | memcpy+2 lanci/servizio | **43,06** | TTFT 22,0 s |
| v0: staging pulled dal kernel (mapped) + doorbell mapped | 0 chiamate | 17,4 | corretto; le letture mapped nel kernel costano ~2,2 ms/layer (la lezione P7 in nuova forma) |
| v2: staging via DMA + doorbell di device | 2 chiamate | **42,08** | TTFT 39,1 s; pool 81 µs/layer |

## La decomposizione (braccio profilato, confronto braccio-contro-braccio)

| Fase | legacy (audit del mattino) | persist v2 |
|---|---:|---:|
| pool (dispatch host) | 688 µs/layer | **81 µs/layer** |
| — actq | 122 µs/layer | ~51 µs/layer |
| ring (grafo 3060 + attesa tier) | 415 µs/layer | 988 µs/layer |
| finestra (ms, mediana) | 54,3 @ 2,06 tok/fin | **49,3 @ 2,22 tok/fin** |
| ms/token | 26,3 | **22,2 (−15%)** |

Il lato host è un trionfo: il tax di dispatch (~600 µs/layer) è **morto** e il veleno per
ordini di visita è strutturalmente impossibile. Ma il tempo liberato è riassorbito dal lato
kernel: il servizio residente (barrier ×3, risveglio doorbell, down a occupanza del gup)
serializza ciò che i lanci pipeline-rizzavano. Il tok/s end-to-end a parità di strumentazione
è pari (41,7 vs 41,5); pulito 42,08 vs 43,06 (**−2,3%**, bordo rumore). È la lezione E1m2
generalizzata: il pavimento della catena è il compute M10 + un giro di risveglio/flag,
in qualunque confezione si imballi.

## Regressione TTFT (+17 s)

Il prefill (path legacy per chunk) paga halt+join per chunk e il primo decode rilancia il
kernel; 1546 token → 39,1 s contro 22,0. Da indagare prima di ogni promozione (sospetto: il
costo per chunk dell'halt/join in 25 chunk × 4 card).

## Stato

- Default OFF (`STRATA_MW_PERSIST`), percorso default inerte e intatto; server di produzione
  ripristinato sulla config embed-vram (binario nuovo, comportamento default identico).
- **Non promosso**: parità decode, TTFT peggiorato. Levie residue censite: barrier 1 di 3
  rimovibile (~25 µs/layer), occupanza down (2,9 KB contro 11,2: dynamic shared o kernel
  separato per fase), TTFT path.
- **Rischio noto da review**: dentro un kernel solo, i Riutilizzi del buffer di parità oltre
  due servizi non passano da invalidazione L1/RO-cache tra servizi (i caricamenti sono
  empiricamente corretti su tutti i bench; serve analisi formale prima di ogni promozione).

## File

- `bench-persist.json` — 3×3+TTFT puliti della v2 (42,08)
- `bench-persist3.json` — braccio profilato v2
- `prof-lines-v2.txt` — righe MWPROF/decode-timing della v2
