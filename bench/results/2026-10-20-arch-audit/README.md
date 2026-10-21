# Audit dell'architettura documentata: dove va il tempo in decode

**Data:** 20 ottobre 2026. **Scopo:** rilanciare l'architettura descritta in
`ARCHITETTURA_SERVER_ATTUALE.md` (whole-expert, split statico 3060 + 4×M10,
coordinamento host per layer), verificarla contro i marcatori del documento e
misurare dove si perde il tempo quando nessuna risorsa è satura.

**Binario:** `dd2852bc63599585` (= `engine/strata` = `build/strata`), same-day.
Configurazioni: baseline `/tmp/opencode/spec-sweep.json` (embedding host
mappato, come il corpo del documento); braccio profilato = stessa config +
`STRATA_MW_LAYERPROF=1` + `STRATA_DECODE_TIMING=1`. Driver NVVIDIA 580.178.04,
VM KVM, tutte le GPU PHB sotto lo stesso host bridge.

## Verifica dell'architettura eseguita (braccio baseline)

| Marcatore | Atteso dal documento | Rilevato |
|---|---|---|
| Embedding | host mappato, 322 MiB | "IQ4_XS in mapped host memory (322 MiB)" |
| Cache 3060 | 1.874 esperti, 2,53 GiB | 1874 slot, 2,53 GiB, profilo precompilato |
| Tier M10 | 5.676/5.676/5.675/5.675 | identici (blob convertiti) |
| Round-trip gate | 4× PASSED | 4× PASSED (worst rel 1,26e-02) |
| Ambiente motore | nessun flag opt-in | solo `CUDA_VISIBLE_DEVICES=0,1,2,3,4`, `STRATA_SPEC_DEPTH=1` |

Il binario contiene il codice Meta/TP del 20 ottobre, inerte senza
`--maxwell-split tensor`: percorso whole-expert intatto.

## Protocollo

3×3 (story / fib / code, temperatura 0, limite 512, sequenziali) + TTFT su
prompt unico ~1.546 token in streaming; burn-in non contato; tok/s decode dalla
riga di log del motore. Utilizzazione GPU campionata a 1 Hz con
`nvidia-smi --query-gpu` durante una singola richiesta da 512 token.

### Baseline (non profilata)

| Prompt | tok/s (3 run) | Media |
|---|---|---:|
| story | 39,4 / 39,5 / 39,6 | 39,50 |
| fib | 45,0 / 44,9 / 44,9 | 44,93 |
| code | 45,6 / 43,0 / 45,6 | 44,73 |

Media complessiva **43,06 tok/s**. TTFT 22,04 s su prompt 1.546 token
(prefill ~70 tok/s). In famiglia con 42,68–43,21 delle campagne 19–20 ottobre.

### Utilizzazione durante una richiesta decode (16 campioni × 1 s)

| GPU | util media | util max | mem ctrl | potenza |
|---|---:|---:|---:|---:|
| RTX 3060 | 35,2% | 52% | 22,3% | 81,2 W |
| M10 #1 | 23,8% | 39% | 7,8% | 28,1 W |
| M10 #2 | 21,4% | 36% | 7,0% | 27,4 W |
| M10 #3 | 24,8% | 42% | 7,4% | 24,2 W |
| M10 #4 | 24,4% | 41% | 8,1% | 27,7 W |

Nessuna risorsa satura: conferma del sintomo riportato.

### Braccio profilato

Media **41,52 tok/s** (tassa di strumentazione −3,6% sul braccio baseline;
confronti solo braccio-contro-braccio). TTFT 23,14 s.

## Il budget per layer (mediane sui report MWPROF della corsa profilata)

| Fase | µs/layer | cos'è |
|---|---:|---|
| ring | 414,6 | attesa del grafo 3060 (attenzione/GDN/router: lavoro 3060) |
| pool — plan | 6,4 | assegnazione kind |
| pool — actq | 122,3 | staging attivazioni + 4 upload + lanci gup |
| pool — jobs | 14,4 | tabelle job |
| pool — run | 542,4 | lanci down + finish-sync M10 + attesa scatter |
| post | 14,3 | PLE gather + pubblicazione flag |
| **somma** | **~1.117** | × 48 layer = **53,6 ms/finestra** |

Riconciliazione: finestra misurata (mediana 11 richieste) **54,27 ms** — il
conto chiude al 99%. Per finestra: GPU-reach wait 19,04 + host 30,46 + draft
2,99 ms, T medio 2,68.

## Le M10, per card (mediane, 67 report per card)

| Card | jobs/L | MB/L | ideale µs/L | kernel µs/L | slack | gup | down | sync-wait |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| CUDA1 | 6,6 | 6,3 | 76 | 432 | 5,6× | 342 | 90 | 373 |
| CUDA2 | 6,5 | 5,9 | 72 | 529 | 7,0× | 338 | 191 | 87 |
| CUDA3 | 6,7 | 6,3 | 76 | 586 | 7,4× | 345 | 241 | 56 |
| CUDA4 | 6,9 | 6,4 | 77 | 636 | 7,8× | 354 | 273 | 36 |

- **gup uniforme (~340–354 µs/L) e instruction-bound**: 5,6–7,8× sopra il
  floor DRAM (sm_50 senza DP4A), coerente con S0.
- **down scaglionato per ordine di visita**: 90 → 273 µs/L dalla prima
  all'ultima card servita (veleno delle chiamate host, campagna "baton").
- **sync-wait inverso**: la prima card servita finisce prima e aspetta
  373 µs/L; l'ultima 36.

## Dove si perde il tempo (risposta)

Per layer (~1,12 ms, tutti e 48 seriali, nessuna sovrapposizione fra layer):

1. **~37%** lavoro grafo 3060 (le M10 ferme);
2. **~13%** dispatch host puro (plan+actq+jobs: staging, upload, lanci);
3. **~48%** fase M10 dentro run, in cui il grafo principale della 3060 è
   fermo: il tempo è catena [attivazioni → kernel M10 (a loro volta 5–8×
   sopra il floor) → sync/flag], non calcolo aggiuntivo;
4. **~1%** post.

Le utilizzazioni 35%/52% (3060) e 21–25% (M10) sono la traduzione di questa
struttura: collo di **latenza della catena per layer**, non di throughput di
nessuna risorsa. Leve già identificate nel WORKLOG: riordino del grafo /
sovrapposizione (soffitto +10–20%), mode-3 contatore aggregato (−100–150
µs/L), E2 pre-quant del layer successivo (nasconde ~115 µs/L di dispatch),
dieta istruzioni gup (+3–6%).

## Seguito (stessa sessione): le leve verificate

- **Riordino del grafo**: esiste già in albero come split-window P6 (`--spec-split`,
  `src/core/verify.cpp:1012`): i due gruppi di token della finestra sono interleaved
  (`pre(A,l) pre(B,l) post(A,l) pre(A,l+1)…`) così il lavoro del mixer B copre l'attesa
  delle righe A. Aritmetica identica; lo studio d'overlap la misura **~7% più lenta**
  (raddoppio dei dispatch per finestra, kernel dimezzati). Non ri-misurata oggi.
- **Dieta gup micro** (hoist qh IQ1_M + `#pragma unroll 2` del loop k): parity 0 failure,
  banco 294,8→289,2 (IQ2_S) / 287,3→293,3 (XXS) / 312,2→314,0 (IQ1_M) µs = rumore.
  Il kernel resta latency/issue-bound a 31-50% di occupancy; il SASS del loop caldo è
  ~347 istruzioni di cui solo 64 VMAD (il dot puro). **Rifiutata e ripristinata**; la
  variante che varrebbe (merge qh nel layout al load del blob) non entra nei 7,6/8 GiB
  di VRAM delle M10. Registro in `WORKLOG.md`, voce "arch-audit".

## Limiti

Giornata singola, screening 3×3 sequenziale (non ABBA/CI). Il braccio
profilato paga −3,6% di strumentazione: le fasi sono confrontabili solo
braccio-contro-braccio. Campionamento nvidia-smi a 1 Hz (medie su richiesta
intera, non per fase). Le mediane MWPROF coprono il mix di richieste della
corsa (story/fib/code + TTFT). Riconciliazione somma-fasi contro finestra al
99% con i numeri sopra.

## File

- `bench-baseline.json`, `bench-prof.json` — osservazioni 3×3+TTFT per braccio
- `gpu-sample-decode.csv` — campioni utilizzazione (1 s) durante la richiesta
- `prof-lines.txt` — righe MWPROF/decode-timing grezze (381)
- `prof-medians.txt` — medie/medianie calcolate
