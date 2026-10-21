# Gate di produzione per `--spec-split`: QUALITÀ E STABILITÀ VERIFICATE

**Data:** 21 ottobre 2026. **Config:** `strata-iq2xs-embed-vram-split.json` (produzione + `--spec-split`),
binario `d4f80481`. Controllo: la stessa config senza il flag. Protocollo: A/B di divergenza,
batteria formale, soak 30 minuti.

## 1. Divergenza A/B — la non-determinismo è della BASE, non dello split

Sei prompt fissi, temperatura 0, due passate per config:

| Confronto | prompt 0 (story) | 1 (fib) | 2 (code) | 3-4 (fatti) | 5 (haiku) |
|---|---|---|---|---|---|
| **baseline** pass0 vs pass1 | **DIFFER** (982/813) | **DIFFER** (443/475) | identico | identico | **DIFFER** |
| **split** pass0 vs pass1 | DIFFER (984/840) | identico | identico | identico | DIFFER |

La produzione **senza** split diverge run-to-run esattamente nello stesso modo (due storie
diverse dal primo token sulla stessa richiesta!). È la non-determinismo documentata della base
k8v4 + cache (2026-09-27-cache-parity: "same quality"; già visibile ieri: fib 463/506/512).
**Lo split non aggiunge divergenza** — in questo campione è anzi più stabile (fib identico).
I prompt fattuali/brevi sono stabili su entrambi.

## 2. Batteria formale su split — TUTTO VERDE

- **Risposte fisse**: 2+2→"4", Francia→"Paris", 17 primo→"Yes", 6×7→"42" — 4/4 ✓
- **8 richieste concorrenti identiche**: 8/8 completate, **1 solo output distinto** ("2, 3, 5, 7, 11") ✓
- **Prompt lungo (6.225 token)**: prefill ~68 tok/s, nessun hang; qualità del contesto lungo:
  prima parola del corpus → **"river"** (corretta); parola assente ("quantum") → **"no"** (corretto) ✓
- **Riuso del prefisso**: seconda passata del prompt lungo → **6.220/6.225 token cached**, 4,4 s ✓
- Nota protocollo: il conteggio parole su 6k token consuma >640 token di thinking senza
  contenuto — comportamento del modello (thinking budget), non un difetto dello split.

## 3. Soak 30 minuti — PULITO

276 richieste, **0 errori**, 85.710 token generati, traffico misto continuo (5 prompt rotanti,
thinking inclusi):

- **tok/s richieste lunghe (n=221)**: media 48,7, mediana 48,5, min 36,2, max 56,1
- **Nessun decadimento**: primo quarto 48,3 vs ultimo quarto 48,8 tok/s (nessuna perdita/decay)
- Motore vivo in tutti i 28 campionamenti; **zero** nuove righe d'errore nel log
- Utilizzazione in carico continua: 3060 **63-92%** (era ~35% pre-split: la pipeline riempie la
  GPU), M10 29-50%

## Verdetto

**PRONTO PER LA PRODUZIONE**:
- velocità: 52,44 tok/s (3×3) / 48,7 sostenuti nel soak, +21,8% vs baseline, TTFT invariato
- qualità: parità dei kernel gated (0 failure); la divergenza run-to-run è pre-esistente e
  identica senza il flag; factual/concorrente/long-context/prefix-reuse tutti corretti
- stabilità: 30 minuti di soak continuo senza errori né decadimento
- ripristino: un riavvio sulla config senza flag

## File

- `ab-split.json`, `ab-baseline.json` — i testi delle 4 passate (2 per config)
- `soak.jsonl` — richieste e campionamenti del soak
