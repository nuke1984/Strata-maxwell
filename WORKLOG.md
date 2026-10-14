# Handoff — Strata + Maxwell M10 su v0.1.40.1

**Data**: 2026-10-07
**Repo**: `/home/toor/Strata` · **ramo di lavoro**: `maxwell-0.1.40`
**Scopo**: portare avanti l'ottimizzazione "bilanciamento 3060 ↔ M10 ↔ CPU" senza ripartire da zero.

> **AGGIORNAMENTO — MODE 3 ATERRATA: +6,3% stesso-giorno, gates verdi, PRODUZIONE (2026-10-14)**
>
> **Configurazione vincente** (`strata-iq2xs-mode3.json`, server :8088): `STRATA_MW_GPUFLAG=2` +
> `STRATA_S2_SPLIT_D=1` + `STRATA_HOST_WAIT_US=0` + il kernel nuovo `copy_rows_from_mapped_after_mw`
> (elementwise.cu: il multi-wait E1 — main≥ring E 4 flag tier≥expected — piegato nel poll per-blocco della
> copia fusa 143-amdk; `STRATA_MW_FUSE_MW=0` lo disarma, default on quando E1 è armato).
> **Il pre-test chiave**: HOST_WAIT_US=0 (spin puro su h_seq_, niente chiamate driver → niente veleno) da
> solo valeva −4,5% (lo sleep_for di questo VM sfora ~100 µs/layer, non 20); mode-3 sopra ha aggiunto il resto.
> **Numeri** (stesso giorno, stesso protocollo 3×3): baseline 43,5 → mode3 **46,3 tok/s = +6,3%** (story +6,
> fib +8, code +3); engine-direct 53,8→49,8 ms/finestra. **Gates**: 391 ✓ 9999 ✓ round-trip al load ✓
> 391×8 concorrente ✓; PARITY invariata (il percorso default senza E1 è byte-identico: mw_wait_ non disarma
> più la fusione ma è false senza GPUFLAG).
> **Nota onesta sulla determinismo**: il run-to-run a temp 0 deriva ANCHE sul baseline puro (story 423/428/512
> su tre run identiche — proprietà della base k8v4 + righe cache-3060 vs righe M10 non bitwise-uguali tra
> kernel diversi, stato cache dipendente); mode3 mostra lo stesso profilo. "Lunghezze identiche" come gate è
> obsoleto SU QUESTA base — la determinismo stretta, se voluta, è un'indagine separata (equivalenza righe).
> **Percorso dei guadagni cumulati della campagna baton**: veleno identificato → E1m2 de-avvelena → D=1 →
> spin host → fusione mode-3. Prossime leve residue (bounded): dieta gup qh-merge (+2-4%), LUT down
> precomputata (+1-2%), shared-expert→M10 (+2%).

> **AGGIORNAMENTO — parità E1m2 spiegata e il disegno mode-3 (2026-10-14, seguito "baton")**
>
> **A/B pulita** (strumentazione OFF, altrimenti le chiamate-event del harvest differito ri-avvelenano la
> finestra): baseline 53,81 ms/finestra @ 2,00 token vs E1m2+D1 57,01 @ 2,11 → **26,9 vs 27,0 ms/token: parità**.
> Il "flat" storico di E1m2 è ora completamente spiegato: de-avvelena i kernel (tier 540→330 µs/layer) e la sua
> via di handoff si ripaga tutto (4 flag M10 da attraversare in PCIe, latenza poll-release del multi-wait,
> pubblicazione dell'expected-ring, granularità del sonno host 20 µs, post +18 µs/layer).
> **Il "ring gap" si dissolve**: ring 886 µs/layer = la catena seriale ONESTA (kernel M10 ~330 + fasi 3060
> ~400 + handoff ~155). Nulla manca: è il battente, misurato. Catena per layer ≈ fasi-grafo 400 + host 165 +
> M10 330 + handoff ~100 ≈ 1000 µs; le M10 stanno ferme ~700 (33% busy — l'utilizzazione osservata).
>
> **PROSSIMO: mode-3** (il disegno, da implementare a sessione nuova con gates pieni): finestra a zero
> chiamate host come E1m2, MA [4 flag tier + multi-wait + copia non fusa] sostituiti da **un contatore
> aggregato alzato dalla GPU**: la coda di ogni down M10 incrementa atomicamente UN contatore mapped; il
> quarto che arriva alza la flag principale; il nodo 143-amdk GIÀ catturato (wait+copy fuse) aspetta UNA
> parola e copia. Zero veleno, un solo attraversamento, fusione mantenuta. Stima: −100-150 µs/layer vs E1m2.
> **Lista leve residue (bounded)**: dieta gup qh-merge (−40-80), LUT down precomputata (−20-40), mode-3
> (−100-150), shared-expert→M10 overlap (−25). Somma ≈ +9-17%. Soffitto onesto ~60-65 tok/s single-user.

> **AGGIORNAMENTO — il veleno trovato: le chiamate host durante i kernel M10 (2026-10-14, campagna "baton")**
>
> **Meccanismo DIMOSTRATO**: ogni chiamata driver host durante la finestra dei down avvelena il completamento
> delle schede servite dopo (~+55 µs per scheda precedente; down 83→240-550 a parità di lavoro). Prove:
> (1) falsificazioni: scritture mapped→locali = stagger invariato; f16 (metà bytes) = nulla; local-down+D2H =
> peggiore; T256 = pareggio; microbench standalone (5 forme, incluso spinner 3060 e coppia upload→gup→down) =
> NESSUNO stagger → non è piattaforma, driver-pattern o topologia. (2) **STRATA_MW_REV=1 (ordine di servizio
> invertito): lo stagger SI INVERTE** (servita prima = veloce: 80 µs; ultima = 243) → è l'ORDINE di servizio.
> (3) STRATA_MW_GENTLE_SYNC (poll da 30 µs invece di spin): stagger persiste → non è lo spin ma le chiamate.
> (4) STRATA_MW_BLOCKSYNC (flag CUDA): rifiutato dallo stack (il prefill vede 0 MiB liberi) — documentato, off.
>
> **LA CURA: E1 mode 2 (`STRATA_MW_GPUFLAG=2`)** — l'host esce subito, il grafo multiaspetta i flag GPU:
> down **uniformi 123-135 µs** (da 82/190/304/407), sync-wait 0. Il muro del tier diventa PURO tempo kernel.
> Con esso, il re-sweep di STRATA_S2_SPLIT_D (le vecchie misure erano inquinate dal veleno!): **1→86,
> 2→126, 4→178, 8→287** — D=1 vince (il "2 misurato migliore" di ieri era un artefatto).
> **Stato**: kernel tier 540-660 → ~330 µs/layer (−40%), MA la finestra non cala (54,4→56,0): il ring-wait
> del grafo sotto E1m2 è 886 µs/layer contro ~480 di actq+kernel → **gap ~400-550 µs/layer nel multi-wait**
> (sequencing/expected-value in verify.cpp) — questo spiega anche il vecchio "E1m2 flat": cura lo stagger e
> poi il suo wait si riprende il guadagno. **Il prossimo bersaglio è esattamente quel gap.**
> Bracci nuovi (tutti opt-in-off, bitwise-neutral per costruzione): STRATA_MW_REV, STRATA_MW_GENTLE_SYNC,
> STRATA_MW_BLOCKSYNC, STRATA_MW_DOWN_LOCAL_TIMING (solo-misura, output garbage), STRATA_MW_IQ_T256,
> STRATA_MW_LOCAL_DOWN; strumentazione `MWPROF m10` per scheda (gup/down/sync per layer + MB/ideale).

> **AGGIORNAMENTO — S0: la verità del tier M10, misurata (2026-10-14, campagna "aggregate card")**
>
> **Domanda**: il pool M10 (19-23 ms/finestra = il 36% waitCPU) è bandwidth-bound o altro? Strumenti: eventi
> device per scheda (gup→down) + contatori byte/lanci sotto `STRATA_MW_LAYERPROF` (instrumentazione inerte di
> default, maxwell_experts.cpp; il report `MWPROF m10 CUDAx` stampa ogni 1440 layer/scheda).
>
> **VERDETTO**: le schede sono BILANCIATE (5,7-6,6 MB/layer/scheda, max/mean 1,08 → row-split inutile) ma i
> kernel girano a **5-7× sopra il floor DRAM** (ideale ~73 µs/L/scheda, misurato 332-557):
> - **gup (dot i-quant IQ2_S/XXS/IQ1_M): 260-300 µs/L uniforme** = instruction-throughput-bound: sm_50 non ha
>   DP4A (4×VMAD per dp4a) + girofpolate grid/segni. A/B: SPLIT 1→~720, 2→~326, 4→~260 (default confermato);
>   STRATA_MW_IQ_T256 (256 thread/blocco, occupancy 31→50-100%): **pareggio** — non è occupancy, è il flusso
>   istruzioni (tenuto opt-in-off come riferimento).
> - **down (S2 planar): 83 µs/L su CUDA1 ma 240-550 sulle successive** per lo STESSO lavoro — contenzione in
>   ordine di visita sulla macchina virtuale (via condivisa: chipset/vIOMMU/driver; P3.4 aveva già visto i sync
>   concorrenti serializzarsi). A/B: STRATA_MW_ROWS_F16 (metà bytes scritti, ingaggiato): **nulla**; 
>   STRATA_MW_LOCAL_DOWN (staging locale + D2H + scatter host): **PEGGIO** (run 469→771 µs/L) — rifiutato,
>   tenuto opt-in-off. La catena paga max-card ≈ 540-560 µs/L.
>
> **Conseguenze sul piano**: niente leva economica sul tier M10 — la dieta-istruzioni del dot (merge qh al load,
> ~15-30% del gup) è l'unica via bitwise e vale +3-6%; la via grande resta il riordino del grafo (riempire
> i ~19 ms di attesa con fasi 3060 indipendenti, soffitto +10-20%) ora che S0 esclude l'aggiustamento M10.
> Binario: strumentazione + bracci opt-in tutti default-off; produzioni successive su questa base.

> **AGGIORNAMENTO — chiusura ricognizione: i tre colpi dell'ecosistema misurati, il muro resta la catena (2026-10-14)**
>
> **Metodologia**: profilo completo per-fase a 256k (`STRATA_VERIFY_PROFILE=1`, 205 finestre, T medio 2,57,
> 53,2 ms/finestra) + A/B env dei percorsi HC alternativi.
>
> **#1 IndexShare MTP: INESISTENTE qui** — il drafter di Strata è già attention DENSA su ring di finestra
> ("dense attention: no indexer state is read", mtp.cpp:337): non c'è indexer-per-step da riusare. La tecnica
> SGLang risolve un problema che Strata non ha.
> **#2 kernel HC**: i knob alternativi A/B sulla 3060 — GR_V3 (split 164-328 blocchi, NON bitwise): story
> −2,2 / code +1,7 → pari e scartato; GR_DOWN_MAX4: −1,4% (come upstream sul 5070); staged resta il default.
> Il profilo: hc0+hc1 = 5,3 ms/finestra contro un floor pesi di ~4,1 → recuperabile bitwise ≈ 1,2 ms (~2%):
> debole, giù nella lista.
> **#3 indexer QSA ∥ QKV**: q+q-idx 1,02 + scores+topk 0,07 + kv-resolve 0,01 = **1,10 ms/finestra (2%)** —
> anche la sovrapposizione perfetta vale ≤1,5%: depriorizzato con i dati.
>
> **IL dato che comanda tutto**: `waitCPU 19,08 ms/finestra = 36%` dell'intera finestra (attesa del flag
> principale = la catena 3060→dispatch→M10→post). Tutto converge lì: batching (plateau 58-59 = tasso della
> catena), E1 (neutro senza sovrapposizione), knobs host (neutri). **La prossima leva È la sovrapposizione
> nella catena**: il riordino del grafo (fasi 3060 indipendenti dalle righe esperte PRIMA dell'attesa —
> ai 19 ms di wait si possono riempire: shared+quant 1,19, out-proj 1,31, hc 5,3, head 1,35 sono tutte fasi
> che non dipendono dalle righe M10 del layer corrente). Stima di soffitto: 5-10 ms/finestra ≈ +10-20%.

> **AGGIORNAMENTO — port su v0.1.44 + ricognizione mondiale + sweep spec (2026-10-14)**
>
> **Ricognizione** (utente: "esplorare il mondo"): upstream 0.1.43/.43.1/**0.1.44** (376 commit, +2.171 righe
> sui nostri file) + ecosistema Qwen3.8-Flash-Next (SGLang day-0: **IndexShare MTP** = riuso della selezione
> QSA su tutti gli step del draft, N→1 invocazioni indexer — la nostra biggest unused lever a 256k;
> **kernel HC Mix/Combine small-M split-K**: 2× kernel, +7,6%/+5,5% end-to-end — il nostro hc0 è 3,58 ms/fin;
> QSA indexer sovrapposto a QKV su secondo stream; FlashQLA 2-3× forward GDN per Hopper). Il "5-11% RTX
> 30-series" di 0.1.43 è per la maggior parte inerte per noi (adaptive overlap = arena RAM, MMVQ = pack Q2_0,
> PCIe share = quota CPU): verificato, niente inseguimenti a vuoto.
>
> **Sweep spec** (base 256k+k8v4, 3×3+burn per arm): 2/3/**4**/5/6 = 41,6/44,0/**45,0**/42,0/41,1 — spec 4
> È già l'ottimo (curva concava pulita); a 256k il costo indexer del draft supera il rendimento oltre 4.
>
> **Port v0.1.44** (merge `7897f55b`, ramo `maxwell-0.1.44`): 7 conflitti risolti — il critico in verify.cpp:
> il fused wait+copy 143-amdk di upstream resta il percorso default; E1 multi-wait e f16-rows prendono i
> loro rami opt-in e disarmano la fusione SOLO se attivi. Il probe PLE: la logica STRATA_PLE_AUTO_MMAP di
> 0.1.43.1 subsume la nostra guardia Maxwell (rimosso il ramo-notizia). **Gate**: parity ×2 (0 failures,
> PARITY OK), 391/9999 ✓, velocità in pari con l'ancora (45,1 vs 45,0). **Knob 0.1.44 A/B**:
> COMMIT_BATCH neutro (default ON tenuto), VERIFY_ASYNC_LAUNCH −2%, POOL_UMWAIT −1,8% — tutti e tre
> coerenti con la diagnosi: il muro è la catena cross-GPU, non il percorso host che ottimizzano.
>
> **Prossimi colpi** (dalla ricognizione, in ordine): **IndexShare MTP** (algorithmic, a 256k il draft è
> 3,2-7,1 ms/finestra), kernel HC small-M (+3-5% stimato), QSA indexer ∥ QKV, poi i nostri E2/riordino grafo.

> **NOTA — adaptive tier: esiste, spento nei Maxwell; idea di elevazione M10→3060 (2026-10-13)**
>
> **Verifica meccanismo** (domanda utente: "gli esperti più usati vengono promossi verso la GPU principale
> coi turni?"): la cache 3060 si riempie ALL'AVVIO dal profilo fisso (`--expert-profile`, hottest-first;
> 1.874 slot). La promozione a runtime ESISTE — il **tier adattivo** (`--adapt-every 4` finestre,
> `--adapt-swaps 96`): conta gli instradamenti (`d.usage`), calcola i guadagni di swap e porta esperti
> caldi non-residenti nella cache su uno stream laterale (macchinaria pending/event/slot completa),
> persisterli con `--expert-profile-save` (il profilo stesso è conoscenza congelata di quel meccanismo).
> MA: con i tier Maxwell attivi la fonte RAM/file è rilasciata e il motore azzera `adapt_every`
> (generate.cpp:5517, "disable the two paths that would read it") — sul nostro server la split 3060/M10
> è **statica dopo il load**; non esiste elevazione M10→3060. Oggi non costa nulla (100% hit, zero
> letture file), ma se il routing del workload deriva dal profilo, la quota 3060 smette di essere
> ottimale in silenzio.
>
> **Idea futura**: riutilizzare la macchinaria (usage counting, pending/event swap, aggiornamento
> residency table) per swap **VRAM↔VRAM** fra cache 3060 e tier M10 (hottest-in / coldest-out, senza
> fonte RAM/file). Costo del trasporto: le righe esperte viaggiano già via host mapped; il blob
> (1,38 MiB/expert) dovrebbe salire D10→host→D0 — ~0,5 ms su ~2,5 GB/s, fattibile fuori dal percorso
> caldo (stream laterale, come il tier adattivo originale).

> **AGGIORNAMENTO — B-series: richieste simultanee / --batch — il plateau è la catena (2026-10-13)**
>
> **Idea (utente)**: GPU al 30-47% di utilizzo → c'è spazio per richieste simultanee. **Confermato, con
> forma precisa.** Senza `--batch` il server serializza FIFO (4 richieste: aggregato 42 ≈ singola).
> Con `--batch N` (slot = sessione propria, finestre greedy 1 token/slot, niente draft MTP):
> aggregato 39,4 → 46,4 → 52,4 → **58,2** (N=1..4), **plateau 58-59** a N=6 (58,0) e N=8 (59,4);
> per-request 14,6 @4, 9,7 @6, 7,4 @8. `--batch-mtp` (1 proposta MTP/slot/finestra) vince SOLO a
> N=2 (50,6 agg, +9%): a N≥3 la finestra porta S×2 token, i job per layer scalano lineari → netto
> −11% a N=4. Correttezza: 391 ✓ fino a 8 richieste simultanee, output identici fra slot.
>
> **Verdetto**: a N=8 satura l'utilizzo è ANCORA 47% (3060) / 32% (M10) — il plateau ~59 tok/s è il
> **tasso seriale della catena per-layer** (3060-pre → dispatch → M10 → post), non una risorsa GPU:
> converge con E1 da direzione opposta. Il batching come leva è esaurito; serve la sovrapposizione
> nella catena (riordino del grafo). **Matrice di serving**: 1 utente = spec 46,0; 2 = batch2+mtp
> 25,4/req (50,6 tot); ≥4 = batch4 14,6/req (58,2 tot). Config di test in /tmp/opencode (non commessi).

> **AGGIORNAMENTO — E1 "GPU-raised doorbell": costruito, corretto, neutro (2026-10-13)**
>
> **Cosa gira ora (opt-in, `STRATA_MW_GPUFLAG`)**: 1 = i flag per-tierMapped alzati dall'ULTIMO blocco
> del kernel down (fence system + atomicAdd su contatore device, reset in-stream) + il multi-wait nel
> grafo della 3060 (attesa: flag principale E i 4 flag tier ≥ l'atteso letto a run-time da uno slot
> ring per-step — la cattura cuoce solo puntatori); l'host mantiene il sync. 2 = anche IL SYNC SPARISCE
> dal percorso decodifica: dispatch senza attesa + staging doppiato (2 set alternati per servizio).
>
> **Risultati**: mode 1 neutro (58,3 ms/finestra, zero stall — la correttezza del nuovo gating è
> dimostrata). Mode 2: **il dispatch crolla 660 → 115 µs/layer (run 550 → 3,8!)**, zero stall, 391 ✓,
> 2000 token di race-smoke puliti — ma la finestra è PIATTA (59,1 engine-direct) e il server misura
> **44,1 vs 45,9 (−4%)**: il ring-wait assorbe tutto il tempo liberato. **Lettura**: la catena di
> dipendenze per layer (3060-pre 420 → dispatch 115 → M10 360 → post) era ED È il limite; l'handoff
> host è sceso da 330 a 115 µs/layer, ma senza sovrapposizione TRA layer il muro non si muove.
> **Decisione**: default OFF (il −4% non si regala); il valore è strutturale: l'host è libero per
> ~830 µs/layer — la prossima mossa È la sovrapposizione: **E2** (worker che pre-quantizza e pre-carica
> gli atti del layer l+1 durante il ring-wait, ~115 µs da nascondere) e **il riordino del grafo**
> (le fasi 3060 indipendenti dalle righe esperte — shared expert, parti attention — devono correre
> PRIMA dell'attesa dei flag M10: è la Fase 4 storica, ora quantificata in ~360 µs/layer recuperabili).

> **AGGIORNAMENTO — T1: il budget per-layer del roundtrip, etichettato (2026-10-12, notte)**
>
> **Strumento**: `STRATA_MW_LAYERPROF=1` (resta nel codice): timestamp per fase nel servizio layer,
> stampa ogni 1920 layer. Corsa engine-direct, GEN 250, 3.840 layer caldi.
>
> | fase | µs/layer | cos'è |
> |---|---|---|---|
> | ring wait | **420** | il lavoro 3060 del layer (attenzione/router/norm — GPU compute reale) |
> | dispatch plan | 7 | assegnazione kind |
> | dispatch actq | **90** | staging + 4 upload + 4 lanci gup |
> | dispatch jobs | 14 | tabelle job |
> | dispatch run | **530** | lanci down + **finish-sync** + attesa scatter |
> | post | 50 | pubblicazioni flag |
> | **totale** | **~1.118** | |
>
> **Il risultato chiave**: i kernel M10 computano ~360 µs e la 3060 ~420 µs — **~330 µs/layer (~37%
> della finestra, ~16 ms) non è compute di NESSUNA GPU: è il costo del handoff** (API host, sync,
> protocollo). Entrambe le GPU inattive ~2/3 del tempo: collo di LATENZA, non di throughput — esattamente
> la diagnosi. DoS (duty): 3060 ~38%, M10 ~32%, handoff ~30%.
>
> **Il protocollo, mappato**: le righe M10 (scatter mapped) sono guardate dal doorbell principale
> `h_flag_`, che l'host alza DOPO il finish-sync del layer. Il sync esiste per (a) riusare `h_up_`
> (buffer singolo) e (b) l'alzata del doorbell — NON perché il risultato serva all'host.
>
> **Design E1 "GPU-raised doorbell" (pronto per la prossima sessione)**: l'ultimo blocco del kernel down
> alza `h_flag_` (pattern last-block su contatore device, reset nello stream; monotono per ring value);
> il grafo della 3060 attende h_flag_ AND i 4 flag per-tier mapped (nuovo kernel wait multi-flag);
> l'host smette di attendere: il run scende da 530 a ~360 µs/layer. Con doppio buffer `h_up_` (2 set
> alternati) lo staging non corre col kernel precedente. **Atteso: −8 ms/finestra ≈ +15%.** Le fasi
> successive: E2 (pre-quant actq durante il ring wait, nasconde 90 µs), E5 (sonda overlap/ceiling),
> poi il sondaggio movable-work (shared expert 1,42 · out-proj 1,97 · hc0 3,58 ms/finestra).

> **AGGIORNAMENTO — actq/zero-copy: misurato e RIFIUTATO (2026-10-12, sera)**
>
> **Composizione finestra post-campagna-kernel** (engine-direct, corsa fredda): attesa GPU-reach 19,1 ·
> ms_run (lanci+sync M10) 22,0 · actq 4,2 · draft 7,1 (freddo) — il collo SI È SPOSTATO: non più il gup.
> L'anomalia actq (~87 µs/layer per staging+upload di ~16 KB su 4 device): già ottimale a livello di
> API (buffer pinned, UN memcpy impacchettato per device per layer); il costo È il prezzo delle chiamate
> CUDA in questa VM (~4 µs ciascuna, ~16/layer).
>
> **Esperimento zero-copy** (P7): staging pinned MAPPED, kernel che legge atti+jobs direttamente da
> memoria host via PCIe — elimina i memcpy (1 sola chiamata per device per layer). Sicurezza verificata
> (buffer per-tier, disciplina di sync per layer in decodifica E prefill). **Risultato: 28,2/34,1/33,8
> tok/s = −30%** — le letture mappate sono memoria di sistema NON cacheata su sm_50: il kernel va in
> stallo sulla latenza PCIe. **Rifiutato**: DEFAULT OFF, resta opt-in (`STRATA_MW_ZEROCOPY=1`) con la
> misura nel commento. **Lezione**: su questa architettura le righe mapped si SCRIVONO bene (finish le
> usa da sempre) ma si LEGGONO male.
>
> **Stato**: produzione sul build buono (40,6/50,7 verificati al ripristino, 391 ✓). Le successive
> candidate rimaste: ridurre le CHIAMATE API in altro modo (es. fondere gup+down per device per layer in
> un solo lancio — richiede sincronizzazione intra-griglia non disponibile su cc 5.0, o catene di eventi),
> oppure il lato 3060 (draft 3,2 ms, percorso host 14,5 ms).

> **AGGIORNAMENTO — campagna gup IQ, vittoria 2 (kernel) e termine della traslazione engine (2026-10-12)**
>
> **SPLIT re-sweep post-griglie**: 4 riconfermato (314/319 vs 2: 346/350 vs 1: 963 µs). **Unroll 4 del
> loop k**: neutro, rimosso. **`__launch_bounds__(128,4)`**: neutro, rimosso.
>
> **Riduzione SPLIT a shuffle di warp** (via shared `sh`, 1 KB risparmiato): somme nello STESSO ordine
> sequenziale → bit-identico. **Lezione dura**: la prima versione metteva gli `__shfl_sync` sotto
> `if (part == 0)` divergente — **comportamento indefinito**, parity 5.578 failure (una run precedente
> "0 failure" era un collegamento stantio del binario: da ora il gate parity si gira SEMPRE dopo un
> rebuild completo). La versione corretta fa shufflare TUTTE le lane (ogni gruppo converge sulla propria
> somma; solo il leader part==0 consuma). Bench: **300 → 274,8 µs × 3 run identiche** (spread azzerato).
> Kernel totale: **430 → 275 µs (−36%)**. Parity: 0 failure su tutto.
>
> **Engine A/B (3×3)**: story 40,0 · fib 51,0 · code 46,7 · **media 45,9 vs 46,0 = piatta**. Lettura
> onesta: la vittoria-1 (griglie) aveva catturato il guadagno engine disponibile; il −8% kernel
> aggiuntivo non trasla (≈1 ms/finestra ≈ 2%, nel rumore; la finestra non è più gup-bound in questo
> punto operativo). **Il cambiamento resta** (kernel più veloce e deterministico, zero rischio).
>
> **Prossimo passo obbligato prima di altro lavoro kernel**: re-profilare la finestra COM'È ORA
> (nsys a conteggi): la composizione è cambiata (gup 81% → ~65% della catena). Le candidate non-kernel:
> actq (3,5 ms), draft (3,2 ms), percorso host (14,5 ms). Il pipelining preload-words resta documentato
> ma con la nuova aspettativa: ~2% engine, non di più, finché la catena non torna il collo.

> **AGGIORNAMENTO — campagna gup IQ, vittoria 1: le griglie codebook in shared memory (2026-10-12)**
>
> **Harness nuovo**: `iq_gup_bench` (target CMake accanto a maxwell_iq_parity) — il kernel di
> produzione su righe GGUF reali, job da decodifica, burn 2 s + cudaEvent: l'iterazione del kernel scende
> da ~8 min (engine reload) a ~40 s. Baseline microbench (layer 0 IQ2_S, 6 job): **430 µs/launch,
> 14,6 GB/s** — coerente col profilo engine (393 µs).
>
> **La scoperta**: i punti di tabella (`iq2s_grid` 8 KB, `iq2xxs_grid` 2 KB, `iq1s_grid_gpu` 8 KB) sono
> `static const __device__` in **memoria globale**, indicizzati dai dati: catene di carichi
> latency-esposti dentro ogni dot. **Staging in shared una volta per blocco** (copie int4 coalesced,
> coperto dal syncthreads esistente): microbench **430 → ~290–320 µs (−25…−33%)**; IQ1_M 445 → 320.
> Parity: 0 failure su tutto (bit-identico per costruzione: stessa tabella, stessa matematica).
> **Misurato e scartato**: `__ldg` sui carichi del blob (302–388 µs: nullo/peggiorativo — i carichi
> semplici già passano da L1).
>
> **Engine A/B (stesso protocollo 3×3)**: story **40,6** (+8,6%) · fib **50,9** (+9,5%) · code **46,4**
> (+6,9%) · **media 46,0 vs 42,4 = +8,4%** — ben oltre ogni soglia. Lunghezze di generazione identiche
> alla baseline (numerica invariata), 391 ✓, round-trip gate PASSED ×4.
>
> **Prossima leva (documentata)**: software pipelining dei carichi di codice nel loop k (la lezione S2
> di 287a708 valeva −5% lì); richiede la scomposizione load/compute dei tre dot. Il banco (`iq_gup_bench`)
> è lo strumento: ~286 µs restanti contro il floor di banda ~72 µs (6 MB @ 83 GB/s) — margine ancora 4×.

> **AGGIORNAMENTO — diagnostica nsys della catena M10 (2026-10-12): il collo è il kernel gup IQ**
>
> Profilo CUPTI (nsys, corsa engine-direct, GEN 40 ≈ 20 finestre; **conteggi, non tempi** — regola del
> profilo): per finestra per M10 **98 kernel** (48 gup + 48 down) ≈ 0,3–0,4 ms di API di lancio = **~1,5%
> della finestra** → il "launch overhead" da 32% di llama.cpp **non è il nostro problema**: le nostre
> finestre batched per layer lo hanno già risolto per costruzione. Il GPU-reach wait (22,5 ms) È la
> catena M10: busy per M10 ≈ 22,7 ms/finestra, di cui **gup IQ = 81%** (down 16%, ~87 µs/launch).
>
> **Il gup IQ in decodifica** (mediana 333 µs, top ~1,4 ms = finestre prompt): IQ2_S 2.771 launch,
> media 393 µs, ~6,6 job, 40 registri · IQ2_XXS 907 / 298 µs / 46 reg · IQ1_M 249 / 445 µs / 48 reg.
> Stima banda (IQ2_S, ~8 MB/launch): **~21 GB/s contro il soffio pratico di 83** → il kernel è
> latency/ALU-bound sul percorso codebook, NON saturo di banda: c'è margine reale.
>
> **Prossima sessione (design contro il gup)**: ncu per stalli/occupancy (spiegare, non misurare);
> struttura ILP del loop lane-strided; riuso di `sact` su più job per blocco; interazione con SPLIT.
> Le leve già chiuse: SPLIT 4>2>1 (misurato), prefetch L2 (veleno, da M10-TEST), VMAD (già ovunque).

> **AGGIORNAMENTO — metodologia da M10-TEST-FINAL.md + re-baseline in persistence mode (2026-10-12)**
>
> **Incrocio con M10-TEST-FINAL.md** (la campagna llama.cpp/gemma4 sugli STESSI M10): le due patch vincenti
> loro (MMVQ Maxwell table + VMAD globale) sono **già** nei nostri kernel per costruzione (i nostri S2/IQ
> usano `dp4a_s32`/vmad ovunque e il design v2 a atti condivise multi-riga È il loro rpb=8; la loro
> dispatch MMVQ non gira mai sui nostri M10 — solo `strata_maxwell` è sm_50). L'MTP-lento loro non ci
> tocca (il loro dirupo era `ncols_dst==1`-only; le nostre finestre sono multi-token native). **Adottato**:
> persistence mode (`nvidia-smi -pm 1`, ora ON — **riattivarlo dopo ogni reboot**); la regola del
> profilo (CUPTI accelera il decode M10 ~17%: mai misurare sotto profiler); il soffio pratico DRAM M10
> ≈ **83 GB/s** (71 self-copy) per i calcoli di floor; il protocollo a blocchi appaiati qui sotto.
>
> **Protocollo di misura (sostituisce il 3×3 a occhio per i prossimi sweep)**: blocchi alternati
> A-B-B-A poi B-A-A-B (A = parent, B = candidata); screening R=3, conferma R≥7; delta per blocco =
> tg_B/tg_A − 1; CI95 (Student-t) sui blocchi indipendenti, non sulle ripetizioni; soglia di
> promozione **≥ +0,5%** con CI sopra zero; scarto immediato per errore numerico nuovo, crash, o
> regressione >2% ripetuta; Amdahl sul percorso critico misurato, mai sulle quote aggregate dei kernel.
>
> **Re-baseline in persistence mode** (stessa config finale, burn + 3×3): story **37,4** · fib **46,5**
> · code **43,4** · media **42,4 tok/s** — invariata (attesa: il beneficio è al riavvio, non a regime).
> Carico 471 s (pari al solito: dominato dai 34 GB da NVMe). 391 ✓.

> **AGGIORNAMENTO — sweep miglioramenti su IQ2_XS @4096, split M10@150 (2026-10-12)**
>
> **Baseline (3 prompt calibrati × 3 run, protocollo burn+3×3)**: story **36,6** · fib **46,8** · code
> **42,9** · media **42,1 tok/s** — con lo split 22.702 esperti su M10 (riserva 150) + 1.874 in cache,
> 4,17 GiB di VRAM 3060 liberi, "+83–87% su PCIe" (più voci sulla catena M10: il prezzo dichiarato).
>
> **T1 — STRATA_HOST_WAIT_US=10** (20→10): story 37,1 · fib 46,7 · code 43,4 · media **42,4 (+0,7%)** —
> marginale, dentro il rumore, ma gratuito e mai negativo: **tenuto** (ora nell'`env` della config).
> **T3 — STRATA_MW_ROWS_F16=1**: warm ~42,0, misto (story −0,6, fib −2,3, code +2,6) e lunghezze di
> generazione cambiate (fib 387 vs 354 — la numerica fp16 sposta l'output): **scartato**, conferma
> l'era-0.1.40 ("sotto il rumore, resta opt-in"). 391 ✓ anche con fp16.
> **T4 — sweep SPLIT del kernel gup IQ** (ora `STRATA_MW_IQ_SPLIT`, default 4; epilogo a chunk multipli
> bit-identico, parità 0 failure per 4/2/1 su sintetico E righe reali): **SPLIT=2 = 28,1 / 32,5 / 31,2
> (−23…−31%) — RIFIUTATO**; SPLIT=1 non misurato: stessa direzione strutturale (il doppio del lavoro
> seriale di dot per thread di nuovo, su 5 SM) — monotonia. **SPLIT=4 resta l'ottimo misurato.**
>
> **Config finale**: `--maxwell-reserve-mib 150`, `--expert-cache 1800`, `env.STRATA_HOST_WAIT_US=10`,
> SPLIT default. Conferma finale: 37,0 / 46,8 / 43,4, media 42,4, lunghezze identiche alla baseline
> (numerica invariata), 391 ✓.

> **AGGIORNAMENTO — chiusura del port (2026-10-11, notte): `--ple-io direct` è il DEFAULT del ramo**
>
> La guardia vive nel codice (`generate.cpp`, blocco della sonda PLE): con un tier Maxwell configurato la
> sonda **non passa mai a mmap**, stampa la riga "keeping direct" con il perché, e il commento in codice
> registra la misura completa (44–68 ms/finestra di fault sincroni, finestre 92–121 ms, −45% tok/s; direct
> 46,6 ms = parità 0.1.40) più il **TODO del fix upstream-grade** (misurare entrambe le mode, o gate su
> MemAvailable vs byte della tabella — la sonda attuale misura solo direct e non può vedere che mmap è
> peggio dove la RAM non tiene la tabella). Le config non portano più il flag (è il default); i backup
> con il flag restano in `/tmp/*.fix-backup`. Tag di chiusura del port:
> **`maxwell-v.0.1.42-Q2_0-IQ2_XS-OK`**.

> **AGGIORNAMENTO — sessione del 2026-10-11 (notte): CAUSA TROVATA E RISOLTA — il port 0.1.42 è a parità**
>
> **Causa radice**: 0.1.42 introduce una **sonda PLE all'avvio** che misura la lettura *direct* della tabella
> (qui: 4.731–5.721 rows/s ≈ 19–23 MB/s random 4 KiB) e, sotto 15.000 rows/s, **passa da sola a `--ple-io
> mmap`** ("using --ple-io mmap instead"). Sulla macchina da 64 GB dell'upstream le pagine della tabella
> (19 GB) restano nella page cache; **qui (7,7 GiB di RAM) la cache è sempre buttata fuori dai 34 GB di
> letture esperti del load**, quindi ogni `gather_batch` del k==0 di ogni finestra decodifica fa **fault di
> pagina sincroni da NVMe lento**: ~44–68 ms/finestra persi in `folio_wait_bit_common` (main thread in D).
> 0.1.40 usava `direct` (pipeline di pread asincroni che maschera la latenza) e per questo era veloce.
> **Non era un problema del port**: il codice era identico — era un knob deciso dalla nuova sonda.
>
> **Individuazione** (per il metodo, non rifare i miei errori): strace ha mostrato 40.738 `MADV_WILLNEED`
> (tutti a load: NativeHead/Dense/Embed + advise_pairs — innocenti); l'istogramma dei return-address su
> `file_read_bytes_` (24.576 chiamate = tutti gli esperti, a load, via `copy_blob` del tier M10 — legittimo)
> ha smentito il "57 GB per richiesta" del log (**quel contatore è cumulativo da avvio in 0.1.42**, non per
> richiesta: i 79k "major faults" includevano il load); il delta reale per richiesta era ~1.7k fault. La
> strada giusta è stata il A/B dei `--ple-io`: direct → **finestra 46,6 ms calda** (era 92–121).
>
> **Riscontro finale (produzione, stesso protocollo: burn + 3 prompt × 3 run)**:
> - **Q2_0**: 41,3–57,0 tok/s, media **50,9** (maxwell-0.1.40: 43,8–55,9, media 51,1 — **parità**, picco più alto);
> - **IQ2_XS** (@4096): 38,8–41,3 tok/s su chat reali (banda validata 35–40 — **a/oltre la banda**);
> - correttezza 391 / 9999 / Paris / 23 ✓; `expert_maxwell_parity` PARITY OK; `maxwell_iq_parity` 0 failure;
>   tail skip 0 (guardia `cff9c68e` attiva); round-trip gate PASSED ×4 a ogni avvio.
>
> **La correzione**: `--ple-io direct` in **entrambe** le config (`strata-q2_0.json`, `strata-iq2xs.json` —
> sono gitignore: questa riga è l'unica traccia durable, insieme al backup dei file). Con RAM piena
> (≥64 GB) il default mmap di 0.1.42 va bene e la sonda ha ragione; **su questa macchina no**.
>
> **Nota upstream-grade (futuro)**: la sonda misura solo `direct`, mai `mmap`: non può accorgersi che mmap
> è peggio. Un confronto entrambe-le-mode prima di scegliere sarebbe la correzione giusta a monte.

> **AGGIORNAMENTO — sessione del 2026-10-11 (sera): IL PORT 0.1.42 È LENTO — REGRESSIONE MISURATA, CAUSA IN SEGUITO**
>
> **Misura (protocollo: burn + 3 prompt fissi × 3 run, stessi prompt su entrambi i binari)**:
> - maxwell-0.1.40 (`strata.bak-0140`): **43,8–55,9 tok/s** (media ~51);
> - maxwell-0.1.42: **18,1–53,1 tok/s** (media ~28) — **−45%**, con varianza enorme per richiesta.
> **Firma della regressione** (`STRATA_DECODE_TIMING=1` + strumentazione MWPROF aggiunta a run() e a
> `expert_pool_dispatch_multi`): il loop decodifica passa da ~45–54 a ~90–107 ms/finestra; i timer interni
> del dispatch (plan/actq/jobs/run) e l'attesa GPU-reach sono IDENTICI ai 0.1.40; il tempo in più il main
> thread lo passa **bloccato nel kernel su page fault della mappa del file esperti**
> (`folio_wait_bit_common`, stato D al 40% dei campioni): **57 GB letti da storage e ~79k major fault per
> richiesta** contro **0,4 GB e 3–7k** del 0.1.40. Il contatore del file tier conta "33980,8 MB di expert
> reads" a richiesta anche con ogni esperto su GPU — contabilità di `blob_stable`/`io_blob` (nuove in
> 0.1.42: contano i byte dell'intero layer per chiamata).
> **Scartate con misura**: kernel Maxwell (identici byte a byte), dispatch/callback (0,39 ms/call), tail
> skip (0 skip dopo `cff9c68e`), PLE gather (non chiamato in decodifica), predittore lookahead
> (STRATA_LOOKAHEAD=0: nessun recupero), pcie-frac auto (single-GPU), route-resident/prior (opt-in),
> io_fill_ (STRATA_IO_PF_STAGE=1: nessun recupero).
> **Non isolato (onestà)**: il punto esatto che tocca la mappa per richiesta richiede perf (bloccato da
> perf_event_paranoid=4, nessun root). Il sospetto primario è l'interazione tra il nuovo anello
> io-prefetch/staging del file tier (0.1.42) e la RAM di 7,7 GiB di questa macchina: il working set
> richiesto non sta nella page cache, e ogni tocco freddo costa un fault da NVMe lento (19 MB/s random).
> **Decisione**: produzione su **maxwell-0.1.40** (`engine/strata` = `strata.bak-0140`); il port resta sul
> ramo `maxwell-0.1.42` (tag `maxwell-0.1.42-IQ2_XS` = stato "corretto ma lento"), da riprendere quando il
> path I/O sarà chiaro (o con perf abilitato, o su una macchina con RAM piena).

> **AGGIORNAMENTO — sessione del 2026-10-11 (pomeriggio): PORT SUL v0.1.42 — FATTO E VALIDATO**
>
> **Politica**: il v0.1.42 resta verbatim; il lavoro Maxwell è uno strato opt-in sopra (i ganci tutti
> sotto guardia, inerti senza `--maxwell-devices`). Merge `baf71a10` di `maxwell-0.1.40` (tag
> `maxwell-0.1.40-IQ2_XS`) su `v0.1.42` nel ramo nuovo **`maxwell-0.1.42`**; 8 file in conflitto risolti
> "vince l'upstream" (prefill: blocco CPU-share nuovo + blocco Maxwell, il layout Dm è blocco principale →
> righe Maxwell → righe CPU; generate: swap Foresight nuovo accanto al binding rows_f16, binding split
> N-stage dell'upstream al posto del vecchio 2-stage; pool: ctor upstream con `tasks`/`phase_tasks` + il
> nostro start lazy; server.py: `ordered_gpus` + le device Maxwell aggiunte alla visibilità).
>
> **Una interazione semantica trovata e chiusa** (`cff9c68e`): il tail skip nuovo di 0.1.42 (default 7 su
> CUDA) guarda solo `host_res` (la copia RAM): gli esperti dei tier Maxwell sembravano miss PCIe e venivano
> **azzerati** (44.582 skip nella prima run). Ora un esperto tenuto da un tier è "in VRAM" e non si salta;
> senza `--maxwell-devices` il comportamento è lo stock 0.1.42.
>
> **Validazione (tutta su questo motore)**:
> - parità kernel: `expert_maxwell_parity` **PARITY OK** (0/640, 0/2560); `maxwell_iq_parity` **0 failure**
>   su sintetico (iq2_s/iq2_xxs/iq1_m) e su righe reali del GGUF (layer 0/1/8), rel ~1,7e-07 vs i punti di
>   produzione della 3060 — come prima del merge;
> - round-trip gate **PASSED ×4** su entrambi i pack (Q2_0 e IQ2_XS), "expert file cache dropped (every
>   expert is on a GPU)" su entrambi;
> - correttezza: Q2_0 17×23=**391** ✓ (240k di contesto), IQ2_XS **391 / 9999 / Paris / 23** ✓ (@4096);
>   expert cache 100%, skip del tail a **0** dopo la correzione;
> - logpos A/B (stessi id di GEN al motore nuovo e al binario validato 0.1.40 `strata.bak-0140`): **23/23
>   argmax concordi**, delta medio **+0,02** nats/token — le differenze residue sono le cambi numeriche
>   dell'upstream 0.1.41/42 (clamp q8_1 alle emit, kernel verify), identiche per i due sistemi;
> - velocità: nelle bande attese (Q2_0 42–65 tok/s visti; IQ2_XS 25–40 tok/s su chat reali, spec 4).
>
> **Non misurato (onestà)**: nessuna card Pascal/Volta in macchina — "il v0.1.42 per Pascal/Volta non
> cambia" vale *per costruzione* (arch flag intoccate, ganci inerti) più il controllo su Ampere, non per
> misura diretta. Il logpos assoluto del prompt GEN a mano (−6,0) non è paragonabile ai −1,648 del
> WORKLOG (prompt loro): conta il delta A/B, non la media.
>
> **Prossimi passi**: le voci in sospeso di sotto (scaling contesto, needle, sweep SPLIT) ora su base
> 0.1.42; il binario di produzione è `engine/strata` (backup del 0.1.40: `engine/strata.bak-0140`).

> **AGGIORNAMENTO — sessione del 2026-10-11 (mattina): ANCORAGGIO DELLO STATO (housekeeping)**
>
> - **Tag**: lo stato validato IQ2_XS è ancorato — **`maxwell-0.1.40-IQ2_XS` → `c7157a7`** (annotato, gemello
>   di `maxwell-0.1.40-Q2_0` → `4ea6ce2`). Il binario in `engine/strata` è costruito da questo commit
>   (compilato 23:01, copiato 23:02 dell'8/10).
> - **Stash**: il vecchio `stash@{0}` (tentativo IQ2_XS non validato, parcheggiato il 9/10 su `4ea6ce2`) è
>   stato droppato — era `86e1386`; il lavoro valido era già sopravvissuto come i 4 commit puliti
>   `b91b72e`…`c7157a7`, quindi nulla è andato perso.
> - **Questo worklog entra in git**: prima era untracked — la registrazione delle misure non può più perdersi.
> - **Stato corrente**: sistema validato e in uso; run sane il 10–11/10 (chat reali 35–40 tok/s, expert cache
>   100%, arresti puliti). Lavoro solo locale: niente da spingere.
> - **Prossimi passi (in sospeso, per una prossima sessione)**: (1) scaling del contesto verso 240k come il
>   Q2_0 (riserva M10 600→300 = +1,2 GB e/o `--kv k8v4` = +0,7 GB; oppure 128k comodo — da misurare);
>   (2) needle test alla scala; (3) sweep SPLIT del kernel IQ (fissato a 4 dalla fusione).

> **AGGIORNAMENTO — sessione del 2026-10-08 (sera): IQ2_XS SUL TIER MAXWELL — PORTATO A TERMINE**
> **(lavoro pulito da `4ea6ce2`, niente stash: 4 commit `b91b72e`, `ff67a28`, `2be71c4`, `c7157a7`)**
>
> **Risultato**: l'IQ2_XS gira come il Q2_0 — **tutti i 24.576 esperti su GPU** (4×5.008 sulle M10, 6,73 GiB
> cadauna, + 4.544 slot in cache 3060; "expert file cache dropped (every expert is on a GPU)"), correttezza
> 17×23=**391 ✓**, logpos sani, **decode 43,2 tok/s @4096** (spec 4) — **A/B stesso config: Q2_0 42,7**:
> nessuna penalità end-to-end (il prototipo vecchio misurava il gup IQ a 0,84-0,88× dell'S2, ma il gup non è
> il collo della finestra). Prefill 25 token in 601 ms; 16.002/16.002 routed entries servite su GPU.
>
> **I due sospetti del vecchio tentativo, chiusi con prove**:
> - sospetto 1 (embedding IQ4_XS): **ESCLUSO** — run IQ2_XS *senza* M10 (cache 3060 + pool CPU; pack
>   ricostruito fresco con `iq_pack.py`, byte-identale a quello del vecchio tentativo — verificato, e la
>   copia vecchia eliminata): logpos sani, media **−1,6467** vs −1,6086 del Q2_0 (stesso prompt, 25 token;
>   posizioni prevedibili −0,008…−0,03; il modello rotto dava **~−6 ovunque**);
> - sospetto 2 (conversione src→blob del tier): **era il colpo** (coerente con il sintomo: allora non c'era
>   guardia, il codice leggeva le righe IQ con la geometria Q2_0 = righe plausibili-sbagliate, silenziose) —
>   e ora è chiuso *per costruzione*: il **round-trip gate** sotto.
>
> **Il lavoro**:
> 1. `b91b72e` guardia: `open()` rifiuta i formati che i kernel non prendono (prima: garbage silenzioso).
> 2. `ff67a28` i tre vec_dot IQ (IQ2_S=22, IQ2_XXS=16, IQ1_M=29 — le tre codifiche gate/up del file reale,
>    down Q2_0 ovunque) trascritti da `iq_kernels.cu` (llama.cpp) con **una sola sostituzione per corpo**:
>    `ggml_cuda_dp4a` → `dp4a_s32` (vmad, come m10-final.patch); fp16 letto **a bit** (`h2f_dev`, mai le
>    intrinseche: `__float2half` sull'M10 è uno stub che ritorna zero, misurato). L'attivazione è la
>    **stessa ActQ** che il tier già riceve: `ActQ.q` è `block_q8_1.qs` bit a bit e `d = f2h_rn(scale)` col
>    clamp #606 = i bit della d del quantizzatore device. Kernel `expert_iq_stage1_batch_v2q`: geometria
>    della v2q (epilogo quant fuso IDENTICO → il down consuma l'immagine FF byte-identica, kernel down
>    invariato), righe gate/up come blocchi GGUF grezzi (`o_gate/o_up/gu_row`), chiamate dot lane-strided
>    come `row_dot`. **Zero cambi a dispatch/prefill per gli atti** (il prefill riempie già le ActQ sempre).
> 3. `2be71c4` plumbing: blob store **per layer** (blob di dimensione diversa per layer: 1.510.400 /
>    1.305.600 / 1.177.600 B per esperto), conversione per tipo (gate/up **nativi verbatim**, down Q2_0
>    de-interleaved → S2 planar agli offset del layer, dimensione invariata); kernel down con
>    stride/offset parametrici (default = costanti S2 → percorso Q2_0 identico). **ROUND-TRIP GATE a ogni
>    `open()`**: un esperto per codifica per device, catena intera (blob convertito, ActQ packed, percorso
>    di lancio del tier) vs `to_float`+fp64: **rel ~1,2e-02** (il rumore di quantizzazione atti+FF), PASSED
>    su tutti e 4 i device a ogni avvio — un offset sbagliato dà ~1,0 e non passa.
> 4. `c7157a7` dispatch: i token kind-3 dei layer IQ ricevono anche la ActQ (prima: solo `nact_multi` Q8_K
>    per la CPU; il `begin` del tier leggeva ActQ mai riempite — con la guardia di (1) non accadeva nulla,
>    ma il supporto vero la richiede).
>
> **Parità** (`build/maxwell_iq_parity`, righe reali del GGUF IQ2_XS, layer 0/1/8 = IQ2_S/IQ2_XXS/IQ1_M):
> - (0) blocchi q8_1 derivati dall'ActQ vs quantizzatore device: **0/240 differiscono**;
> - (1) kernel M10 vs **punti di produzione della 3060** (`iq_mmvq`, stessi atti): **rel 1,7e-07**;
> - (2) immagine epilogo: 0 flip/1920, fasc max rel 7e-07; (3) vs riferimento float (atti grezzi, riferito):
>   ~8,6e-03. Sintetico (`--synthetic iq2_s/q2_0 …`): 0 failure.
> - **NB metodo**: un primo riferimento fp64+`to_float` gatato a 2e-5 FALLIVA a ~6e-5 sui layer IQ2_S/XXS —
>   è il **bias del riferimento**, non il kernel: gli epiloghi interi esatti dei dot (es. IQ2_S
>   `(sumi0*ls0+sumi1*ls1+(sumi0+sumi1)/2)/4`, troncamenti inclusi) non sono riprodotti da un
>   dequantizzatore float. Il riferimento giusto è il percorso di produzione della 3060 (stesso contratto
>   d'atti q8_1): 1,7e-07.
> - **Regressione Q2_0**: `expert_maxwell_parity` PARITY OK; logpos serve-mode **byte-identico** alla
>   baseline pre-cambiamento (24 posizioni, media −1,6086); 391 ✓; round-trip gate PASSED ×4 anche sul
>   pack Q2_0 nativo (tipo 42).
>
> **Logpos IQ2_XS con M10 attivi**: media **−1,6481** vs −1,6467 senza (delta medio −0,0014 — la classe di
> differenza atti q8_1-tier vs Q8_K-CPU che il motore già accetta tra i tier); posizioni prevedibili
> −0,01…−0,03. Metodo: `STRATA_LOGPOS` su engine serve (stdin `GEN`), `--short-read 640`.
>
> **Config**: `strata-iq2xs.json` + `run-iq2xs.sh` (porta **8088**, host 127.0.0.1; un sistema per volta —
> le GPU tengono un modello). Il vecchio `stash@{0}` (tentativo IQ2_XS) è ora inutile — il lavoro nuovo è
> completo e validato — e si può droppare.
>
> **Prossimi passi**: (1) scaling del contesto: 240k come il Q2_0 richiede le manopole (+1,48 GB di esperti:
> riserva M10 600→300 = +1,2 GB e/o `--kv k8v4` = +0,7 GB di cache, oppure 128k comodo; da misurare); (2)
> needle a scala; (3) sweep SPLIT del kernel IQ (fissato a 4 dalla fusione, come la v2q).



> **AGGIORNAMENTO — sessione autonoma del 2026-10-07 (pomeriggio)**
> - **Leva 1 attiva e verificata**: CPU host **54–70%**, RTX 3060 **57–69%**, M10 ~20%.
>   Finestra ~46,3 ms (vs 44,2 baseline); tok/s 44–46 (dipende dall'accettazione speculativa, T 2,46–2,64).
> - **`--host-core last`** aggiunto alla config (M5, ~1% gratis).
> - **H4 (togliere il `finish` M10 dalla critical path) NON serve**: `cudaStreamSynchronize` è un
>   blocco (0% CPU) e spostare l'attesa lato GPU non cambia la finestra. **SCARTATO dalle priorità.**
> - **Flag opt-in testati** (`STRATA_FUSE_HEAD_GR`, `STRATA_GDN_SPLIT`, `STRATA_GR_DOWN_MAX4`):
>   **neutrali su CUDA** (sono ottimizzazioni per Intel/AMD). Scartati.
> - **Leve rimaste, in ordine**: (1) ridurre il **lavoro non-esperto della 3060** (~24,6 ms/finestra:
>   attention/dense/router/head/hc0/shared); (2) **sovrapporre** la catena M10 al lavoro indipendente
>   della 3060 (riordino del grafo: es. shared expert tra flagA e flagB — guadagno teorico ~1,4 ms);
>   (3) **Fase 4**: far calcolare alle M10 un pezzo di non-esperto **in parallelo** alla fase pre-router
>   della 3060 (le M10 sono idle lì). La catena M10 (306 µs/layer) è al limite hardware del kernel.

---

## Bus PCIe / PLX — misure e conclusione (2026-10-07, aggiornata 2026-10-08)

**Topologia (lato host, non visibile dalla VM).** Le 4× Tesla M10 stanno su un unico **PLX PEX 8747**:
**16 lane reali** verso il CPU, 4×8 verso le GPU. La 3060 ha invece una root port dedicata. Lo
**stato del link lo comanda l'host**: il guest ne legge solo un valore stale/emulato (vedi sotto).

**Stato del link — Gen1 (stabile) dal 2026-10-07; un giorno di Gen2 il 2026-10-08, poi ripristinato Gen1.**
Il link parte a **Gen1 (2.5 GT/s)**; un primo tentativo di Gen2 (2026-10-07) non aveva cambiato la velocità
misurata ed era stato ripristinato. Il **2026-10-08 l'utente è riuscito a ritrainare il PLX a Gen2** e stavolta
ha preso: le misure *dal guest* raddoppiano (tabella sotto). **Ma l'ha poi riportato a Gen1**: a Gen2 il
sistema diventa **instabile** — compaiono **errori correggibili sul bus**, nonostante le schede siano
Gen3-capable — e il guadagno end-to-end è ~1% (invisibile). **La stabilità batte la banda: si resta a Gen1.**
La VM legge "2.5 GT/s" sui link in entrambi i casi — è una lettura **stale/emulata** (la legge anche per la
3060, che misura 13 GB/s): **la prova è la banda misurata, non il registro**.

**Misure fatte *dal guest*** (microbench CUDA sm_50, trasferimenti da 64 MB):

| percorso | 1× M10 | 4× M10 insieme | RTX 3060 |
|---|---|---|---|
| H2D bulk | 3,14 GB/s | 3,13 GB/s agg | 11,4 GB/s |
| D2H bulk | 3,35 GB/s | 3,35 GB/s agg | 13,2 GB/s |
| kernel → memoria mappata | 3,35 GB/s | 3,35 GB/s agg | 13,2 GB/s |
| lettura da memoria mappata | 3,14 GB/s | 3,13 GB/s agg | 11,3 GB/s |
| atti 11 KB H2D + sync | 11,7 µs | 25,9 µs | 7,8 µs |
| righe 56 KB (write mappata) + sync | 28,9 µs | 78,1 µs | 10,9 µs |

**Conclusione onesta (2026-10-08): il link era il tetto del bus e del down, NON della finestra.** Due
letture, in direzioni opposte, da correggere:
- La conclusione del 2026-10-07 ("il tetto di ~3,3 GB/s è il percorso della M10, non il link") era
  **sbagliata nel motivo**: col Gen2 lo stesso microbench dà **6,70 GB/s scrittura / 6,12 lettura
  aggregati = 2,0×**, esattamente il raddoppio del link — il percorso M10 rende ~84% di qualunque link
  gli si dia (e le 4 insieme condividono l'uplink: non scalano oltre).
- La mia conclusione del 2026-10-08 ("la leva ERA il link") era **sbagliata nell'uso**: il link è il tetto
  della banda e del down *in isolamento*, ma **non è mai stato il tetto della finestra** — il down è ~9%
  del percorso critico del layer e il guadagno end-to-end è ~1%, sotto il rumore.
In pratica: aveva ragione la vecchia conclusione ("la leva non è il link"), per il motivo sbagliato.

**Misure Gen2 (2026-10-08, il giorno del retrain) — cosa ha cambiato davvero** (sonda `bw_probe`: 16 MB per
device, 64 MB insieme, 20 iterazioni; i valori Gen1 di questa tabella sono quelli del microbench 2026-10-07,
taglia simile — il confronto a **condizioni identiche** è la riga del down di produzione, misurata con lo
stesso T6 in entrambe le generazioni):

| misura | Gen1 | Gen2 | Δ |
|---|---|---|---|
| banda mappata 4× M10, write aggregata | 3,35 GB/s | **6,70 GB/s** | 2,0× |
| banda mappata 4× M10, read aggregata | 3,13 GB/s | **6,12 GB/s** | 2,0× |
| down di produzione fp32 (4 M10, righe mappate) | 95,7/95,9 µs | **74,3/74,2 µs** | **−21,5 µs (−22%)** |
| down fp16 (stesso test) | 76,2 µs | 75,7/75,9 µs | a Gen2 la leva fp16 è superata |
| 3060 mappata (root port dedicata) | — | 13,2 / 12,4 GB/s | mai stata un collo |

**Effetto sull'engine: ~1%, invisibile.** Run calde a Gen2 (stesso binario `4ea6ce2`, config approvata):
finestre **39,5–40,9 ms**, **tok/s 49,0–52,1** (media 50,5), `run` **~209 µs/layer** (era ~229), finish sync
58,6–58,7, correttezza 391 ✓ 3/3. La baseline Gen1 *pulita* era **40,5 ms di media** (39,0–41,3) e
**48,6–49,7 tok/s** → **~1%**, dentro il rumore run-to-run (l'acceptance varia 1,74–2,19 tok/win). E l'utente,
con **diverse prove di inferenza reali, ha visto risultati identici**. (Il mio primo confronto, −4,2%, era
contro una baseline Gen1 sfortunata: non un numero solido.)

**Perché il −22% del down non si vede**: il down è **~9% del percorso critico del layer** (~833 µs = 40 ms /
48). Il layer è dominato dall'attesa della CPU per la 3060 (**421 µs/layer**) e dalla fase di serve
(**326 µs/layer**, di cui il run coi finish 209–229). −22% su un nono del tempo = ~2%: impercettibile.
A Gen2 il down diventa **compute-bound** (74 µs ≈ il floor del calcolo): l'uplink non è più il suo vincolo,
ma alla finestra non cambia quasi nulla.

**Il motivo decisivo per ripristinare Gen1: l'instabilità.** A Gen2 l'utente vede **errori correggibili sul
bus** (nonostante le schede siano Gen3-capable): il sistema è marginale già a Gen2. La stabilità batte il 2%
teorico → **si resta a Gen1**.

**fp16**: resta **opt-in e spenta** — a Gen1 il guadagno isolato sul down (−19,6 µs/layer) non si vede in
engine (finish sync 61,0 fp32 vs 61,7 fp16), a Gen2 sarebbe superata, e Gen2 è ripristinata per instabilità.

**Regola operativa (definitiva):** il link **resta a Gen1**. Niente retrain né scritture di registri PCIe:
a Gen2 il sistema è instabile e il guadagno end-to-end non c'è. **La banda del bus non è una leva di questo
engine** — chi riprende non riprovi il link: le leve vere restano il **gup** (168 µs/layer, latency-bound:
l'ILP nei 56 registri; gli sweep negativi sono nel punto 10 della sessione 2026-10-08), il lavoro della
**3060** (~63% della finestra, profilo C1 compatto) e la **serializzazione** (421 µs/layer di CPU in attesa).

---

## Contesto a 240k — PROVATO (2026-10-08, sera)

Config: **`--max-context 32768 → 240000`** (backup in `strata-q2_0.json.bak-32768`). Nient'altro cambiato:
la ripartizione degli esperti si è fatta **da sola** (la cache 3060 si dimensiona sullo spazio libero dopo
il KV; le M10 prendono tutto quello che avanza).

| | 32k (prima) | 240k (dopo) |
|---|---|---|
| cache 3060 | 4352 slot (5,60 GiB) | **2004 slot (2,58 GiB)** |
| M10 ×4 | 5056 cad. (6,51 GiB) | **5643 cad. (7,27 GiB)** |
| residenti | 24576 (tutti su GPU) | **24576 (tutti su GPU)** |

- Il conto del KV: **12 layer QSA** (`qsa_interval=4`) × **1056 B/cella** (`kv_q8_bytes_per_cell`) ≈ 12,4
  KB/token, + indicizzatore/rope ≈ **13,5 KB/token** → a 240k **~3,0 GiB** (i docs dicono 13,7 KB/token
  per Q2_0 ✓). La differenza si prende dalla cache della 3060 (−2348 slot) e va sulle M10 (+587 cad.).
- Verifiche: correttezza **391 ✓**; `/props` → `n_ctx: 240000` ✓; **needle a 39k token**: risposta
  `ZAFFIRO-7731` ✓ (l'attenzione raggiunge il centro del contesto); **riuso prefisso**: la stessa richiesta
  → **38917 token riusati in 138 ms** ✓ (le conversazioni lunghe crescono a costo incrementale).
- **Prefill: ~85 tok/s costante** (85,5 a 4k, 84,6 a 39k — lineare per token, **non degrada col contesto**).
  Un prompt da 240k in un colpo solo ≈ **47 min**; per una conversazione incrementale non è un problema.
- Decode a 39k: 44,9 tok/s (vs ~50 a contesto corto: l'indicizzatore QSA costa un po'). RAM invariata
  (il KV resta in VRAM: `--kv-resident` assente = 0).
- **Se serve tornare indietro**: `cp strata-q2_0.json.bak-32768 strata-q2_0.json` + riavvio.

---

## Pulizia disco e valutazione IQ2_XS (2026-10-08, sera)

- **Spazio liberato su `/mnt/DATASET`**: rimossi l'Unsloth UD-Q2_K_XL (74 GB) e il modello "Qwen3.8-HC" (36 GB,
  non referenziato da alcuna config — il suo "shard 2" era un symlink allo shard 2 del Q2_0, per non ricopiare
  la PLE table). **Da 78 GB a 186 GB liberi.** Il Q2_0 in `Strata-data` è **intatto** (è quello che l'engine
  usa: `--pack`/`--native`/`--ple-gguf`/`--mtp` puntano lì). `/home/toor/MODELS.GGUF` (Gemma 4 + Ornith, 26 GB)
  lasciato.
- **IQ2_XS: valutato, bloccato dal formato.** Il tier M10 è **S2-only**: blob fisso da 1.382.400 B, kernel
  `expert_s2_*`, conversione che assume i blocchi "18 byte / 64 pesi" del Q2_0; `tools/iq_pack.py` dice che gli
  esperti i-quant *"cannot be re-expressed in the Q2_0 pack form"*. I kernel IQ esistono solo per la 3060 (da
  ggml-cuda, con dp4a — che su sm_50 non c'è). Quindi l'IQ2_XS **non può girare sulle M10** senza un kernel
  nuovo + un layout blob per-layer; e senza M10 servirebbe CPU/RAM (35,5 GB di esperti vs 7,7 GB di RAM).
  **Primo passo prima di scaricare 68 GB: un microbenchmark del decode IQ2_XS su una M10** — il decode è
  ALU-intensive (codebook) e la M10 è debole in ALU: il rischio è che vada più lento del Q2_0.
- Se si procede: l'IQ2_XS (68 GB + ~1,5 GB di pack) va su DATASET (186 GB liberi ✓), l'MTP si riusa. Per la
  VRAM: riserva Maxwell 600→300 (+~1,2 GB sulle M10) e, se serve, `--kv k8v4` (−0,7 GB di KV → +~0,7 GB di
  cache sulla 3060) = ~1,9 GB, sopra i +1,5 GB che servono.

---

## IQ2_XS su M10 — studio e prototipo (2026-10-08, notte)

**Formati reali del modello** (letti dall'header GGUF, scaricando solo i primi 100 MB dello shard 1 — decide
lo scope prima di scaricare 68 GB):

| ruolo | tipo | layer |
|---|---|---|
| gate/up | **IQ2_S** (ggml 22) | 34 |
| gate/up | **IQ2_XXS** (16) | 11 |
| gate/up | **IQ1_M** (29) | 3 |
| **down** | **Q2_0 (42) = l'S2 di oggi** | **48/48** ✓ |

→ Servono solo **3 decodifiche** (non una per layer) e il **down riusa il kernel S2 esistente**: blob stimato
~33,7 GB (il Q2_0 è 34,0) → la VRAM non è un problema.

**La scoperta che cambia il costo**: l'M10 ha già `dp4a_s32` (PTX `vmad`, in `expert_maxwell.cu`) e le
`vec_dot` IQ del motore (`iq_kernels.cu`, da llama.cpp) si portano su sm_50 **cambiando solo `ggml_cuda_dp4a`
→ `dp4a_s32`**. Non serve riscrivere i decode.

**Prototipo `iq_proto.cu` (validato sulla M10, device 1):**
- **parità CPU** IQ2_S e IQ2_XXS: **max rel 1,7e-06** (gli int sono esatti; solo l'ordine di somma fp32) ✓
- **self-test `vmad` vs scalare: 0 / 1.048.576 diversi** ✓ (l'istruzione Maxwell è bit-esatta)
- **velocità ALU-bound** (working set in L2, 240 blocchi, stesso lavoro = 19,7M valori): **IQ2_S = 0,88×,
  IQ2_XXS = 0,84× l'S2** → il decode IQ **non è più lento** (la paura +20-50% era infondata).
- IQ1_M: esegue, valori finiti ✓ (niente riferimento CPU: `iq1s_grid_gpu` esiste solo per il device; coperto
  dal self-test del dp4a + sorgente identico a quello validato del motore).

**Atterrato nel motore** (compila, libreria `strata_maxwell` sm_50 ✓): le 3 `vec_dot` + `expert_iq_stage1_batch_v2q`
(stessa struttura della v2q: job per blocco, riga per 4 thread, atti in smem come `block_q8_1`, **epilogo quant
fuso identico** → il down S2 lo consuma così com'è) + l'entry point `s2_expert_maxwell_iq_gup_quant_batch_async`
+ `-Ithird_party/ggml` per le griglie. **Manca il plumbing**: conversione blob per-layer (GU copia nativa +
down interleaved→planare), dispatch per tipo in `maxwell_experts.cpp`, offset del down parametrici, parità nel
motore.

**Stato**: download in corso in `/mnt/DATASET/STRATA_MODELS/models/iq2_xs/` (~11,6 MB/s, ~2h). Il sistema Q2_0
attuale **non è stato toccato** (nessun binario copiato, nessun riavvio).

**Plumbing ATTERRATO e validato (2026-10-08, notte)**: layout device per-layer (S2 invariato: BLOB uniforme;
i layer i-quant: `[D codes][D scales][GU nativo]` con gli offset `O_IQ_*`, blob per-layer e stride device
paddato); la conversione in `MaxwellExperts::open` copia i blocchi nativi del gate/up e converte il down
Q2_0 come un S2; il dispatch in `begin_chain`/`finish_chain` sceglie gup IQ + down con gli offset IQ.
Le entry point batch hanno `blob_stride`/`d_codes_off`/`d_scales_off` con default = i valori S2, quindi il
percorso Q2_0 è byte-identico. **Parità rifatta: PARITY OK** (tutte le sezioni, identica a HEAD).
*Gotcha misurato*: il test di parità va lanciato con **`CUDA_VISIBLE_DEVICES=1`** — dopo il reboot la 3060 è
il device 0 e il binario sm_50 non ci gira (senza l'env: stage1 in overflow, il down "passa" per caso).

**Prossimi passi**: (1) fine download + pack (`tools/iq_pack.py`); (2) config `strata-iq2xs.json` + script
separato (porta 8087, un sistema per volta); (3) bring-up: correttezza 391, needle, logprob vs Q2_0, velocità.

---

### IQ2_XS — bring-up e caccia al guasto (2026-10-08/09)

**Fatto**: pack costruito (`iq_pack`, 1079 tensori, 302 nativi, arena 1.43 GiB, esperti 35,45 GB; tipi
confermati: IQ2_S 34 layer / IQ2_XXS 11 / IQ1_M 3, down Q2_0 su 48/48). Config + script separati
(`strata-iq2xs.json`, `run-iq2xs.sh`, `strata-server-iq2xs.sh`) — il sistema Q2_0 intatto.

**Il motore IQ è corretto**: (a) parità del prototipo (IQ2_S/IQ2_XXS vs riferimento CPU: max rel 1,7e-06,
dp4a vmad bit-esatto); (b) il test dell'**entry point del motore** con blob sintetico nel layout device
(`iq_engine_test.cu`): IQ2_S max rel 7,5e-04, IQ2_XXS 2,1e-04, 0/2560 fuori tolleranza; (c) la parità del
motore (Q2_0) resta **PARITY OK** con l'IQ attivo (plumbing byte-identico); (d) `native_expert_parity` sul
GGUF IQ2_XS reale: layer 0/1/8 ok (GPU dequant **0 bit diversi** su 4,9M valori, CPU-gpu 1,5e-02).

**Il guasto**: il modello IQ2_XS è **qualitativamente rotto** (logprob del prompt ~−6 nats dove il Q2_0 dà
−0,003; output degeneri, spec acceptance 0%, ~2× più lento). Il **fix delle righe gate/up** (la conversione
scrive gli slab gate‖up, il kernel leggeva la convenzione interleaved S2) era necessario ma non basta.

**Esclusi con prova**: i kernel M10 (entry point in isolamento ✓), i kernel nativi della 3060 (parità sul
GGUF reale ✓), il CPU (idem ✓ + @4k il tier CPU ha 0 entry e il modello resta rotto ✓), il router e i densi
(**byte-identici** tra i pack Q2_0 e IQ2_XS ✓), il plumbing del motore (Q2_0 col binario nuovo: 391 ✓,
55,8 tok/s, logprob sani ✓), il tier CPU/file (a 4k tutto residente: 20032 M10 + 4544 cache, file cache
droppata ✓ rotto uguale ✗).

**Restano due sospetti, in ordine**: (1) **l'embedding dei token** (IQ2_XS = **IQ4_XS** in mapped host
memory, Q2_0 = Q3_K): è la prima op di ogni token; se il percorso embedding nativo mishandla l'IQ4_XS →
modello uniformemente danneggiato ✓ sintomi; (2) la conversione del blob reale dal **source** del motore
(la parità nativa valida i kernel, non il `src->blob()` del percorso Maxwell). Test pronti: un dump
dell'embedding dequantizzato vs ggml, e un `--dump-routing` comparato Q2_0 vs IQ2_XS (2 riavvii).

**Stato attuale**: **Q2_0 riavviato e sano col binario nuovo** (391 ✓, 55,8 tok/s, logpos sani) — il
sistema dell'utente è a posto. L'IQ2_XS resta sperimentale e **non usabile** finché il sospetto (1)/(2) non
è chiuso. Config di test: `strata-iq2xs.json` (@240k) e `strata-iq2xs-t4k.json` (@4k).
*(2026-10-09: i file di avvio IQ2_XS — `run-iq2xs.sh`, `run-iq2xs-t4k.sh`, i due config e
`strata-server-iq2xs.sh` — sono stati **rimossi**; si ricreano dai percorsi annotati qui
(`/mnt/DATASET/STRATA_MODELS/{models,packs}/iq2_xs` + `mtp/rt`).)*

---

### Spostamento dati: tutto in `/mnt/DATASET/STRATA_MODELS` (2026-10-09)

**Regola nuova**: i modelli vivono **solo** in `/mnt/DATASET/STRATA_MODELS/{models,packs}/<nome>/`, più l'MTP
condiviso in `/mnt/DATASET/STRATA_MODELS/mtp/`. Niente più `/home/toor/Strata-data`.

- Spostato: `models/Q2_0` → `models/q2_0` (maiuscola uniformata a minuscola come `iq2_xs`), `packs/q2_0`,
  `mtp/` (usato da entrambi i sistemi). `/home/toor/Strata-data` rimossa.
- Copia `rsync -a` + **verifica checksum completa** (`rsync -rcn`): identico ✓ (models 66,4 GB / packs
  1,5 GB / mtp 6,9 GB). Spazio liberato su `/`: 23 → 93 GB (occupato 126 → 56 GB).
- Config aggiornati: `strata-q2_0.json` (pack, native, ple-gguf, mtp, tokenizer), `strata-iq2xs.json` e
  `strata-iq2xs-t4k.json` (solo `--mtp`). Script `run-*.sh` e `strata-server*.sh` invariati (puntano ai
  config in `/home/toor/Strata`). Backup dei config pre-modifica in `/tmp/opencode/q2move/`.
- Q2_0 riavviato dalla nuova posizione: **391 ✓**, 47,9 tok/s a freddo (46,4 a caldo), path nuovi
  verificati sia dal cmdline del motore sia dall'ultima riga di avvio nel log.

---

### Ritorno a `4ea6ce2` + tag `maxwell-0.1.40-Q2_0` (2026-10-09)

Le modifiche IQ2_XS (kernel IQ + plumbing Maxwell; modello qualitativamente rotto, causa mai chiusa)
sono state **parcheggiate in `git stash`** (`stash@{0}`) e il working tree è tornato **esattamente a
`4ea6ce2`** (ramo `maxwell-0.1.40`, `git diff 4ea6ce2` vuoto). I modelli restano entrambi in
`/mnt/DATASET/STRATA_MODELS`; `iq2_xs` è senza motore e si riprende più avanti dallo stash.

Creato il **tag annotato `maxwell-0.1.40-Q2_0`** su `4ea6ce2`, messaggio: "Q2_0 OK — sistema di
produzione validato (2026-10-09) … Engine costruito da questo commit (ramo maxwell-0.1.40): Q2_0 su
4×Tesla M10 + RTX 3060, contesto 240k, KV int8, con i dati spostati in /mnt/DATASET/STRATA_MODELS."
(`main`, `maxwell-m10` e il tag `maxwell-v0138` restano intoccati.)

**Engine ricompilato dal tree pulito** (0 errori; binario 57.133.136 byte, identico per dimensione al
pre-IQ; `engine/strata` aggiornato dal build, verificato `cmp` identico) e **riavviato**: 391 ✓,
52,1 tok/s, path dati nuovi nel cmdline e nel log. Backup del binario sperimentale IQ:
`engine/strata.bak-iq2xs`.

*Nota operativa*: al primo avvio post-build `strata-server.sh` ha dichiarato "morto" al primo check
(race: controlla prima che Python exec'd; il tool poi ripulisce il gruppo processi). Riavviato con
`setsid nohup ./run-q2_0.sh >> strata-server.out 2>&1 < /dev/null &` → READY in ~6,5 min. Se
ricapita, usare questa forma o ritentare lo script.

---

## Kernel S2 v2 + dispatch — le due leve che hanno pagato (2026-10-07, sera)

**Contesto**: analisi del giro dei token (nsys + SASS + microbench, tutto in-VM). Il collo vero è la
**catena M10 (kernel) + il lavoro della 3060**; il bus PCIe è escluso con misure.

**1. Dispatch — commit `48524d2`**
- **Un solo upload per layer**: atti + job table in un'unica area di staging contigua (`d_up_`/`h_up_`)
  → 1 `cudaMemcpyAsync` invece di 2 → **begin 21,5 → 18,1 µs/device**, **actq 4,16 → 3,7 ms/finestra**.
- **LPT**: i 4 `begin` sono seriali e il device più carico partiva per ultimo; ora parte per primo
  (`count_owned` + ordinamento) → **finish sync 83,2 → 78,2 µs**.
- *Scartato dopo misura*: **CUDA graph** della catena — in questa VM `cudaGraphLaunch` a 5 nodi costa
  **18,8 µs** contro **22,8** delle 5 chiamate: risparmio reale 4 µs, non vale la complessità.

**2. Kernel S2 v2 — commit `1496b1d` (il guadagno grosso)**
- **Diagnosi**: v1 fa **~640 load GLOBALI SCALARI degli atti per riga** e incatena i 4 `vmad` di ogni dot in
  serie; SASS: 810 istruzioni, ~1000 ops di scompatto, 56 registri, 56% occupancy; banda raggiunta
  **29 GB/s contro i 75,7 GB/s misurati** (microbench) → **NON era al limite hardware** (il commento P3.5
  "~90% del teorico" non regge alla misura).
- **v2**: atti **staged in shared memory** una volta per blocco (LDS 128-bit), scompatto 2-bit via
  **tabella di 256 entry in smem** (1 LDS invece di ~6 ALU per 4 codici), **4 catene `vmad` indipendenti**
  (gli interi sono esatti → **bit-identico** a v1).
- **Kernel** (njobs=12): gup **378 → 256 µs**, down **197 → 127 µs**, banda **29 → 43,3 GB/s**; registri 48,
  occupancy ~62%; su tutta la curva **−30% costante**.
- **Parity test esteso al percorso batch**: max abs/rel identici a v1 → PARITY OK.
- **Engine** (8 GB/4 vCPU): finestra **45,53 → 41,50 ms (−8,9%)**, tok/s **+5/+9%** (46,7–48,8), CPU phase
  **13,7 → 10,3 ms/finestra (−25%)**, finish sync **78,2 → 59,5 µs (−24%)**, correttezza ✓ (17×23=391),
  prefill lungo 3000 token OK. **`STRATA_S2_V2=0` ripristina v1** per l'A/B.

**3. Chiusi dopo misura**
- **P2P tra le M10**: `cudaDeviceCanAccessPeer` dice YES ma `cudaDeviceEnablePeerAccess` fallisce sempre
  ("mapping of buffer object failed" — IOMMU/vfio) → inutilizzabile; 3060↔M10 = no. Il tensor-parallel tra
  M10 non converrebbe comunque (stesso compute/banda, +comunicazione).
- **Profiling hardware**: `nvprof` traccia i tempi (nascondendo la 3060: `CUDA_VISIBLE_DEVICES=1`) ma le
  **metriche** falliscono ("Internal profiling error 4211"); `ncu` non supporta Maxwell → la prova si fa con
  SASS + microbench.
- **Documentazione** in `/home/toor/nvidia-docs/` (PDF): Maxwell Tuning Guide, CUDA C Programming Guide,
  Best Practices, PTX ISA, whitepaper GM107.

**2b. Prefetch del gup — commit `287a708`**
- **La misura decisiva**: il *pattern di accesso* del gup, senza dot product, fa **72,2 GB/s** (136 µs a
  njobs=12) — identico a una lettura sequenziale; il gup completo ne fa 256 → **~120 µs di compute NON si
  sovrapponevano alle letture**. (Isolato anche: la LUT costa solo +15 µs; 8 catene vmad peggiorano per
  registri. **Nota**: un mio test "ALU pura" con UN solo accumulatore vmad dava 1947 µs — quello misura una
  catena seriale, non l'ALU. Il confronto vero ALU-vs-LUT nel kernel è **35,2 vs 43,3 GB/s**: l'ALU è ~19%
  più lenta, non disastrosa; e lo spazio dei trucchi sui bit è già saturo (~6 op LOP3/BFI per 4 codici).)
- **Fix**: software pipeline (`block_pair_s2_pre`): il blocco di codici dell'iterazione successiva si carica
  *prima* del dot corrente → gup **255,6 → 241,7 µs**, **finish sync 59,5 → 55,9 µs**, **finestra 41,50 →
  40,29 ms (−2,9%)**. Parity bit-identica.
- *Provato e revertito*: lo stesso prefetch sul **down** (SC_D=10, 5 iterazioni) **peggiora** (126,7 → 146,4 µs).
- **Riverifica a caldo (2026-10-08, protocollo burn)**: gup **255,5 → 220,8 µs a njobs=12 (−13,6%)**, 137,4 →
  118,6 a njobs=6 (−13,7%). Il vecchio −5,5% era falsato dal clock (la prima misura della coppia era fredda);
  l'engine aveva già misurato il −2,9% sulla finestra sotto carico sostenuto.

**Dove resta il tempo dopo v2 (mappa delle fasi dell'engine, 2026-10-08)** — per layer (48 per finestra):
- **GPU-reach wait**: 421 µs/layer (la CPU gira a vuoto aspettando la 3060: è il lavoro della 3060).
- **Pool (dispatch)**: 326 µs/layer = plan 3,5 + **actq 77** (quant ~13 + 4 begin seriali ~64) + jobs 10 +
  **run 229** (i 4 finish sync aspettano le catene; il run = fine dell'ultima catena − inizio del run).
- La catena del device più carico (LPT, ~9 job): upload 20 + gup ~168 + quant 7 + down ~98 ≈ **293 µs**; il
  floor di banda della stessa catena ≈ 191 µs → **~100 µs/layer di margine residuo, tutto nei kernel**.
- *Gen2, un giorno (2026-10-08)*: col PLX a Gen2 il **down** scende **~98 → 74 µs** (kernel di produzione,
  4 M10, righe mappate) e la catena a ~269 µs — ma in engine il `run` scende solo **~229 → ~209 µs/layer** e
  la finestra **~1%** (invisibile). **Gen2 ripristinata a Gen1 lo stesso giorno: instabilità (errori
  correggibili sul bus) e nessun guadagno end-to-end.** Vedi «Bus PCIe / PLX».
Il draft vale ~3 ms/finestra; il lavoro della 3060 resta il 63% della finestra.

---

## Sessione 2026-10-08 — protocollo di misura, sweep SPLIT, A1 scartato con prova

**1. Scoperta di metodo (da tenere sempre)**: la M10 da idle sta a **P8 / 135 MHz** e il governor impiega
**secondi** di carico per salire a **P0 / 1137 MHz**. Tutti i numeri harness precedenti erano ~9%
pessimistici e alcune coppie A/B erano falsate (prima run fredda, seconda calda). `s2_bench` ora fa un
**burn di ~2 s a njobs=12** prima di misurare; i numeri sono riproducibili allo **0,3%**.

**2. Sweep SPLIT rifatto a caldo sui kernel v2** (commit `240f115`, hook `STRATA_S2_SPLIT_GU/_D`):
- gup: **GU=4 resta il migliore** (102/119/221 µs a njobs 5/6/12); GU=8: 115/135/257 (+16%); GU=2: 170/190/341.
- down: **D=2** (58/68/127); D=4: 75/88/167 (+32%); D=8: 114/135/261; **D=1**: 58/68/124 — pari fino a
  njobs=6, leggermente meglio a 12 (123,7 vs 126,9). Default invariati.
- Il commento "~90% del throughput teorico" era falso: i kernel fanno **~52 GB/s dei 75,7 GB/s** di picco
  misurato (microbench float4).

**3. Down al limite strutturale**: nessuna variante lo muove (SPLIT, prefetch dei codici, riduzione
smem→shuffle) — è latency-bound a ~1,9× il suo floor di banda (36 µs a 6 job). Parity OK per SPLIT_D=1.

**4. A1 (hoist/split dell'upload atti) — ANALIZZATO E SCARTATO SENZA IMPLEMENTARLO.** L'ordine attuale
(copia+lanci *per device*, device per device) è già lo schedule ottimale a chiamate seriali: la catena di un
device parte quando atterra la sua copia (≈ issue+20 µs) e i suoi lanci sono già emessi prima. Anticipare le
4 copie (prepare-first) ritarda i lanci del **primo** device — quello che vincola la catena (LPT) — e non
guadagna nulla sul ultimo perché la copia atterra comunque dopo il lancio. L'unica leva vera nella colla è
**meno chiamate API**: A2 (fondre `quant_ff` nell'epilogo del gup: −1 lancio/device + il kernel quant fuori
dalla catena ≈ **−20 µs/layer**) resta da fare, con parità bit-esatta obbligatoria.

**5. Commit della sessione**: `240f115` (hook SPLIT + commento corretto), `5e3466a` (D=1 nell'hook).

**6. A2 — fusione `quant_ff` nell'epilogo del gup (commit `e5b8602`)**. Il blocco v2 copre esattamente un
chunk da 32 valori del suo job (`PER_BLOCK=32`, `FF/32=20` chunk per job), quindi l'epilogo SwiGLU quantizza
in loco con lo STESSO albero di shuffle di `expert_s2_quant_ff`:
- **Parity bit-esatta**: `fused vs pair: q 0/640, scale 0/20, hx 0/20, out 0/2560 diff`.
- **Engine (caldo)**: `begin` **18,0 → 14,6 µs/device** (una chiamata API ~4 µs in meno) e **actq 3,58–3,70 →
  2,97–3,27 ms/finestra** (**−0,6 ms, −19%**); la catena accorcia ~3 µs/device (harness: la coppia
  102,5+6,9 µs vs il fuso 104,3 a 5 job). `STRATA_S2_V2=0` tiene la coppia v1.
- Effetto stimato sulla finestra: **~−0,75 ms (−1,9%)**, sotto il rumore run-to-run (±0,5 ms) ma con il
  meccanismo misurato fase per fase.

**7. Pattern test a njobs=6 (mai fatto prima: era solo a 12)**: la sola lettura del gup fa **69,7 µs
(70,5 GB/s)**; il gup reale ~120 → **+50 µs** di compute/latency (LUT +6, vmad ~13, il resto latenza/coda
d'onda). Questo è il margine residuo del kernel (~30–40 µs/gup), da attaccare con grid-stride/ILP.

**8. C1 — profilo degli stadi della 3060** (`STRATA_VERIFY_PROFILE=1`; il run col profilo è perturbato nei
tempi wall — stage 23,46 ms/finestra, waitCPU corrotto sui GDN — ma gli stadi GPU sono leggibili):
- GDN (ms/finestra): hc-read1+router 3,38 · q8+qkv 3,23 · z 2,10 · VRAM hits 2,01 · head 1,98 · out-proj
  1,50 · hc0 2,87 (norm 0,44 + down 1,09 + up 1,34) · shared+quant 1,11 · waitA 1,12 · copy+combine 0,94.
- QSA: q+q-idx 1,43 · hc-read1+router 1,13 · attention 0,84 · VRAM hits 0,67 · out-proj 0,56 · hc0 0,87.
- **(gap) 0,02+0,01 ms → nessun buco fra gli stadi**: la 3060 è già compatta (waitA piano 1,47 totali).
- **Conclusione per C2/C3**: i candidati offload (VRAM hits 2,7 · shared+quant 1,5 · head 2,0 ms/finestra)
  non pagano i trasferimenti (round trip ~10–20 µs/layer; la head ha ~600 KB di logits in uscita). Non si tocca.

**9. Sweep kernel negativo (misurato a engine fermo)**: riduzione a shuffle (2 KB smem in meno) e
`__launch_bounds__` (10/12 blocchi/SM): **il gup peggiora** (238,4 vs 224,8 µs a 12 job con lo shuffle;
247,0 con LB=10 — meno registri, meno ILP), il down guadagna ~1% (67,4 vs 68,4 a 6 job). Tutto revertito:
lo stato committato resta il migliore. **Lezione**: il gup è latency-bound e il suo ILP vive nei 56
registri del prefetch; aumentare l'occupancy a spese dei registri perde.

**10. Grid-stride/persistent del gup — TENTATO E SCARTATO (misurato)**. Kernel persistente con
work-stealing (contatore atomico per gruppo-riga, staging ricaricato solo al cambio job, contatori
azzerati dall'ultimo blocco), **parity bit-esatta OK** su entrambi i percorsi. Ma nell'harness (a engine
fermo): **n=6: 132,0 vs 120,0 µs (+10%); n=12: 246,1 vs 224,5 (+9,6%)** — il work-stealing annulla
l'amortizzazione (i gruppi consecutivi sono a `gridDim` di distanza → job diverso ogni volta) e soprattutto
**lo scheduler hardware già sovrappone lo staging dei nuovi blocchi con la coda dei vecchi**: la versione
persistente perde quell'overlap. Revertito.

**11. Righe fp16 — TETTO MISURATO, integrazione NON atterrata (piano pronto)**. Micro-test `t5_rows.cu`
(4 device che scrivono le loro righe in memoria mappata, 6 righe da 2560 float, come l'engine):
| caso | µs/layer (4 device) |
|---|---|
| righe fp32 (oggi) | **73,9** |
| righe fp16 | **43,0** (−31) |
| fp32 + copy-back H2D | 136,4 |
| fp16 + copy-back H2D | 87,4 (**−49**) |

**12. Righe fp16 — INTEGRATE come opt-in (commit `4ea6ce2`, `STRATA_MW_ROWS_F16=1`)**. Il tier Maxwell scrive
le sue righe come **H metà fp16 impacchettate nel proprio slot fp32** (le prime H/2 float): così l'offset
per-gruppo arriva dal `dst_mapped` esistente — nessun buffer nuovo, nessun plumbing nuovo. Il verify marca
le entry scritte dal tier in `GpuPlanSink::kind16` (il pool lo riempie dai kind che già calcola) e la sua
copia righe converte solo quelle (un terzo ramo nel kernel di copia: zero / fp32 / fp16→fp32).

**Due bug trovati e corretti, entrambi misurati**:
- La prima versione scriveva le metà in un **buffer separato** indicizzato per entry LOCALE al gruppo,
  mentre la copia per-gruppo del verify le leggeva all'offset di finestra: il **prefill** (`run()`, gruppi
  con tb>0) era corrotto — i logprob teacher-forced del prompt **crollavano** (|Δ| medio **2,56 nats**
  contro l'fp32; max 6,75). Impacchettare le metà nello slot della riga fa arrivare l'offset dal
  `dst_mapped` che già lo porta: ora i logprob concordano a **|Δ| medio 0,019 nats, max 0,187**, e
  17×23=391 è giusto **3/3** (la versione rotta: 1/3).
- Il **grafo catturato incapsulava un `kind16` nullo** (lo assegnava solo `set_plan_slot`, che gira dopo il
  capture): il kernel di copia dereferenziava null → **illegal memory access a layer 1**. Ora il sink
  inizializza `kind16` al setup.

**Numerica**: conversione hand-rolled round-to-nearest-even (l'intrinseca CUDA su sm_50 è uno stub:
verificato, ritorna 0); i denormali flushano a zero. Il kernel di produzione contro il suo gemello fp32:
**max rel 4,2e-4, 0/15360 righe fuori tolleranza**.

**Misura**: il down di produzione, 4 M10, righe MAPPATE (ciò che il run phase aspetta): **fp32 95,7/95,9 µs
vs fp16 76,2/76,2 = −19,6 µs/layer** (riproducibile) — **ma l'engine NON lo conferma**: nella stessa finestra
temporale, stesso binario, il `finish sync` (media su 320K campioni) è **fp32 61,0 µs vs fp16 61,7–61,8 µs**
(nessun guadagno; +0,7 µs = rumore). Ipotesi: il guadagno del T6 era in parte un artefatto del suo setup (il
fp16 scriveva in un buffer separato più piccolo = meno pagine/TLB), mentre nell'engine impacchettato le
pagine toccate sono le stesse. **Esito: opt-in, non default** — è corretto e validato, ma senza beneficio
dimostrato end-to-end.

**Aggiornamento 2026-10-08 (PLX Gen2, poi ripristinato a Gen1)**: a Gen2 la leva era superata (down fp32
74,3 µs sotto il vecchio fp16, e l'fp16 costava +1,6 µs) — ma Gen2 è stata ripristinata per instabilità, e
**a Gen1 resta comunque senza beneficio end-to-end**: il guadagno isolato sul down (−19,6 µs/layer) non si
vede in engine (finish sync 61,0 fp32 vs 61,7 fp16), perché il down è ~9% del percorso critico. Resta
**opt-in e spenta**.

---

## 0. Regole d'oro (leggere prima di toccare qualsiasi cosa)

1. **`--spec` NON si tocca**: resta al default (4). Niente tuning speculazione.
2. **La 3060 non cambia ruolo/kernel.** La sua cache esperti è stata toccata solo in un esperimento (poi ripristinato): trattarla con cautela.
3. **Un cambio per volta**, sempre A/B con misura prima/dopo. Le misure vengono prima delle ottimizzazioni.
4. **Niente push su GitHub.** Tutto locale.
5. `main` (v0.1.38 originale) e `maxwell-m10` (lavoro su v0.1.38) **non si toccano**; c'è il tag di sicurezza `maxwell-v0138`.
6. **Il link PCIe resta a Gen1.** A Gen2 il sistema è instabile (errori correggibili sul bus, schede Gen3-capable) e il guadagno end-to-end è ~1% (invisibile): niente retrain, niente scritture di registri PCIe. La banda del bus **non è una leva** di questo engine (vedi «Bus PCIe / PLX»).

---

## 1. Stato del repository

```
maxwell-0.1.40 (HEAD)  4ea6ce2  maxwell: fp16 expert rows, opt-in (STRATA_MW_ROWS_F16)
                       e5b8602  maxwell: fuse the FF quantisation into the v2 gate/up kernel
                       5e3466a  maxwell: accept SPLIT_D=1 in the A/B hook
                       240f115  maxwell: SPLIT A/B hooks and a warm-clock sweep of the v2 geometry
                       287a708  maxwell: software-pipeline the gup code loads (gup −5%, window −2,9%)
                       1496b1d  maxwell: S2 batch kernels v2 — shared-memory acts, table-driven unpack, 4-way ILP
                       48524d2  maxwell: one staged acts+jobs upload per layer, and begin the busiest device first
                       b663ede  waits: sleep the host doorbell and block the GPU on a driver value wait
                       15ac39c  pool: create the CPU expert-pool workers lazily
                       f149d5d  maxwell: experts on Tesla M10 (SM 5.0) with mmap expert source
82f46a8                        base: v0.1.40.1 upstream

main                   99f3dbd  v0.1.38 originale (intatto)
maxwell-m10            aa6d1bb  lavoro Maxwell su v0.1.38 (intatto) — tag: maxwell-v0138
```

- Working tree **pulito** (solo untracked: `WORKLOG.md`, `strata-api-key.txt` — non codice).
- Diff totale vs 0.1.40.1 puro: **22 file, +2697 / −24** (`git diff 82f46a8..HEAD --stat`).
- I commit sono **solo locali**, mai pushati.

---

## 2. Cosa contengono i commit del port (`f149d5d`, `15ac39c`, `b663ede`)

### `f149d5d` — il port (Maxwell su 0.1.40.1)
Port del backend Maxwell da v0.1.38 a v0.1.40.1, via cherry-pick + risoluzione manuale di 6 conflitti
(`CMakeLists.txt`, `expert_source.{hpp,cpp}`, `verify.cpp`, `prefill.cpp`, `generate.cpp`).
- **File nuovi**: `src/core/maxwell_experts.cpp`, `include/strata/core/maxwell_experts.hpp`,
  `src/kernels/cuda/expert_maxwell.cu`, `include/strata/kernels/cuda/expert_maxwell.hpp`,
  `src/kernels/cuda/expert_maxwell_parity.cpp`.
- **Integrazioni**: `expert_source.*` (dispatch kind=3, `release_arena`, `drop_file_cache`),
  `peer_experts.*` (`set_maxwell_portable`), `verify.cpp` (portable), `pinned.cu`, `prefill.*`
  (`set_maxwell`, `finish_to`), `generate.cpp` (opzioni `--maxwell-devices`/`--maxwell-reserve-mib`,
  apertura tier, `set_maxwell_portable`, rilascio arena, drop cache), `CMakeLists.txt`
  (`strata_maxwell` static lib, `CUDA_ARCHITECTURES 50`), `serve/server.py` (CUDA_VISIBLE_DEVICES
  include i device Maxwell).
- Verificato: build verde, **parity test OK sul M10**, engine funzionante.

### `15ac39c` — pool worker lazy + diagnostica
- I worker del pool CPU nascono **lazy** (`ExpertPool::start_workers()`), al primo batch reale:
  con tutti gli esperti su GPU **non vengono mai creati** (79 → 64 thread). File: `pool.hpp/cpp`,
  `generate.cpp` (messaggio di log).
- Diagnostica env-gated `STRATA_MAXWELL_TIMING` (begin/finish M10 in µs) in `maxwell_experts.cpp`.

### `b663ede` — Leva 1: attese efficienti
- **Host**: dopo un breve spin il thread dorme (`STRATA_HOST_WAIT_US`, default 20 µs; `0` = spin puro).
  File: `src/core/verify.cpp` (loop host del verify).
- **GPU**: `wait_flag_ge` usa **`cuStreamWaitValue32`** (attesa driver-level, **nessun kernel residente**),
  catturata nel grafo verify; fallback automatico al kernel. File: `src/kernels/cuda/verify_kernels.cu`.
  `STRATA_WAITVALUE=0` ripristina il kernel.

---

## 3. Hardware e come si avvia/misura

**Hardware**: RTX 3060 (CUDA device 0) + 4× Tesla M10 (device 1–4, sm_50, 8 GB ciascuna).
CPU: Xeon E5-2683 v4, 16 core fisici.

**Avvio**:
```bash
cd /home/toor/Strata
cmake --build build -j16          # build
cp build/strata engine/strata     # il server usa engine/strata
nohup ./run-q2_0.sh > out 2>&1 &  # server su 0.0.0.0:8087
```
- Config attiva: `strata-q2_0.json` (untracked, gitignored). Args chiave:
  `--pack`, `--native`, `--ple-gguf`, `--expert-profile data/expert-profile.bin`,
  `--expert-cache auto`, `--prefill auto`, `--spec 4` (NON toccare), `--mtp`,
  `--max-context 32768`, `--kv int8`, `--maxwell-devices 1,2,3,4`,
  `--maxwell-reserve-mib 600`, `--mmap-experts`.
- API: `http://127.0.0.1:8087/v1` (OpenAI) · key `O2kVxQBDDxANYak0ocuhigbQTzhzw1fJt7AaqqSg`.
- Load completo: ~250 s.

**Misure**:
| Env | Cosa dà |
|---|---|
| `STRATA_DECODE_TIMING=1` | riga `strata decode timing`: wait / plan / actq / jobs / run / stage / draft per finestra |
| `STRATA_VERIFY_PROFILE=1` | riga `strata decode GPU stages`: tempo reale per stadio GPU (waitA/waitB/waitCPU inclusi) |
| `STRATA_MAXWELL_TIMING=1` | `Maxwell TIMING begin/finish`: µs per device per layer |
| `top -b -H -n1 -p $(pgrep -x strata)` | CPU% del thread host |
| `nvidia-smi --query-gpu=index,utilization.gpu --format=csv` | GPU% (3060 e M10) |

**Manopole introdotte**:
- `STRATA_HOST_WAIT_US` (default 20): grana del sonno host; `0` = vecchio spin.
- `STRATA_WAITVALUE` (default 1): `0` = ripristina il kernel di attesa.
- `STRATA_MAXWELL_HOT=N` / `STRATA_MAXWELL_CAP=N` (dal port): esperti caldi duplicati sulle M10 / cap totale.

**Backup config utili**: `strata-q2_0.json.bak-spec4`, `.bak-pre-hotmove`, `.bak-pre-pleram`,
`.bak-pre-mmap`, `.pleram-on`, `.bak-nomaxwell`.

---

## 4. Le misure (Fase 0 — completata)

### Baseline (spec 4, T medio 2,62, prompt lungo 1200 token → 48,1 tok/s)
| Voce | ms/finestra |
|---|---|
| **GPU-reach wait** (host aspetta la GPU) | **19,2** |
| **per-layer host** | **19,9** |
| · plan | 0,19 |
| · actq (include i 4 begin M10) | 3,66 |
| · jobs | 0,42 |
| · **run = finish M10** | **14,7** |
| stage | 0,4 |
| draft (MTP) | 3,0 |
| **totale** | **44,2** |

### Profilo GPU per stadio (`STRATA_VERIFY_PROFILE=1`) — il dato decisivo
- GDN: `waitCPU 10,95` · hc-read1+router 3,23 · q8+qkv 3,05 · z 1,99 · VRAM hits 1,99 · out-proj 1,43 · head 1,87 · hc0 2,74 · shared 1,07
- QSA: `waitCPU 3,40` · q+q-idx 1,35 · hc-read1+router 1,07 · attention 0,92 · VRAM hits 0,68
- **totale 40,57 ms/finestra**; `waitA`/`waitB` **trascurabili** (0,2–0,5).
- → La GPU aspetta la CPU per **14,35 ms/finestra** (`waitCPU`); il lavoro GPU reale è ~26 ms, di cui
  **esperti solo ~2,7 ms** → il resto è **NON-esperto** (attention/dense/router/head/hc0/shared).

### Altri dati
- **M10 finish = ~306 µs/layer** (sync); begin = 18 µs/device. Micro-benchmark: la latenza grezza della
  catena (2 memcpy + 3 kernel + sync) è **25 µs** → i 306 µs sono **esecuzione dei kernel S2**, che sono
  già `int4`-vettorizzati e al **~90% del limite pratico della M10** (commento misurato in `expert_maxwell.cu`).
- **Spec sweep**: spec 2 → 46,1 · spec 4 → 48,1 · spec 6 → 44,9 tok/s (spec 4 è il migliore; **lasciarlo**).
- **`--host-core last`**: 48,7 vs 48,1 (~1%, gratis).
- **M6 (T=1)**: non eseguibile — il pack nativo rifiuta `--spec < 2`.
- **Hot-move** (esperimento): spostare esperti dalla cache 3060 alle M10 → 3060 resta 100% e **rallenta**
  (la 3060 è carica di non-esperto, non di esperti). Config ripristinata.

### Leva 1 — risultato (fatta)
| | Baseline | Leva 1 |
|---|---|---|
| CPU host | ~100% | **60–70%** |
| RTX 3060 | 100% | **57–69%** |
| M10 | 20–25% | 14–27% |
| tok/s (lungo) | 48,1 | **46,2 (−4%)** |
| test brevi | 53,8 / 54,1 | 54,8 / 55,2 (invariati/meglio) |

Correttezza verificata (144, Fibonacci, racconto). Il calo di tok/s viene dal sonno host da 20 µs.

---

## 5. Diagnosi consolidata

- I "100%" di CPU e 3060 erano in gran parte **spin** (ciascuno aspettava l'altro): CPU ~19 ms di spin +
  ~20 ms di lavoro; 3060 ~14,4 ms di `waitCPU` + ~26 ms di lavoro.
- Il **collo residuo** è il **lavoro host esposto** (il `finish` M10, che la 3060 aspetta → `waitCPU`)
  e il **lavoro non-esperto della 3060** (~24,6 ms/finestra).
- La **flagA è già anticipata** (i residenti della 3060 si sovrappongono al dispatch); è la **flagB**
  (righe M10) a essere esposta.
- La catena M10 è **serializzata** con la 3060: ogni cosa messa sulle M10 **allunga** la critical path,
  a meno che non sia **sovrapposta**.

---

## 6. Piano rimanente (in ordine consigliato)

### Fase 2 — HOST
- ~~**H4 — togliere il `finish` M10 dalla critical path.**~~ **SCARTATO (2026-10-07)**: il
  `cudaStreamSynchronize` è un'attesa bloccante (0% CPU) e spostarla lato GPU non cambia la finestra
  (la GPU aspetterebbe lo stesso tempo). Vedi l'aggiornamento in testa al documento.
- **H4 originale (per riferimento)** — togliere il `finish` M10 dalla critical path.
  Oggi `finish_chain` fa `cudaStreamSynchronize(streams_[0])` per device e la 3060 aspetta la flagB.
  Idea (a): **handshake GPU-side** — il kernel down M10 scrive le righe e segnala una flag in memoria
  host mappata (`__threadfence_system` + store); la 3060 aspetta quella flag prima del combine.
  Idea (b): sync non bloccante (`cudaEventQuery`) o batch dei sync.
  File: `src/core/maxwell_experts.cpp` (`finish_chain`), `src/core/expert_source.cpp` (dispatch),
  `src/core/verify.cpp` (grafo, wait), `src/kernels/cuda/expert_maxwell.cu` (flag dal kernel down).
  Atteso: ~10 ms/finestra → +20–25% tok/s e meno lavoro host esposto.
- **H1** — parallelizzare `actq` (quantizzazione attivazioni) sui thread liberi.
- **H2** — parallelizzare i 4 `begin/finish` M10 su 4 thread (da ri-testare: in v0.1.38 era più lento).
- **H3** — ridurre le copie/sync host per layer.
- **H5** — usare i worker idle per plan/jobs/actq.

### Fase 3 — 3060 (lavoro non-esperto)
- **S2** — KV: `--kv k8v4`, `STRATA_KV_ROT=1`, `--kv-grow` (0.1.40, libera VRAM → più cache esperti).
- **S3** — flag kernel 0.1.40: `STRATA_ATTN_CHUNK`, `STRATA_GDN_CP_ASYNC`, `STRATA_HC_SPLIT`, `STRATA_LFUSE`, …
- **S4** — `--expert-cache`: con più VRAM libera, riportare più esperti sulla 3060 se aiuta.
- ~~S1 (spec)~~ — **escluso** (spec a default).

### Fase 4 — usare le M10 per il NON-esperto (il salto)
- **X1** — shared expert (FFN denso per layer) sulle M10 (estensione del kernel Maxwell a GEMV densa).
- **X2** — drafter MTP sulle M10.
- **X3** — offload parziale di attention/dense.
- ⚠️ Requisito: il lavoro spostato deve **sovrapporsi** alla 3060, non entrare nella catena seriale.
  Candidati dal profilo (ms/finestra): hc-read1+router 4,3 · q8+qkv 3,05 · hc0 3,58 · z 1,99 ·
  out-proj 1,97 · head 1,87 · shared 1,42.

### Fase 5 — throughput (opzionale)
- **B1** — continuous batching (`serve/test_parallel.py`): maschera il ping-pong con più richieste in volo.

### Micro
- **Tune `STRATA_HOST_WAIT_US`** (20 → 10 µs): recupera parte del −4% alzando un po' la CPU.

---

## 7. File chiave (punti di aggancio)

| Area | File |
|---|---|
| Backend M10 | `src/core/maxwell_experts.cpp`, `include/strata/core/maxwell_experts.hpp` |
| Kernel M10 | `src/kernels/cuda/expert_maxwell.cu`, `include/strata/kernels/cuda/expert_maxwell.hpp`, `src/kernels/cuda/expert_maxwell_parity.cpp` |
| Dispatch esperti | `src/core/expert_source.cpp` (`expert_pool_dispatch_multi`, kind 3), `include/strata/core/expert_source.hpp` (`ExpertDispatch.maxwell[]`) |
| Grafo verify / flag | `src/core/verify.cpp` (`run`, `set_plan_slot`, `raise_flag`, loop host), `src/kernels/cuda/verify_kernels.cu` (`wait_flag_ge`) |
| Prefill | `src/prefill/prefill.cpp` (`set_maxwell`, blocco righe M10 in `Dm`) |
| Driver/CLI | `src/program/generate.cpp` (opzioni, apertura tier, wiring) |
| Pool CPU | `src/kernels/cpu/pool.cpp/.hpp` (lazy workers) |
| Server | `serve/server.py` (CUDA_VISIBLE_DEVICES con device Maxwell) |
| Build | `CMakeLists.txt` (`strata_maxwell`, `CUDA_ARCHITECTURES 50`) |

---

## 8. Come riprendere

1. Verificare lo stato: `git log --oneline -3`, `git status`, engine attivo su `:8087`.
2. **Protocollo di misura (obbligatorio, lezione del 2026-10-08)**: l'harness deve fare un **burn di ~2 s**
   prima di misurare — la M10 sta a **P8/135 MHz** da idle e a **P0/1137 MHz** sotto carico: gli stessi
   numeri differiscono del **~9%**. `s2_bench` lo fa da sé; per l'A/B sull'engine servono **3 run calde**
   (la prima dopo il load è lenta) e si confrontano finestra + fasi (`plan/actq/jobs/run`) + begin/finish
   sync, mai una run sola.
3. Stato attuale: **v2 + prefetch + quant fuso nel gup** (commit `e5b8602`); finestra **39,0–41,3 ms**
   (media ~40,5), **~49 tok/s** sotto carico, `actq` 3,0–3,15 ms (era 3,6), `begin` 14,8 µs/device (era
   18,0), correttezza 17×23=391 ✓. Il margine residuo è nel **kernel**: il gup a 6 job fa ~120 µs contro i
   **69,7 µs della sola lettura** (pattern puro); il down è a ~1,9× il suo floor di banda.
4. Leve: **righe fp16 — INTEGRATE come opt-in** (`STRATA_MW_ROWS_F16=1`, commit `4ea6ce2`): il down di
   produzione è **−19,6 µs/layer** verso memoria mappata (misurato due volte), la numerica validata (logprob
   |Δ| medio 0,019 nats, 391 giusto 3/3); l'effetto sulla finestra è sotto il rumore, quindi resta opt-in.
   **Grid-stride/persistent: chiuso** (misurato +10%). Il resto è la 3060: compatta (gap 0,02 ms), i
   candidati offload non pagano i trasferimenti (vedi C1).
5. Dopo ogni modifica: build → `cp build/strata engine/strata` → riavvio (~440 s) → misure → commit.

**Nota**: il binario `engine/strata` è **build artefact** (gitignored); dopo ogni build va ricopiato.
