# IQ3 — Qwen3.8-Flash-Next IQ3_S sul tier Maxwell (Tesla M10)

Questo LOG registra **una voce per iterazione**: obiettivo, ipotesi, modifiche, comandi esatti per
aidrogen, risultato atteso e risultato misurato. Le voci sono in italiano; codice, commenti e messaggi
di commit sono in inglese, nello stile della repo.

Regole: una sola variabile per iterazione; niente numeri dichiarati senza misura; il percorso IQ2_XS
di produzione resta invariato; nessun `api_key` nella repo (i config del server restano fuori).

Indice:

- [Iterazione 0 — base `iq3` e baseline IQ2_XS](#iterazione-0--base-iq3-e-baseline-iq2_xs)
- [Iterazione 0b — `STRATA_MW_FASTCOMPLETE=1` (esperimento separato)](#iterazione-0b--strata_mw_fastcomplete1-esperimento-separato)
- [Iterazione 1 — censimento dei formati](#iterazione-1--censimento-dei-formati)
- [Iterazione 2 — gate/up IQ3_XXS / IQ3_S / IQ4_XS](#iterazione-2--gateup-iq3_xxs--iq3_s--iq4_xs)
- [Iterazione 3 — design: down IQ4_NL nativo (da rivedere prima del codice)](#iterazione-3--design-down-iq4_nl-nativo-da-rivedere-prima-del-codice)
- [Iterazione 3 — implementazione e verifiche di kernel (misurato)](#iterazione-3--implementazione-e-verifiche-di-kernel-misurato-2026-11-07)
  — con la [finestra operativa](#finestra-operativa-misurato-2026-11-07) in coda
- [Iterazione 5 — banco fisso e verifiche per la produzione (protocollo)](#iterazione-5--banco-fisso-e-verifiche-per-la-produzione-protocollo-prossima-finestra)
  — con le [misure](#iterazione-5--banco-fisso-contesto-lungo-e-soak-misurato-2026-11-07) in coda
- [Iterazione 6 — diagnosi: potenza, profilo per-layer, margine VRAM (protocollo)](#iterazione-6--diagnosi-potenza-profilo-per-layer-margine-vram-protocollo-prossima-finestra)

---

## Iterazione 0 — base `iq3` e baseline IQ2_XS   (commit `0283a48`, base `6175fa0`)

**Obiettivo.** Creare il branch di lavoro `iq3` da `main` (`6175fa0`), portare il percorso
**expert-split** da 4 a 8 dispositivi Maxwell e misurare la **nuova baseline IQ2_XS** con il motore
`main` (0.1.51) compilato da questo branch. La baseline è necessaria perché la patch `mw8`
dell'operatore era nata su un checkout più vecchio (motore 0.1.47).

**Ipotesi.** Le modifiche sono di sola capacità: non cambiano il percorso per dispositivo. Con 8 die
il tier deve caricare 8×~2838 esperti (~3,81 GiB per die), tutti i round-trip gate devono passare, e
decode/prefill devono restare nel range misurato il 5–6 novembre (decode 65–68,7 tok/s, prefill
~181 tok/s) entro il rumore. Se il numero di tier o i gate non tornano, l'ipotesi è smentita e non si
misura la velocità.

**Modifiche.** Tutte sul percorso **expert split** (whole experts); la modalità `tensor`/Meta resta
4-way e non è toccata.

| File | Modifica |
|---|---|
| `include/strata/core/expert_source.hpp` | `ExpertDispatch::maxwell[4]` → `[8]`; `mw_partial[4]` → `[8]`; commento aggiornato. |
| `src/program/generate.cpp` | `std::array<MaxwellExperts, 4>` → `8`; parser `--maxwell-devices` 1–8; testo di help; guard: se `e1_on && maxwell_count > 4` errore e `return 1` (E1 è 4-way per costruzione). |
| `src/core/expert_source.cpp` | LPT del dispatch: `order[4]`/`cnt[4]` → `[8]`, inizializzati a 8. |
| `src/prefill/prefill.cpp` | `Prefill::Impl::mw[4]` → `[8]`; `set_maxwell` clampa a 8; commento FASTCOMPLETE aggiornato. |
| `include/strata/core/maxwell_experts.hpp` | Commenti: “up to 4” → “up to 8”. |

**Cosa NON è stato toccato, e perché.**

- `src/core/maxwell_meta.cpp` (`std::array<MaxwellExperts, 4> rank`), `include/strata/core/verify.hpp`
  (`h_mwf_[4]`, `set_mw_wait`), `include/strata/core/maxwell_tp_plan.hpp`: sono il percorso
  **tensor**/E1, 4-way per costruzione. Il guard in `generate.cpp` impedisce di armare E1 con più di
  4 device.
- Il kernel `src/kernels/cuda/expert_maxwell.cu`: è già per-die, nessuna modifica necessaria.
- I default del percorso IQ2_XS: nessuno è cambiato.

**Nota sullo stato del server al momento dell'esecuzione (2026-11-07).** Il checkout di
produzione non era quello dato per scontato dal brief: `/root/Strata-maxwell` è sul branch `mw8`
(motore 0.1.47, commit locale `892d8df0`) con remote `nuke1984`, e la produzione **girava già con
8 die** su `engine-mw8`. Il `main` del fork (`6175fa0`, 0.1.51) è un ramo diverso: 597 commit di
distanza, base comune `011c9e60`. Per questo la baseline 0a è stata eseguita in un worktree
separato e confrontata A/B con il motore in produzione, invece di sostituirlo.

**Nota sulla patch `mw8`.** Il file `0001-mw8-up-to-8-Maxwell-devices-arrays-prefill-tier-cap-.patch`
non era presente nel checkout locale. Le modifiche sono state riapplicate a mano seguendo la
descrizione della patch, adattandole a `main` 6175fa0. L'unico hunk **omesso** è il warning +
`unsetenv("STRATA_MW_DRED")` nel prefill: su `main` tutti i riferimenti a `STRATA_MW_DRED` sono stati
rimossi (commento in `src/core/maxwell_experts.cpp`: “All references removed”; verificato con `grep`),
quindi sarebbe codice morto. Se in futuro DRED tornasse, il guard va reintrodotto.

**Percorso IQ2_XS: invariato? Come lo garantisco.**

- Per **≤4 device** array e limiti più grandi non cambiano nessun indice o ordine: tutti i loop sono
  già su `maxwell_count`/`mw_count`. Il parser e il guard E1 cambiano comportamento solo sopra 4.
- Nessun default è cambiato. La modalità `tensor` è invariata.
- L'unica differenza è il layout di `ExpertDispatch` (array più grandi): serve una **ricompilazione
  completa**, non un rimpasto di oggetti compilati con header diversi.
- Con **8 device** il percorso per dispositivo è lo stesso di prima; cambia solo quanti tier
  ricevono gli esperti (round-robin su 8 invece di 4). La qualità si giudica con il round-trip gate,
  non confrontando risposte a temperatura 0 (già non bit-identiche prima di questo lavoro).

**Comandi per aidrogen.** Eseguire in ordine. Il servizio di produzione va fermato; i test girano in
`tmux` con il motore nuovo, poi la produzione si riavvia.

```sh
# --- 0. ferma la produzione ---
systemctl stop strata

# --- 1. aggiorna il branch ---
cd /root/Strata-maxwell
git status --short          # atteso: solo file non tracciati (nessuna modifica locale)
git branch --show-current   # atteso: mw8
git fetch https://github.com/BitMakerMan/Strata-maxwell iq3
git checkout -B iq3 FETCH_HEAD
git log --oneline -2          # atteso: 0283a48 maxwell: up to 8 devices..., 6175fa0 ...

# --- 2. compila (build-iq3, come da brief §6) ---
export PATH=/opt/cmake-venv/bin:$PATH   # obbligatorio: il cmake 3.25.1 di sistema non supporta il dialetto CUDA20 con nvcc 12.9
cmake -S . -B build-iq3 -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.9/bin/nvcc -DCMAKE_CUDA_ARCHITECTURES=86 \
  -DSTRATA_ENABLE_CUDA=ON -DSTRATA_EXPERIMENTAL_SM60=ON \
  -DSTRATA_GGML_DIR=/root/Strata-maxwell/third_party/llama.cpp \
  -DSTRATA_NATIVE_EXPERTS=ON -DSTRATA_Q6K_EXPERTS=ON -DSTRATA_MMQ_KQUANTS=ON \
  -DSTRATA_MMQ_TQ_OVERLAY=ON -DGGML_STATIC=ON
cmake --build build-iq3 -j 8 --target strata
mkdir -p engine-iq3 && cp build-iq3/strata build-iq3/libstrata_kq.so engine-iq3/

# --- 3. config di test per IQ2_XS (resta sul server: contiene api_key, NON va committato) ---
CFG_IN=/root/Strata-maxwell/strata-iq2_xs-8mw-hdd.json    # produzione attuale
/root/Strata-maxwell/.venv/bin/python - "$CFG_IN" <<'PY'
import json, os, sys
src, dst = sys.argv[1], "/root/Strata-maxwell/strata-iq2-xs-baseline.json"
c = json.load(open(src))
c["exe"] = "/root/Strata-maxwell/engine-iq3/strata"
c["log"] = "strata-iq2-xs-baseline.log"
fd = os.open(dst, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
with os.fdopen(fd, "w") as f:
    json.dump(c, f, indent=2)
print("wrote", dst)
print("exe:", c["exe"])
print("maxwell:", [a for a in c["args"] if a.startswith("--maxwell")])
print("env:", c["env"])
PY
# Atteso: exe /root/Strata-maxwell/engine-iq3/strata, --maxwell-devices 1,2,3,4,5,6,7,8,
# e in c["env"] CUDA_DEVICE_ORDER/CUDA_VISIBLE_DEVICES, STRATA_EMBED_VRAM e i sei STRATA_MW_*.

# --- 4. avvia il motore nuovo in tmux (NON usare il servizio systemd) ---
tmux new -s iq3-baseline
cd /root/Strata-maxwell
# Nessun export: il config ha già "env" con CUDA_DEVICE_ORDER/CUDA_VISIBLE_DEVICES,
# STRATA_EMBED_VRAM e i sei STRATA_MW_*; il server li applica al motore.
/root/Strata-maxwell/.venv/bin/python serve/server.py --engine strata \
  --config strata-iq2-xs-baseline.json --port 8080
```

Da un'altra shell, attendere il caricamento (~13 minuti: IQ2_XS shard 1 e pack sono sull'HDD) e
verificare il tier:

```sh
curl -sf http://127.0.0.1:8080/health && echo " health OK"
grep -E "Maxwell GPU|round-trip|expert cache|Maxwell tier" \
  /root/Strata-maxwell/strata-iq2-xs-baseline.log | tail -40
```

Misure (stesso protocollo del 5–6 novembre):

```sh
# prefill: 3 prompt da ~2220 token (script sul server)
python3 /root/prefill_test.py

# decode: richiesta PI agent identica a quella del 5–6 novembre (1801 token generati).
# Annotare i tok/s riportati dal client.
```

Salvare gli output (senza chiavi) in `docs/iq3/results/00-baseline.txt`, poi chiudere:

```sh
# Ctrl-C nel tmux, poi riavvia la produzione
systemctl start strata
```

**Risultato atteso / criterio di successo.**

- Nel log: **8** righe `Maxwell GPU k: ~2838 experts, ~3.81 GiB`; **8** `round-trip ... PASSED`;
  `strata prefill: 8 Maxwell tier(s)`.
- 3060: ~1874 slot di cache, 24576 esperti totali (cache + 8 tier), ~2246 MiB liberi.
- Decode nel range 65–68,7 tok/s e prefill ~181 tok/s (nuova baseline: il numero che esce, se i gate
  sono verdi, è la reference per le iterazioni successive). Se un gate fallisce o i tier sono meno
  di 8, **fermarsi** e riportare il log: non misurare la velocità.
- AER: annotare il contatore `TOTAL_ERR_COR` su `0000:09:11.0` **prima e dopo** il test (nessuna
  indagine in questa iterazione).

**Risultato misurato.** (2026-11-07, aidrogen; output completo in `docs/iq3/results/00-baseline.txt`)

```
data:                         2026-11-07
8 Maxwell tier(s)?            SI
gate PASSED su tutti gli 8?   SI (8/8, worst rel 1.12e-02 .. 1.26e-02)
esperti e GiB per die:        2838 (CUDA1-6), 2837 (CUDA7-8); 3.80-3.82 GiB
slot 3060 / MiB liberi:       1874 slot / 2246 MiB
prefill (tok/s):              173.8 / 178.6 / 178.7   (media 177.0)
decode (tok/s):               dipende dal prompt, vedi sotto
AER:                          0 prima dei test, 2 dopo; entrambi sul PLX 8747 (09:11.0), 0 sulle GPU
```

**A/B a parità di tutto (l'unica misura di decode valida).** Stesso config, stessi
prompt, stessa macchina, stesso stato termico: cambia solo l'eseguibile.

| prompt | 0.1.47 (`engine-mw8`) | 0.1.51 (`build-iq3`) |
|---|---|---|
| prefill 2220 tok × 3 | 173.8 / 181.2 / 181.1 | 173.8 / 178.6 / 178.7 |
| 600 tok, "200 modi per un tramonto" | 48.0 / 49.0 | 47.3 / 48.5 |
| 1801 tok (racconto) | 53.5 + 42.6 (1784 tok in 35.7 s ≈ 50.0) | 54.6 + 44.5 (1783 tok in 34.7 s ≈ 51.4) |

→ **0.1.51 non regredisce**: è pari o leggermente sopra 0.1.47 su ogni prova. Il criterio 5
del brief (IQ2_XS invariato) è soddisfatto.

**Perché il decode non è 65-68,7 e perché non è una regressione.** Il numero assoluto dipende
dall'accettazione delle draft, quindi dal testo generato. Nel run di riferimento del 6 novembre
(1801 tok, 68.6 tok/s) le draft accettate erano **1273 su 1687 (75,5%)**; con i prompt di oggi
sono **727 su 1007 (72,2%)**. La differenza di accettazione vale **~9%**: con `--spec 6` una
finestra costa ~3,5 token accettati al 75,5% e ~3,2 al 72,2%, non il 25-35% che separa 68,6 da
~50 tok/s. Il resto della distanza **non è spiegato dal tasso di accettazione** e va cercato
altrove.

**Ipotesi (da verificare).** Il tipo di testo guida sia l'accettazione delle draft sia il routing
degli expert: il 5 novembre, con lo stesso motore, abbiamo misurato **città ~67, codice ~65,
prosa ~51,5 tok/s**. I prompt di oggi (elenco di modi per descrivere un tramonto, racconto
libero) sono più vicini alla prosa che alla città, e i run brevi da 300-600 token danno 45-49
tok/s **anche sul motore 0.1.47**. Questa è un'ipotesi, non una misura: per confermarla serve lo
stesso banco fisso su città/codice/prosa, rimandato all'Iterazione 5 (A/B IQ2_XS vs IQ3_S nella
stessa sessione). Il valore di riferimento per le iterazioni successive è la colonna 0.1.51 di
questa tabella, a parità di prompt.

**Nota di metodo sulla build.** Il checkout di produzione (`/root/Strata-maxwell`, branch `mw8`,
motore 0.1.47) è stato lasciato intatto: la build è stata fatta in un worktree separato
`/root/Strata-iq3` (detached su `8b15183`), con `third_party/llama.cpp` in symlink alla copia
già presente. La `llama.cpp` sul server corrisponde esattamente al pin `3cf03257` richiesto da
0.1.51 (hash della regione `mmq.cuh` identico). Build pulita in ~10 min, 18 warning (tutti
preesistenti: `-Wrestrict`, `-Wunused-variable`, `-Wsign-compare`), 0 errori.

→ Ipotesi: **CONFERMATA** sulla capacità (8 tier, 8 gate, tempi di caricamento, prefill nel
range). **SMENTITA** nella forma "decode 65-68,7 entro il rumore": il decode è ~47-55 tok/s con
i prompt usati, su **entrambi** i motori. La causa è l'accettazione delle draft, non il codice.

**AER — osservazione da seguire (non una spiegazione).** Il 5 novembre, a Gen1, il contatore AER
era **0**. Il 7 novembre, durante i test di questa iterazione, sono comparsi **due** eventi
`Correctable error message received` e il contatore è poi salito a **3**. Sono tutti su
`0000:09:11.0`, che da `lspci -tv` è una **porta downstream del PLX 8747**; il figlio è
`0000:0d:00.0` = **M10 indice 8 (CUDA8)**. Il registro interessato è `RxErr`
(`TOTAL_ERR_COR 3`, fatal 0, nonfatal 0, tutte le altre classi 0) e **tutte le GPU hanno
contatori AER a 0**. Non è quindi "coerente con il workaround Gen1": è un cambiamento rispetto
al 5 novembre e va tenuto sotto osservazione. Da questa iterazione in poi si registra il
contatore **prima e dopo ogni test** (comando in `docs/iq3/results/00-baseline.txt`); nessuna
indagine oltre a questo, salvo peggioramento.

---

## Iterazione 0b — `STRATA_MW_FASTCOMPLETE=1` (esperimento separato)

**Obiettivo.** Dopo che la baseline 0a è chiusa, misurare l'unica variabile
`STRATA_MW_FASTCOMPLETE=1` sullo stesso motore e con lo stesso protocollo.

**Ipotesi.** Il fork riporta +2,8% di decode con i quattro complete concorrenti; su 8 die il guadagno
può cambiare. I gate di qualità e il round-trip devono restare identici: le righe delle carte sono
disgiunte e i thread si uniscono prima dell'upload.

**Modifiche.** Nessuna al codice: solo una chiave in più nel dizionario `"env"` della copia del
config.

**Comandi per aidrogen.** Creare una copia del config 0a con `STRATA_MW_FASTCOMPLETE=1` nel suo
`"env"`, con nome file e log distinti, poi lanciarla sulla stessa porta 8080:

```sh
/root/Strata-maxwell/.venv/bin/python - <<'PY'
import json, os
src = "/root/Strata-maxwell/strata-iq2-xs-baseline.json"
dst = "/root/Strata-maxwell/strata-iq2-xs-baseline-fastc.json"
c = json.load(open(src))
c["env"]["STRATA_MW_FASTCOMPLETE"] = "1"
c["log"] = "strata-iq2-xs-baseline-fastc.log"
fd = os.open(dst, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
with os.fdopen(fd, "w") as f:
    json.dump(c, f, indent=2)
print("wrote", dst)
print("env:", c["env"])
PY
cd /root/Strata-maxwell
/root/Strata-maxwell/.venv/bin/python serve/server.py --engine strata \
  --config strata-iq2-xs-baseline-fastc.json --port 8080
```

Stesse misure, stesso protocollo, poi salvare in `docs/iq3/results/00b-fastcomplete.txt`.

**Risultato atteso / criterio di successo.** Decode ≥ baseline 0a; prefill e gate invariati. Se il
decode peggiora o un gate cambia, la variabile non entra nella produzione.

**Risultato misurato.** *(da compilare)* → Ipotesi: **CONFERMATA / SMENTITA**.

---

## Iterazione 1 — censimento dei formati

**Obiettivo.** Stabilire, leggendo i metadati dei due pack, quali combinazioni (gate/up, down)
esistono davvero, per sapere esattamente cosa deve coprire il tier Maxwell. Nessun tool nuovo: solo
i due `native_experts.txt` letti a mano.

**Risultato.** In `docs/iq3/results/01-census.txt`:

| pack | combinazioni (gu/down) | n layer |
|---|---|---|
| IQ3_S | 18/20 (14), 18/42 (3), 21/20 (9), 21/42 (1), 22/20 (15), 22/42 (5), 23/20 (1) | 48 |
| IQ2_XS | 16/42 (11), 22/42 (34), 29/42 (3) | 48 |

**Conseguenze.** Il down è **20 (IQ4_NL)** in 39 layer su 48 dell'IQ3_S: aggiungere i gate/up non
basta a caricare il pack. I nove layer con down 42 sono l'unica verifica su pesi reali possibile
prima del down nativo. IQ2_XS non ha nessun layer 20: il down IQ4_NL è esclusivamente dell'IQ3_S.

---

## Iterazione 2 — gate/up IQ3_XXS / IQ3_S / IQ4_XS

**Obiettivo.** Una sola variabile: aggiungere al tier i tre formati gate/up del pack IQ3_S
(`18 IQ3_XXS`, `21 IQ3_S`, `23 IQ4_XS`) al percorso i-quant esistente, lasciando intatto sia il
percorso di produzione IQ2_XS sia il guard che rifiuta il down non-Q2_0.

**Ipotesi.** Il percorso i-quant è parametrico su `IqFmt<TY>` (una `dot` per formato, `qk`/`ipb`/
`step`, `grid_bytes`); aggiungere tre specializzazioni e i loro tre `vec_dot` trascritti da
`iq_kernels.cu` non tocca il codice dei formati esistenti, quindi:
(1) la parity dei tre formati nuovi deve passare con lo stesso gate 1e-5 degli esistenti;
(2) il SASS delle 66 istanze 22/16/29 deve restare identico;
(3) il pack IQ3_S **non** deve caricare (39 layer su 48 chiedono down 20).

**Modifiche.** Cinque file di codice, +343/−32.

| File | Modifica |
|---|---|
| `src/kernels/cuda/expert_maxwell.cu` | `get_int_from_table_16`; `vec_dot_iq3_xxs_q8_1`, `vec_dot_iq3_s_q8_1`, `vec_dot_iq4_xs_q8_1` (griglia/tabella passata come puntatore shared, `dp4a_s32`, `h2f_dev`/`low2f_dev`); `IqFmt<18>` 1 KB, `<21>` 2 KB, `<23>` 16 B (`step 4`); `IqV2qSmem::grid` derivato da `IqFmt<TY>::grid_bytes`; staging byte-wise quando `grid_bytes <= 16`; `IqMt<18>/<21>/<23>`; casi nei tre switch (batch, MT, persistente); `__launch_bounds__(THREADS, 5)` sul persistente. |
| `src/core/maxwell_experts.cpp` | `seen[7]` + `switch` per la mappa `ti` (42/22/16/29/18/21/23); il gate round-trip copre i sei tipi; arming di `STRATA_MW_PERSIST` esteso; commenti. **Guard alla riga ~173 invariato.** |
| `src/kernels/cuda/expert_maxwell_iq_parity.cpp` | `--synthetic` e la modalità shard accettano 18/21/23; commento d'intestazione con i layer reali. |
| `include/strata/kernels/cuda/expert_maxwell.hpp` | contratto `gu_type` con i sei tipi. |
| `src/kernels/cuda/iq_gup_bench.cpp` | whitelist e nomi dei sei tipi. |

**Trascrizione dei tre `vec_dot`.** Copiati riga per riga da `src/kernels/cuda/iq_kernels.cu`
(la stessa fonte che il tier 3060 usa in produzione), con tre sole sostituzioni meccaniche:
`ggml_cuda_dp4a` → `dp4a_s32` (PTX `vmad` su Maxwell), `__half2float` → `h2f_dev`, `__low2float` →
`low2f_dev`, e la griglia globale (`iq3xxs_grid`, `iq3s_grid`, `kvalues_iq4nl`) letta dalla copia
in shared memory. Le semantiche delicate sono quelle del riferimento e sono verificate dalla parity:
`iq3_xxs` usa `bq8_1[iqs/2]` e `(ls*sumi + sumi/2)/2`; `iq3_s` usa `bq8_1[iqs/2]`,
`scales[iqs/4] >> ((iqs<<1)&0x04)`, `qh[iqs/2]`; `iq4_xs` usa `bq8_1[iqs/4]`, `scales_l[iqs/8]`,
`scales_h >> (iqs/2)`, `ls - 32`, `step 4`.

**IQ4_XS non ha codebook.** `kvalues_iq4nl` è una tabella `int8 × 16`, non una griglia a 32 bit:
`grid_bytes = 16` e lo staging usa una copia byte-wise (`if constexpr (F::grid_bytes <= 16)`), perché
l'array sorgente non dichiara l'allineamento a 4 byte che il percorso `int4`/`int32` richiederebbe.

**Risultato misurato.** Output completo in `docs/iq3/results/02-gup-iq3-formats.txt`.

```
parity sintetica iq3_xxs/q2_0:  (1) rel 1.77e-07   (2) faq 0/1920, fasc 5.91e-07   PASSED
parity sintetica iq3_s/q2_0:    (1) rel 1.76e-07   (2) faq 0/1920, fasc 5.91e-07   PASSED
parity sintetica iq4_xs/q2_0:   (1) rel 1.68e-07   (2) faq 0/1920, fasc 4.35e-07   PASSED
parity sintetica iq2_s/q2_0:    (1) rel 1.77e-07   (2) faq 0/1920, fasc 6.34e-07   PASSED  (controllo)
pesi reali, 9 layer IQ3_S con down 42 (5× 22/42, 3× 18/42, 1× 21/42): 0 failures
controllo IQ2_XS, layer 0/1/8 (22/42, 16/42, 29/42): 0 failures
```

**La prova più forte: il SASS non cambia.** Confronto del modulo sm_50 compilato da `8b15183e` e
dalla build modificata, istruzione per istruzione, per ogni istanziazione dei kernel del tier:

```
istanze condivise:                     66 su 66 bit-identiche
istanze perse:                          0
istanze nuove:                         66, tutte con TY in {18, 21, 23}
tipo 22 (IQ2_S):   22 istanze, tutte identiche
tipo 16 (IQ2_XXS): 22 istanze, tutte identiche
tipo 29 (IQ1_M):   22 istanze, tutte identiche
```

Questo è più forte della parity: la parity verifica i valori, il SASS verifica che il codice che li
produce sia lo stesso di prima. Nessuna delle 66 istanze di produzione è stata toccata.

**Un effetto collaterale trovato e corretto.** Il solo kernel non coperto da quella garanzia è
`maxwell_persist_kernel`, il cui `switch` è cambiato. Aggiungendo tre formati è passato da
**95 a 120 registri**, cioè da 5 a 4 blocchi residenti per SM (−20% di occupazione), senza spill:

| versione | registri | blocchi/SM |
|---|---|---|
| HEAD | 95 | 5 |
| dopo l'aggiunta dei tre formati | 120 | 4 |
| con `__launch_bounds__(THREADS, 5)` | 96 | 5 |

Il commento del kernel avvertiva già di questo rischio (era stato scritto proprio per un caso come
questo). La produzione **non arma** `STRATA_MW_PERSIST` — le chiavi `env` del config di produzione
sono `CUDA_DEVICE_ORDER`, `CUDA_VISIBLE_DEVICES`, `STRATA_EMBED_VRAM`, `STRATA_MW_AD2H`,
`STRATA_MW_MT`, `STRATA_MW_MT_T`, `STRATA_MW_NTM`, `STRATA_MW_TWOPHASE` — quindi non era una
regressione sul percorso caldo, ma il fix è applicato perché restava una regressione latente.

**Shared memory.** I tre formati nuovi sono i più piccoli del tier: 18 → 6912 B, 21 → 7936 B,
23 → 5904 B (contro 14080 B di 22 e 29). L'unione del persistente resta `max(22, 29)` con uno
`static_assert` che copre anche 16/18/21/23, così un formato futuro non può sfondare il buffer.

**AER.** `0000:09:11.0` (porta downstream PLX 8747, figlio `0d:00.0` = M10 #8): `TOTAL_ERR_COR 3`
prima e 3 dopo i test. Tutte le altre GPU a 0. Nessun evento nuovo.

→ Ipotesi: **CONFERMATA** su tutti e tre i punti. In particolare il punto (3): il pack IQ3_S **non
carica**, come previsto, perché 39 layer su 48 chiedono ancora il down 20. Il guard resta chiuso.

**Cosa questa iterazione NON dimostra.** Che IQ3_S giri. Il tier non può ancora aprire il pack, quindi
non esiste nessuna misura di decode su IQ3_S; le uniche misure di decode restano quelle IQ2_XS
dell'Iterazione 0. 23 (IQ4_XS) non ha nessun layer reale con down 42 nel pack (il layer 47 è 23/20),
quindi riposa sui pesi sintetici finché non esiste il down nativo.

---

## Iterazione 3 — design: down IQ4_NL nativo (rivisto dopo la revisione dell'operatore)

**Stato: DESIGN, nessun codice scritto.** Revisione 2: sei correzioni dell'operatore applicate
(memoria, termine `corr`, scala delle attivazioni, stride, punti di chiamata, round-trip gate).
Nessuna modifica è stata fatta oltre all'Iterazione 2.

### Perché serve

Il censimento dice: 39 layer su 48 dell'IQ3_S hanno `down = 20` (IQ4_NL, ggml type 20). Oggi
`MaxwellExperts::open` rifiuta ogni layer con `d_type != 42`, quindi il pack IQ3_S non si apre.
L'Iterazione 3 è la variabile che sblocca il caricamento: **un solo formato**, il down IQ4_NL, con
un kernel **separato** accanto al down Q2_0 planare. Il kernel S2 non cambia forma: resta il
percorso Q2_0 che la produzione IQ2_XS usa.

### Il formato e la sua geometria

`block_iq4_nl` (ggml-common.h): `QK4_NL = 32`, `ggml_half d` + `uint8_t qs[16]` = **18 byte** per
32 pesi. Il down ha `H = 2560` righe, ognuna con `FF = 640` pesi = **20 blocchi** = 360 byte/riga,
quindi 921.600 byte per esperto.

Il decode: `valore = d * kvalues_iq4nl[nibble]`, con `kvalues_iq4nl` tabella `int8 × 16` (la stessa
che l'Iterazione 2 già stage in shared per il gate/up 23). A differenza del Q2_0 non c'è
`expand4_s2` né offset dei codici: è un **dot con tabella**, come `vec_dot_iq4_nl_q8_1` di
`iq_kernels.cu`.

### Memoria: nessuna crescita

**Correzione rispetto alla revisione 1 (che era sbagliata).** I blob del pack contengono **già** il
down IQ4_NL nella sua forma raw: il formato 20 è il formato sorgente, non una conversione che
aggiunge byte. La verifica è aritmetica e chiude al byte con il censimento:

```
18/20 = 2·1.638.400·98/256 (gate+up 1.254.400) + 1.638.400·18/32 (down 921.600) = 2.176.000
22/20 = 2·1.638.400·82/256 (gate+up 1.049.600) +                        921.600  = 1.971.200
```

Somma sui 48 layer × 512 esperti = **50.292.326.400 B = 46,84 GiB**, che è esattamente la dimensione
dichiarata di `experts.bin`. Non esiste nessun `+1,22 GiB/die`: quel numero veniva dall'aver
confrontato il blob IQ4_NL con un blob Q2_0 che nel pack IQ3_S non esiste.

**Stima (ipotesi, da misurare al load).** 46,84 GiB su 8 die = **5,86 GiB/die** a ripartizione
perfettamente uniforme; con la stessa cardinalità di oggi (2838 esperti/die) sono **5,41 GiB/die**.
L'M10 ha 8 GiB, quindi il pack entra, ma con meno margine dei 3,80–3,82 GiB/die dell'IQ2_XS: il
numero vero dipende da quanti esperti il profilo terra' residenti, e va letto dal log di load. Non è
un rischio di correttezza, è un rischio di **quanti esperti restano in cache**.

### Le due strade possibili, e perché se ne sceglie una

**(a) Riusare il percorso planare S2** convertendo le righe IQ4_NL in "16 code + 1 scala" come fa
oggi il Q2_0. Il `slut[256]` del Q2_0 esiste per decodificare 4 codici a 2 bit in 4 byte; qui
servirebbe invece un lookup a 16 voci su nibble. Il costo è che il kernel S2 dovrebbe cambiare
forma (la sua `block_pair_s2_*` è cablata su `expand4_s2`), cioè si toccherebbe il codice che la
produzione IQ2_XS usa per il down — esattamente ciò che il brief vieta.

**(b) Un kernel down IQ4_NL separato, accanto all'S2, selezionato dal tipo.** Il codice S2 resta
byte-identico; il nuovo kernel ha la stessa geometria (una riga per `SPLIT`=2 gruppo di lane, una
passata su 20 blocchi) ma il proprio corpo. **Scelta: (b).** È l'unica che tiene il vincolo
"il percorso di produzione IQ2_XS resta invariato" verificabile a livello di SASS, come in
Iterazione 2.

### Layout del blob

Il blob resta "gate/up verbatim + down raw", con la stessa struttura di offset di oggi:

```
[ o_gate = 0                    ] gate: 640 righe × gu_row, blocchi GGUF grezzi
[ o_up = up_off                 ] up:   640 righe × gu_row, blocchi GGUF grezzi
[ d_codes_off = down_off        ] down: 2560 righe × 20 blocchi × {d(fp16) | qs(16 B)}
[ d_scales_off                  ] solo Q2_0: la produzione non lo usa per IQ4_NL
```

Per IQ4_NL **non c'è de-interleaving**: `block_iq4_nl` è già contiguo a 18 byte e il kernel lo legge
così com'è. Quindi la conversione non fa lavoro per blocco: è una `memcpy` della regione down, e
`d_codes_off = f.down_off`, `d_row_stride = 360 B` (20 blocchi × 18), `d_scale_stride` non usato.

**Lo stride per layer esiste già.** `maxwell_experts.cpp:219` usa `lay.blob_bytes(layer)` per il
budget e il commento a ~267 spiega perché: i blob di un pack nativo differiscono per layer, quindi
un `slot*BLOB` piatto non li indirizza. `LayerGeo::blob_bytes` porta già il valore (`f.bytes`, che è
`down_off + d_row*H`), e ogni launch lo passa già come `blob_stride`. **Non serve toccare il
percorso di indirizzamento**: serve solo impostare, per il formato 20, `d_codes_off = f.down_off`,
`d_row_stride = 20*18`, e lasciare `d_scales_off`/`d_scale_stride` ai loro default (che il kernel
IQ4_NL ignora). La revisione 1 diceva che questo era "il punto di rischio numero due"; non lo è.

### Kernel down: la forma

Nome proposto `iq4nl_expert_maxwell_down_batch_async`, simmetrico a `s2_expert_maxwell_down_batch_async`:

- **Geometria invariata**: `SPLIT = 2`, `THREADS = 128`, `PER_BLOCK = 64` → 40 blocchi per una riga
  di `H = 2560`, come l'S2. La riduzione finale (`sh[e][threadIdx.x]`, somma sequenziale `s = 0..1`)
  è **la stessa**, così il contratto di arrotondamento verso il chiamante non cambia.
- **Ingresso**: `faq` + `fasc` prodotti dal gup (già bit-identici, verificato in Iterazione 2).
  **`fahx` è ignorato** (vedi sotto).
- **Per ogni (riga, blocco)**: i 16 code byte si leggono **solo** con `get_int_b2(qs, i)`, mai con
  una lettura `int4`/`int32` diretta: `block_iq4_nl` è 18 byte (fp16 + `qs[16]`), quindi `qs` è
  allineato a **2 byte**, e un load vettoriale su un indirizzo dispari sarebbe un accesso
  disallineato (fault o lettura sbagliata). È la stessa ragione per cui la 3060 usa
  `get_int_b2(bq4->qs, ...)` in `vec_dot_iq4_nl_q8_1` e in `Split<20>::load` — e la stessa
  convenzione vale per la variante MT (che quindi **non** può fare l'hoist `int4` dell'S2).
  Poi `get_int_from_table_16(q4, kvalues_sh)` e `dp4a_s32` contro i word della riga di `faq` —
  esattamente `vec_dot_iq4_nl_q8_1`. La mappatura del riferimento è `kbx = k/ipb`,
  `iqs = step*(k%ipb)` con `qk=32, ipb=2, step=2` (`iq_kernels.cu`, `Fmt<20>`): `nb = 640/32 = 20`
  blocchi, e ogni blocco fa **due** chiamate con `iqs = 0` e `iqs = 2` (16 valori ciascuna), contro
  un solo `block_q8_1` a `kbx`. In concreto le due chiamate leggono `get_int_b2(qs, 0)`/`(qs, 1)`
  (prima metà, word `0..3` di `faq`) e `get_int_b2(qs, 2)`/`(qs, 3)` (seconda metà, word `4..7`).
- **Epilogo**: `out[row] = Σ_b d_b · ssc[chunk] · dot_b`, **senza** termine `corr`, con
  `ssc[chunk] = low2f_dev(f2h_rn(fasc[chunk]))` (punto 3).
- **`kvalues_iq4nl` in shared**: 16 byte, staged una volta per blocco, come il gate/up 23.
- **Multi-token**: la variante MT serve, perché la produzione ha `STRATA_MW_MT=1` e `TWOPHASE=1`
  (punto 5). Forma identica a `expert_s2_stage2_mt_batch`: un RUN per blocco, `faq` di ogni entry
  staged in shared, i code byte caricati una volta sola e riusati su E entry. L'unica differenza
  rispetto all'S2 è il load: qui i 16 code byte arrivano con **quattro** `get_int_b2(qs, 0..3)`
  (2 byte per volta) invece di un `int4`, e vengono espansi con `get_int_from_table_16` in `W.v[0]`
  (prima metà) e `W.v[1]` (seconda), poi applicati con la stessa sequenza `apply` del riferimento.

**Perché il load non è un `int4` (verificato).** `struct block_iq4_nl { ggml_half d; uint8_t qs[16]; }`
ha `sizeof = 18` e nessun `alignas`, quindi `qs = base + 2`: un `*(const int4*) qs` non è
garantito allineato a 16 byte. `vec_dot_iq4_nl_q8_1` evita il problema con `get_int_b2`, e
`Split<20>::load` fa lo stesso; il kernel del tier deve fare altrettanto. La versione MT dell'S2
può permettersi `const int4 c4 = *(const int4*) (dc + lb * 16)` perché la sua regione down è
de-interleaved **a 16 byte per blocco** (planare: solo i codici, solo per Q2_0). Il blob IQ4_NL
invece è raw (18 byte per blocco), quindi la regola cambia: quattro `get_int_b2`, o equivalentemente
`memcpy` di 16 byte — mai un cast vettoriale.

### Niente `corr`/`hx` (punto 2 della revisione)

Nell'S2 il termine `corr = d·Σ hx` **compensa l'offset dei codici Q2_0**: il Q2_0 decodifica
`codice - 1` (codici 2 bit in `{-1,0,1,2}` ridotti a `-1..2`), e la quantizzazione delle attivazioni
ha un termine affine `hx` per chunk che va sottratto. L'IQ4_NL **non ha offset**: i suoi 16 valori
`kvalues_iq4nl` sono già con segno e centrati su zero.

Il riferimento della 3060 lo conferma: `vec_dot_iq4_nl_q8_1` (in `iq_kernels.cu`) chiude con

```c
const float d = __half2float(bq4->d) * __low2float(bq8_1->ds);   // solo d, nessun s
return d * sumi;
```

cioè usa **solo** `ds` (il `d` della scala) e mai il secondo half `s`. Coerentemente, `Split<20>` del
percorso MMVQ fa `r.dw * __low2float(bq8_1->ds)` e nient'altro.

Quindi il kernel IQ4_NL consuma `faq` e `fasc` e **ignora `fahx`**. Un `fahx` letto e sottratto
come nell'S2 sarebbe un errore sistematico, non una differenza di arrotondamento.

**La revisione 1 diceva il contrario** ("ricalcato identico, o il round-trip gate fallisce"): era
sbagliato, perché assumeva che il contratto CPU `dot + hx` valesse per ogni formato down. Vale per
il Q2_0 e per i suoi codici con offset, non per IQ4_NL.

### La scala delle attivazioni (punto 3 della revisione)

Il disallineamento è reale e va dichiarato: la 3060 quantizza `FF` in `block_q8_1` con `d` in
**fp16**, mentre sulle M10 `fasc` è **float**. Il down S2 di oggi non ha una parity contro la 3060 —
`expert_maxwell_parity.cpp` lo confronta con un riferimento CPU che usa `fasc` float — quindi non
c'è un precedente che risolva la questione per il down.

Il gup della M10 invece la risolve già, ed è la politica che il tier ha scelto: in
`iq_stage1_v2q_body` ogni blocco q8_1 viene ricostruito con `ds[0] = f2h_rn(d)` (con il clamp #606),
cioè **arrotondando la scala a fp16** come fa il quantizzatore di dispositivo della 3060. La parity
del gup verifica esattamente questo: `(0)` confronta i blocchi ricostruiti con quelli del
quantizzatore della 3060 e trova 0 differenze su 240.

**Scelta: (a) arrotondare la scala a fp16**, con `f2h_rn` + `low2f_dev` nell'epilogo del kernel
IQ4_NL. Motivi:

1. è la stessa politica già in produzione e già verificata per il gup — un formato sola;
2. rende il gate della parity **onesto**: il riferimento 3060 (`iq_mmvq` tipo 20) legge `ds` fp16,
   quindi con (a) il confronto è mela-contro-mela e il gate può restare a 1e-5;
3. con (b) (tenere float) il kernel sarebbe *più preciso* del riferimento, e la differenza fp16-vs-float
   entrerebbe nel numero: un gate a 1e-5 fallirebbe per un motivo legittimo, e abbassarlo
   nasconderebbe errori veri.

Il costo di (a) è una conversione fp32→fp16 per chunk (20 per riga, 2560 righe): irrilevante, e la
stessa che il gup paga già. Il gate dichiarato: `rel <= 1e-5` contro `iq_mmvq` tipo 20 sugli stessi
`faq`/`fasc`, come in Iterazione 2. La deviazione residua attesa è zero sul percorso q8_1 e non
misurabile sul percorso F16 (che è una riduzione di precisione a valle, già accettata dal tier).

### Guard e arming

- Il guard di `open()` (~riga 173) passa da `d_type != 42` a `d_type != 42 && d_type != 20`, e il
  messaggio d'errore elenca entrambi. **È l'unica riga che si tocca del guard**, ed è il momento in
  cui il pack IQ3_S può finalmente caricare.
- Il whitelist del tool di parity (`expert_maxwell_iq_parity.cpp`) accetta le combinazioni con
  `<gu>/iq4_nl`; la modalità shard oggi rifiuta `t[2]->type != 42` e va estesa a 20.

### Tutti i punti di chiamata (punto 5 della revisione)

Il down viene lanciato da nove punti. Ognuno va instradato sul tipo del layer, e il ramo Q2_0 resta
byte-identico:

| # | Punto | Riga | Forma | In produzione? |
|---|---|---|---|---|
| 1 | round-trip gate in `open()` | ~639 | batch fp32 | solo al load |
| 2 | catena decode, uscita fp32 | 1211 | batch fp32 | **sì** |
| 3 | catena decode, uscita f16 | 1165 | batch f16 | solo con `STRATA_MW_ROWS_F16` |
| 4 | catena TP, chunk f16 | 1187 | batch f16 | solo con `TP_WIRE16` |
| 5 | catena TP, chunk fp32 | 1193 | batch fp32 | solo tensor split |
| 6 | `prefill_launch_down`, ramo MT | 1325 | **MT** | **sì** (`STRATA_MW_MT=1`) |
| 7 | `prefill_launch_down`, ramo batch | 1330 | batch fp32 | no (solo senza MT) |
| 8 | `finish_to`/`finish_chain`, ramo MT | 1401 | **MT** | **sì** |
| 9 | `finish_to`/`finish_chain`, ramo batch | 1406 | batch fp32 | no (solo senza MT) |

La produzione arma `STRATA_MW_MT=1` e `STRATA_MW_TWOPHASE=1` (chiavi `env` del config: vedi
Iterazione 2), quindi la variante **MT è sul percorso caldo** del prefill, con `STRATA_MW_NTM=256`.
Le tre forme da implementare sono quindi: **batch fp32, batch con uscita f16, MT**. Non è
sufficiente la sola variante per-token.

Un punto d'aggancio pulito: una funzione statica `down_launch(...)` che prende `const LayerGeo&` e
instrada su un nuovo campo `LayerGeo::d_type` (oggi assente: va aggiunto, inizializzato a 42 nel
default e a `f.d_type` nel costruttore di `geo_`) verso il kernel giusto. Tutti e nove i punti
chiamano quella, e il ramo `d_type == 42` chiama le funzioni S2 di oggi con gli stessi argomenti —
così il SASS del percorso di produzione è verificabile come in Iterazione 2.

**Il kernel persistente rifiuta l'arming.** `maxwell_persist_kernel` chiama `s2_stage2_v2_body`
fisso; non esiste una sua variante IQ4_NL (e non serve: la produzione non arma `STRATA_MW_PERSIST`).
L'arming va quindi esteso con la condizione "nessun layer con `d_type == 20`": se un pack misto la
viola, `open()` stampa il motivo e declina il persistente, ricadendo sulla catena normale. È
preferibile a un kernel persistente che legga un layout sbagliato in silenzio.

**E1 non è in produzione.** Il percorso E1 (`STRATA_MW_GPUFLAG`) passa `comp_flag_` al down perché
l'ultimo blocco alzi il doorbell; è la variante `e1` a riga 1211. Il kernel IQ4_NL **non porta la
coda di completion flag** nella prima versione: con `d_type == 20` e `gpuflag_mode() >= 1` il tier
rifiuta l'arming e ricade sulla sync per-layer. Documentato, non silenzioso: un messaggio su stderr
al primo layer 20. (Alternativa scartata: aggiungere la coda al kernel nuovo prima di sapere se
serve — il flag è un contratto con il grafo della 3060 e va progettato, non copiato.)

### Round-trip gate di `open()` (punto 6 della revisione)

Il gate oggi sceglie **un esperto per tipo di gate/up** (`seen[7]`) e dequantizza il down con
`ggml_get_type_traits(d_type)->to_float` — che per IQ4_NL esiste già, quindi la parte IQ4_NL del
gate funziona senza modifiche. Il problema è la granularità: con `seen` indicizzato sul solo gate/up,
un pack che ha insieme `18/20` e `18/42` verifica un solo layer e sceglie quello che incontra prima.

La correzione minima: la chiave di deduplica passa da `gu_type` a **`(gu_type, d_type)`**, con una
mappa a 10 combinazioni osservate nel censimento (7 per IQ3_S, 3 per IQ2_XS). Così ogni combinazione
nuova — **18/20, 21/20, 22/20, 23/20** — viene campionata almeno una volta su un layer reale, che è
il punto 6 della revisione. I layer reali, dal censimento:

```
18/20  0 7 14 16 18 22 31 33 37 38 41 42 43 46
21/20  17 25 27 28 29 35 40 44 45
22/20  2 3 4 5 6 9 11 15 19 23 24 26 32 34 39
23/20  47
22/42  1 8 10 13 36       18/42  12 20 30       21/42  21
```

Il gate mantiene la tolleranza `3e-2` di oggi: misura il rumore di quantizzazione delle attivazioni,
non l'esattezza del kernel (quella la misura la parity a 1e-5). Con l'IQ4_NL il riferimento fp64 usa
`to_float` sul down raw e la stessa `x` dequantizzata, quindi il numero atteso resta nell'ordine di
`8e-3`–`9e-3` come per gli altri formati.

### Non regressione del gate su IQ2_XS (punto 2 della revisione dell'operatore)

La dedup a `(gu_type, d_type)` cambia **quali** esperti il gate campiona. Per l'IQ2_XS questo è un
rischio concreto: le combinazioni sono 16/42, 22/42 e 29/42, ognuna con lo stesso `d_type`, quindi
passando alla chiave a due campi il primo esperto incontrato per ogni combinazione **dovrebbe** essere
lo stesso di oggi (il gate itera `held_` nello stesso ordine e la chiave `d_type` è costante). Non
basta supporlo: il test lo verifica.

**Protocollo.** Nel load IQ2_XS si estraggono le righe

```
MaxwellExperts (CUDAk): round-trip gate layer L expert E (tipo): rel X
```

e si confrontano con quelle di `docs/iq3/results/00-baseline.txt`. Devono essere **identiche**
(layer, expert, tipo) e la `rel` deve restare nello stesso ordine di grandezza. Una differenza nei
layer esperti significa che la dedup ha cambiato il campione e il test va considerato rosso, anche se
il gate è PASSED. Se il confronto non è possibile perché il baseline non contiene le righe per-layer
(contiene solo le righe `PASSED` aggregate), il test si esegue **due volte** — prima con il codice
all'HEAD dell'Iterazione 2, poi con quello nuovo — e si confrontano le due liste.

**Scoperta durante l'implementazione (da correggere in questa iterazione).** Il guard di `open()`
(~riga 172) accetta come gate/up solo `42, 22, 16, 29`: l'Iterazione 2 ha aggiunto i kernel 18/21/23
ma **non** ha esteso il guard, quindi oggi un layer 18/20 verrebbe rifiutato per il gate/up prima
ancora del down. La modifica corretta è quindi:

```c
if ((f.d_type != 42 && f.d_type != 20) ||
    (f.gu_type != 42 && f.gu_type != 22 && f.gu_type != 16 && f.gu_type != 29 &&
     f.gu_type != 18 && f.gu_type != 21 && f.gu_type != 23)) { ... }
```

Il ramo `d_type` è quello previsto dal design (42 → 42 && 20); il ramo `gu_type` è un **bug
dell'Iterazione 2** che questa iterazione deve chiudere, altrimenti il pack IQ3_S non carica
comunque. Il test SASS dell'Iterazione 2 non poteva vederlo: è una condizione host, non un kernel.

### Verifica prevista

1. **Parity sintetica**, con il riferimento `iq_mmvq` tipo 20 della 3060 sugli stessi `faq`/`fasc`:
   `iq3_s/iq4_nl`, `iq3_xxs/iq4_nl`, `iq2_s/iq4_nl`, `iq4_xs/iq4_nl`. Gate `1e-5`, come in
   Iterazione 2. Il controllo `iq3_s/q2_0` deve restare invariato.
2. **Pesi reali**: i 39 layer down-20 diventano leggibili. Copertura minima: un layer per
   combinazione nuova, cioè 18/20, 21/20, 22/20, 23/20 (layer 47 per 23/20, che è anche l'unico
   IQ4_XS reale del pack).
3. **SASS**: le istanze 22/16/29 e il kernel S2 Q2_0 devono restare bit-identiche. Il kernel IQ4_NL
   è nuovo codice, quindi le istanze condivise devono essere le stesse 66 di Iterazione 2 più le
   S2, e le nuove devono essere solo quelle del formato 20.
4. **Memoria**: nessuna crescita (i blob sono già IQ4_NL). Da misurare al load: GiB/die e numero di
   esperti residenti, che con ~5,4–5,9 GiB/die sarà più basso dei 2838 dell'IQ2_XS.

**Esito.** Tutti e quattro i punti verificati; i numeri misurati sono nella sezione seguente, la
misura al load (punto 4) nella «Finestra operativa».

## Iterazione 3 — implementazione e verifiche di kernel (misurato, 2026-11-07)

Stato: **codice completo e verde su parity sintetica, parity su pesi reali e SASS, e la finestra
operativa è stata eseguita: IQ2_XS non regredisce, IQ3_S carica e risponde.** Dettagli nella sezione
«Finestra operativa» in coda a questa voce.

### Cosa è stato scritto

| file | modifica |
|---|---|
| `include/strata/kernels/cuda/expert_maxwell.hpp` | 3 dichiarazioni: `iq4nl_expert_maxwell_down_{batch,batch_f16,mt}_async` |
| `src/kernels/cuda/expert_maxwell.cu` | `kIq4nlBlocks`, `kIq4nlDownSmem`, `iq4nl_load_block`, `iq4nl_dot_block`, `iq4nl_stage_act`, `iq4nl_down_body`, `expert_iq4nl_down_batch`, `expert_iq4nl_down_mt`, 3 lanciatori |
| `include/strata/core/maxwell_experts.hpp` | `LayerGeo::d_type` (default 42) |
| `src/core/maxwell_experts.cpp` | guard esteso a `d_type 20`; rifiuto TP per il tipo 20; geometria raw (blob `memcpy` verbatim); dedup del gate a `(gu_type, d_type)`; dispatcher `mwdown`; nove punti di chiamata; rifiuti persist ed E1 |
| `src/kernels/cuda/expert_maxwell_iq_parity.cpp` | `check_down()` + verifica delle tre forme; whitelist estese al 20 |

### Parity sintetica — `results/03-down-iq4nl-synthetic.txt`

`0 failures`. Gate (4), il kernel nuovo contro `iq_mmvq(type 20)` sugli stessi `faq`/`fasc`:

| gate/up | (4) down rel | (1) gup rel | (4b) MT vs batch | (4b) f16 vs round(batch) |
|---|---|---|---|---|
| `iq3_s` | **9.71e-08** | 1.72e-07 | 0 / 7680 | 0 / 7680 |
| `iq3_xxs` | **9.62e-08** | 1.69e-07 | 0 / 7680 | 0 / 7680 |
| `iq2_s` | **9.27e-08** | 1.75e-07 | 0 / 7680 | 0 / 7680 |
| `iq4_xs` | **1.00e-07** | 1.64e-07 | 0 / 7680 | 0 / 7680 |
| `iq3_s/q2_0` (controllo) | — | 1.91e-07 | — | — |

Il controllo `iq3_s/q2_0` non passa nemmeno dal kernel nuovo (nessun layer 20): la sua riga (1)
resta identica all'Iterazione 2, come previsto.

**Nota su (4b) f16.** Il primo tentativo dava 1 differenza su 7680. Non era un bug del kernel: il
riferimento del tool (`host_f2h`) implementa l'IEEE completo, mentre il contratto di `f2h_rn` in
`expert_maxwell.cu` (riga 45) **azzera i denormali** per scelta (`e <= 0 → sign`), documentata nel
commento della funzione. Il valore in questione è un'uscita sotto 6.1e-5, che `f2h_rn` porta a zero
con segno. Aggiunto `contract_f2h` al tool, che rispecchia quel contratto: 0 differenze. `host_f2h`
resta per i controlli sull'immagine delle attivazioni, dove i bit li decide il quantizzatore del
device, non `f2h_rn`.

### Parity su pesi reali — `results/04-down-iq4nl-real.txt`

`0 failures` su tutti e sette i layer. Il layer reale per ogni combinazione nuova, come richiesto
dalla revisione:

| layer | combinazione | (4) down rel | (1) gup rel | (4b) MT / f16 |
|---|---|---|---|---|
| 0 | 18/20 | 9.53e-08 | 1.66e-07 | 0 / 0 |
| 17 | 21/20 | 9.12e-08 | 1.79e-07 | 0 / 0 |
| 2 | 22/20 | 9.59e-08 | 1.65e-07 | 0 / 0 |
| 47 | 23/20 | 9.46e-08 | 1.82e-07 | 0 / 0 |

Controlli con down 42 (Iterazione 2, invariati): layer 12 (18/42) 1.76e-07, 21 (21/42) 1.73e-07,
1 (22/42) 1.74e-07 sulla riga (1); nessun check (4) perché `d_type != 20`, come da progetto.

### SASS — `results/05-down-iq4nl-sass.txt`

Confronto HEAD (Iterazione 2, `6dcaa8e2`) contro worktree, stessi flag (`-O3 -DNDEBUG -std=c++20`,
`sm_50`), `cuobjdump -sass`, normalizzando solo l'hash del namespace anonimo.

```
kernels: HEAD 150   NEW 153
shared: 150   only in HEAD: 0   only in NEW: 3
identical: 150   differing: 0
```

- **S2 (percorso di produzione IQ2_XS)**: `expert_s2_stage2_mt_batch` 1, `expert_s2_stage2_batch_v2`
  5, `expert_s2_stage1` 7, `expert_s2_stage2` 8 istanze — tutte bit-identiche.
- **gate/up i-quant**: `expert_iq_stage1_batch_v2q` 120, `expert_iq_stage1_mt_batch` 12,
  `maxwell_persist_kernel` 1 — tutte bit-identiche.
- **Nuove**: esattamente 3, tutte `expert_iq4nl_down_{batch<false>,batch<true>,mt}`; nessuna
  istanza condivisa contiene `iq4nl`.
- **Registri** (invariati): persist 96, S2 MT 55. I kernel nuovi: down MT 54, batch fp32 40, batch
  f16 40.

Il conteggio «66 istanze» del design era il numero di famiglie dell'Iterazione 2 (120 + 12 + 1
kernel del gate/up, più le famiglie S2); il confronto per nome completo conferma che **nessuna**
delle 150 condivise è cambiata, che è la proprietà che conta.

### Difetti trovati e corretti durante l'implementazione

1. **`g.tp_blks` = 10 non è il numero di blocchi IQ4_NL.** `SC_D` conta i blocchi Q2_0 da 64 valori;
   IQ4_NL ne ha 20 da 32. Il dispatcher passava `g.tp_blks` (default 10) e avrebbe letto metà riga.
   Corretto con `kDBlocksIq4nl = FF / 32` passato esplicitamente (il tipo 20 non è mai in TP: il
   rifiuto in `open()` lo garantisce).
2. **`LayerGeo` è un tipo annidato privato**: il dispatcher `mwdown` non può nominarlo a livello di
   namespace. Risolto rendendo `batch`/`batch_f16`/`mt` template sul tipo della geometria (dedotto
   dai nove punti di chiamata, tutti dentro la classe).
3. **`contract_f2h`** nel tool di parity (vedi sopra).

### Le tre forme richieste

Tutte implementate e verificate, non solo la per-token:

| forma | funzione | misurata |
|---|---|---|
| batch fp32 | `expert_iq4nl_down_batch<...,false>` | (4) su tutti i layer |
| batch f16 | `expert_iq4nl_down_batch<...,true>` | (4b) 0/7680 su tutti |
| MT | `expert_iq4nl_down_mt` | (4b) 0/7680 su tutti |

### Punti di chiamata instradati

Nove, tutti su `mwdown::{batch,batch_f16,mt}`: gate round-trip in `open()`; catena decode fp32;
catena decode f16; catena TP f16; catena TP fp32; `prefill_launch_down` ramo MT; `prefill_launch_down`
ramo batch; `finish` ramo MT; `finish` ramo batch. Il ramo `d_type == 42` chiama le funzioni S2 con
gli stessi argomenti di prima, ed è ciò che il confronto SASS verifica.

### Rifiuti espliciti (non silenziosi)

- **PERSIST** (`STRATA_MW_PERSIST`): declinato se un layer ha `d_type == 20`, con messaggio su
  stderr. Il kernel persistente esegue `s2_stage2_v2_body` fisso (planare Q2_0); armarlo su un down
  raw leggerebbe blocchi da 18 B come planare.
- **E1** (`STRATA_MW_GPUFLAG >= 1`): `open()` fallisce, perché i kernel IQ4_NL non portano la coda
  di completion flag su cui il grafo della 3060 fa multi-wait. Non è in produzione (GPUFLAG=0).
- **Tensor split**: rifiutato per un layer con `d_type == 20` (la geometria a blocchi è quella di
  Q2_0).

→ **Iterazione 3: kernel e parity chiusi. Restano non-regressione IQ2_XS e primo load IQ3_S.**

### Finestra operativa (misurato, 2026-11-07)

Protocollo dell'operatore, eseguito nell'ordine dato. Nessuna variante al volo. Config di finestra
(restano sul server, contengono l'api_key): `strata-iq2-xs-i3-after.json` e `strata-iq3_s-i3.json`,
entrambi con `exe /root/Strata-iq3/build-iq3/strata`, `--maxwell-devices 1,2,3,4,5,6,7,8`,
`--maxwell-reserve-mib 250` e i sei `STRATA_MW_*` della produzione. Il confronto è con la baseline
**Iterazione 0 / engine 0.1.51**, non con 0.1.47.

**Passo 0 — commit prima della finestra.** Codice e risultati 03/04/05 committati e pushati su
`origin/iq3` (`0afed9e0`) prima di toccare la produzione; il worktree server ha lo stesso albero
(`d067dbf0`, tree `32e8d207`).

**Passo 1 — non regressione IQ2_XS con il motore NEW.** Output in `results/06-window-iq2-xs.txt`,
log completo in `docs/iq3/window-iq2-xs-i3-after.txt`.

| controllo | baseline 0.1.51 | motore NEW | esito |
|---|---|---|---|
| righe per layer del round-trip gate | 24 campioni | **24/24 identici**, stesso sha256 | PASS |
| `round-trip gate PASSED` | 8 | **8** | PASS |
| esperti per die | 2838 ×6, 2837 ×2 | **2838 ×6, 2837 ×2** | PASS |
| GiB per die | 3.80–3.82 | **3.80–3.82** | PASS |
| slot cache 3060 | 1874 | **1874** | PASS |
| MiB liberi | 2246 | **2242** | PASS (−4 MiB, rumore) |
| "tramonto", 600 tok | 47.3 / 48.5 tok/s | **47.5 / 49.7 tok/s** | PASS |

Il gate è bit-identico, non solo "entro la tolleranza": le 24 tuple `(device, layer, expert, tipo)`
e i loro `rel` coincidono al terzo decimale. È la verifica che il dedup a `(gu_type, d_type)` non
ha spostato la scelta dei campioni su IQ2_XS (tutti i suoi layer hanno `d_type 42`).

Il prompt è quello esatto della baseline, recuperato dalla sessione che l'ha eseguito:
*"Elenca in modo prolisso e dettagliato 200 modi diversi di descrivere un tramonto, uno per riga,
senza ripeterti."*, `temperature 0`, `max_tokens 600`, due run a caldo. Il conteggio del prompt
(84 token in ingresso) coincide con la baseline, quindi la misura è A/B a parità di tutto.

**Passo 2 — primo load IQ3_S.** Output in `results/07-window-iq3_s.txt`, log completo in
`docs/iq3/window-iq3_s-i3.txt`. Motore NEW, pack `/root/Strata-data/packs/iq3_s`
(`experts.bin` 50.292.326.400 B, il totale previsto dal design).

| controllo | misurato | esito |
|---|---|---|
| `round-trip gate PASSED` | **8/8** | PASS |
| campioni del gate | **56** = 7 combinazioni × 8 die | PASS |
| esperti per die | 2854 ×3, 2853 ×5 | — |
| GiB per die | **5.42–5.45** | conforme all'ipotesi (5.41 al minimo) |
| slot cache 3060 | **1749** | — |
| MiB liberi con tutto caricato | **396** | nessun OOM |
| caricamento | 213 s fino a `/health` | — |
| risposta di prova, 600 tok | **40.0 tok/s** client (42.9 dal motore) | — |

I 56 campioni coprono le **7 combinazioni** del pack: 18/20, 18/42, 21/20, 21/42, 22/20, 22/42,
23/20. La riga del gate stampa solo il tipo gate/up, ma il dedup è su `(gu_type, d_type)` e il
conteggio per die è esattamente 7, quindi tutte e sette — incluse le quattro nuove col down 20
(18/20, 21/20, 22/20, 23/20) — sono state campionate. `worst rel` per die: 1.00e-02 … 1.27e-02,
nello stesso intervallo della baseline IQ2_XS.

**Confronto IQ3_S vs IQ2_XS a parità di prompt (correzione del criterio).** Il «≥55 tok/s» del
brief si riferiva ai prompt **città/codice**, dove l'IQ2_XS faceva 65–67 tok/s il 5 novembre. Sul
prompt tramonto l'IQ2_XS fa 47.5 / 49.7, quindi il confronto valido con la risposta di prova
dell'IQ3_S non è 55 ma la colonna IQ2_XS dello stesso prompt:

| prompt | IQ2_XS (motore) | IQ3_S (motore) | delta |
|---|---|---|---|
| "tramonto", 600 tok, a freddo | **47.5** tok/s | **42.9** tok/s | **−9,7%** |

Cioè **circa −10%**, non "sotto target": il target va misurato sul banco città+codice, che è
l'Iterazione 5. La prima risposta dell'IQ3_S è a freddo (finestre del verifier ancora da catturare),
mentre la seconda colonna IQ2_XS della finestra (49.7) è già a caldo; il confronto omogeneo è
47.5 contro 42.9, entrambi prima generazione dopo il load.

**Passo 3 — errori.** Nessun gate fallito, nessun errore, nessun OOM. Il ramo di stop non è stato
usato.

**Passo 4 — produzione ripristinata.** `systemctl start strata`, 651 s fino a `/health`, servizio
`active` (PID 17340), `model qwen3.8-flash-next-iq2_xs`, **8 tier** con gate PASSED, 1874 slot,
**2246 MiB** liberi — gli stessi valori di prima della finestra. Controllo finale sul log della
produzione riavviata: le sue 24 righe di gate sono **identiche alla baseline 0.1.51** (stesso
`diff` pulito), quindi il motore 0.1.47 in produzione non è stato toccato da nulla di questa
iterazione.

**AER.** `results/08-window-aer.txt`. Sole `0000:09:11.0` (bridge PLX 8747, non una GPU) a **3**
prima, durante e dopo; tutte le GPU `0000:03:00.0`–`0d:00.0` a **0** in ogni istante. Il contatore
non si è mosso in tutta la finestra.

**Nota di metodo.** IQ3_S caricato **non** come servizio ma con il server di test (`up.sh` /
`gate_run.sh`) sulla porta 8080, produzione ferma e schede rilasciate prima di ogni load
(verificato con `nvidia-smi`: 0 MiB usati su tutte e 9 le schede). La produzione è tornata su con
il suo `strata.service` e il suo config originali, mai modificati.

---

## Iterazione 5 — banco fisso e verifiche per la produzione (protocollo, prossima finestra)

Iterazione 3 **confermata** dall'operatore. Questa voce prepara la finestra successiva: il banco
fisso che decide il criterio, la raccolta dei testi per il confronto di qualità, il contesto lungo e
lo soak. Codice e comandi sono pronti; le misure si fanno nella finestra, con l'OK dell'operatore.

### Il criterio, corretto

Il «≥55 tok/s» del brief si riferisce ai prompt **città/codice**, dove l'IQ2_XS faceva 65–67 tok/s
il 5 novembre. Sul prompt tramonto l'IQ2_XS fa 47.5 / 49.7: confrontare l'IQ3_S con 55 su quel
prompt sarebbe un errore di riferimento. Il confronto corretto è **a parità di prompt**:

| prompt | IQ2_XS | IQ3_S | delta |
|---|---|---|---|
| "tramonto", 600 tok, prima generazione dopo il load | 47.5 tok/s | 42.9 tok/s | −9,7% |

→ **IQ3_S costa circa il 10% di decode sullo stesso prompt**, non è "sotto target". Il criterio
≥55 resta valido sulla **media città+codice** del banco fisso, che è ciò che misura questa
iterazione.

### Il banco — `tools/iq3_bench.py`

Un solo strumento per le tre misure, con un test offline (`tools/test_iq3_bench.py`, 6 casi: campi
della riga di log, isolamento della fetta, ultima riga che vince, log assente, criterio, prompt).
Provato contro la produzione viva prima della finestra (1 run, prompt città, 60 token): riga letta
dal log del motore, `59.7 tok/s` engine, `drafts 41/51`, `1949 MiB` liberi sulla 3060, servizio
`active` e health OK dopo — nessun cambio di stato.

| categoria | prompt | note |
|---|---|---|
| `city` | quello di `/root/det_test.py`: *"Descrivi la città di Firenze in circa 150 parole…"* | identico alla baseline del 5 novembre |
| `code` | funzione Python media (`merge_intervals`, ~40 righe, type hint e due test) | |
| `prose` | *"Elenca in modo prolisso e dettagliato 200 modi diversi di descrivere un tramonto…"* | lo stesso della finestra Iterazione 3 |

Temperatura 0, `max_tokens 600`, **3 run per categoria**, IQ2_XS e IQ3_S **nella stessa sessione**
(cambio di motore fra i due, non fra le categorie). Per ogni run il tool legge la riga del motore
(`prompt N tokens … P generated in M ms (T tok/s), drafts accepted A of B`) invece di fidarsi
dell'orologio del client, e riporta tok/s e bozze accettate.

- **Ordine della finestra (operatore)**: **IQ3_S per primo** — banco, testi, contesto lungo, soak —
  poi IQ2_XS, solo banco. Così il primo load dei 3,5 minuti da SSD è quello dell'IQ3_S e l'IQ2_XS
  non paga due volte il tempo di caricamento.
- **Resoconto (operatore)**: per **ogni prompt e ogni modello**, media, **mediana** e **primo run
  riportato a parte** (il primo dopo il load è a freddo). Il tool stampa `mean … median … first …
  (run 1, cold)`, e la stessa riga di sintesi va nel LOG.
- **Criterio**: `≥55 tok/s` sulla media delle 6 run città+codice (la prosa non entra).
- **Testi**: le risposte del primo run di `city` e `code` (temperatura 0) vanno in
  `docs/iq3/results/10-quality-iq2_xs.md` e `…/11-quality-iq3_s.md`, perché l'operatore le confronti.
- **Chiave**: mai sulla riga di comando; il tool la legge da `--api-key-json` (campo `api_key`),
  `--api-key-env` o `--api-key-file`, e senza chiave manda la richiesta senza header.
- **Stop pulito**: un errore HTTP o di rete interrompe il tool con `*** STOP: …`, il messaggio del
  server e la coda del log del motore; il JSON con le righe già misurate resta su disco.

### Contesto lungo

Con **IQ3_S**, un prompt da ~32k e uno da ~64k token (fino a ~128k se regge), costruiti dallo stesso
strumento con il testo della repo e una parola-codice nascosta a metà. Per ogni lunghezza:
token reali del prompt, prefill tok/s, decode tok/s, VRAM libera sulla 3060, e se la parola-codice
è stata ritrovata. Da annotare: OOM (nessuno atteso), e se i 396 MiB liberi bastano alle finestre
del verifier a 32k/64k/128k.

### Soak

**10 minuti** con IQ3_S: prompt città in sequenza continua, 300 token per run, riportando tok/s,
deriva della velocità, VRAM libera e AER **prima e dopo**. Un degrado progressivo o un contatore
AER che si muove sono il risultato da riportare, non da interpretare qui.

### Comandi per aidrogen (finestra, dopo l'OK)

```sh
# --- 0. stato e AER prima ---
systemctl is-active strata
nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv,noheader
for d in /sys/bus/pci/devices/*; do
  [ -f "$d/aer_dev_correctable" ] && \
    echo "$(basename $d) $(grep -i TOTAL_ERR_COR $d/aer_dev_correctable | awk '{print $2}')"
done

# --- 1. IQ3_S: gate, banco fisso (testi inclusi), contesto lungo, soak ---
systemctl stop strata
bash /root/gate_run.sh /root/Strata-maxwell/strata-iq3_s-i3.json
PY=/root/Strata-maxwell/.venv/bin/python
$PY /root/Strata-iq3/tools/iq3_bench.py --label iq3_s \
  --engine-log /root/Strata-maxwell/strata-iq3_s-i3.log \
  --api-key-json /root/Strata-maxwell/strata-iq3_s-i3.json \
  --texts-out /root/Strata-iq3/docs/iq3/results/11-quality-iq3_s.md \
  --json /root/iq3s-bench.json
$PY /root/Strata-iq3/tools/iq3_bench.py --label iq3_s --long-context 32k,64k \
  --engine-log /root/Strata-maxwell/strata-iq3_s-i3.log \
  --api-key-json /root/Strata-maxwell/strata-iq3_s-i3.json \
  --json /root/iq3s-long.json
$PY /root/Strata-iq3/tools/iq3_bench.py --label iq3_s --soak-minutes 10 \
  --engine-log /root/Strata-maxwell/strata-iq3_s-i3.log \
  --api-key-json /root/Strata-maxwell/strata-iq3_s-i3.json \
  --json /root/iq3s-soak.json
# AER dopo lo soak, prima di scaricare l'IQ3_S

# --- 2. IQ2_XS: gate e banco fisso ---
bash /root/down.sh
bash /root/gate_run.sh /root/Strata-maxwell/strata-iq2-xs-i3-after.json
$PY /root/Strata-iq3/tools/iq3_bench.py --label iq2_xs \
  --engine-log /root/Strata-maxwell/strata-iq2-xs-i3-after.log \
  --api-key-json /root/Strata-maxwell/strata-iq2_xs.json \
  --texts-out /root/Strata-iq3/docs/iq3/results/10-quality-iq2_xs.md \
  --json /root/iq2xs-bench.json

# --- 3. produzione di nuovo su IQ2_XS: health, 8 tier, AER ---
bash /root/down.sh
systemctl start strata
curl -sf http://127.0.0.1:8080/health && echo
```

**Regola di stop**, come nella finestra precedente: se un gate fallisce, se c'è un errore o un OOM,
fermarsi, salvare il log, riavviare la produzione e riportare. Nessuna variante al volo.

**Cosa questa iterazione non fa.** Non cambia il codice del motore: è misura. Il criterio ≥55 vale
sulla media città+codice, che è l'unica cosa che il banco deve decidere.

---

## Iterazione 5 — banco fisso, contesto lungo e soak (misurato, 2026-11-07)

Finestra **18:01–19:45 UTC**, produzione ferma, schede rilasciate prima di ogni load. Ordine
dell'operatore: **IQ3_S per primo**, poi IQ2_XS. Nessun OOM, nessun gate fallito, nessun errore
applicativo: la regola di stop non è stata usata. Output completo in `results/09-window-i5.txt`,
log grezzi in `docs/iq3/window-iq3_s-i5.txt` e `window-iq2_xs-i5.txt`, JSON in `results/12`–`15`.

### Il criterio — **FAIL**

| motore | media città+codice (6 run) | mediana | esito |
|---|---|---|---|
| IQ3_S | **46.8 tok/s** | 46.2 | **FAIL** contro 55 |
| IQ2_XS | **58.3 tok/s** | 59.0 | **PASS** contro 55 |

L'IQ3_S costa **−19,7%** sulla media città+codice rispetto all'IQ2_XS (−22,3% su città, −17,2% su
codice, −12,7% su prosa). Non è un problema di draft: l'accettazione è simile (63,1% vs 64,4% su
città, 70,3% vs 67,1% su codice). Il divario è nel costo per token, coerente con il peso maggiore
del pack IQ3_S.

### Banco fisso — ogni prompt, media / mediana / primo run

**IQ3_S** (load 213 s, 396 MiB liberi al load, 93–95 MiB durante le generazioni)

| prompt | media | mediana | primo run (freddo) | stdev | min–max | draft |
|---|---|---|---|---|---|---|
| città | 44.1 | 43.7 | **43.7** | 1.2 | 43.1–45.5 | 1120/1775 (63,1%) |
| codice | 49.5 | 50.6 | **51.1** | 2.3 | 46.9–51.1 | 1207/1718 (70,3%) |
| prosa | 43.3 | 43.6 | **43.8** | 0.7 | 42.5–43.8 | 945/1381 (68,4%) |

**IQ2_XS** (load 640 s da HDD, 2242 MiB liberi)

| prompt | media | mediana | primo run (freddo) | stdev | min–max | draft |
|---|---|---|---|---|---|---|
| città | 56.8 | 58.4 | **53.5** | 2.9 | 53.5–58.5 | 1115/1731 (64,4%) |
| codice | 59.8 | 59.5 | **59.5** | 0.7 | 59.4–60.6 | 1157/1725 (67,1%) |
| prosa | 49.6 | 49.7 | **49.4** | 0.2 | 49.4–49.8 | 868/1315 (66,0%) |

Il primo run è a freddo su entrambi i motori, e su città è il più lento in entrambi (43.7 vs media
44.1 per IQ3_S; 53.5 vs 56.8 per IQ2_XS). La distanza fra i due motori non è un effetto del primo
run: resta su tutte e tre le categorie.

### Qualità — testi per l'operatore, con un caveat

`results/10-quality-iq2_xs.md` e `results/11-quality-iq3_s.md`: risposte città e codice del primo
run, temperatura 0, testo integrale (thinking + risposta).

**Caveat misurato, non previsto dal protocollo.** Con `max_tokens 600` **entrambi** i modelli hanno
speso tutti i 600 token nel ragionamento e non hanno prodotto la risposta finale: il campo
`content` è vuoto in tutte e quattro le risposte salvate (nessun separatore `---` nei file). I testi
sono quindi **ragionamento troncato**, non risposte finite: non sono confrontabili come qualità
della risposta. Non inficia il banco — `600 generated` è registrato dal motore in ogni run, e i
tok/s misurano il costo per token generato — ma per il confronto di qualità serve una raccolta
separata con `chat_template_kwargs.enable_thinking = false` (o un tetto più alto), che richiede di
ricaricare l'IQ3_S e non è stata fatta in questa finestra senza l'OK dell'operatore.

Per la raccolta corretta il tool ha ora `--no-thinking` (12 test offline, verde): manda
`chat_template_kwargs: {enable_thinking: false}` e lascia il ragionamento fuori dalla richiesta.
Il comando, da eseguire su entrambi i motori in una finestra successiva:

```sh
$PY tools/iq3_bench.py --label iq3_s --no-thinking --runs 1 --labels city,code \
  --engine-log ... --api-key-json ... --texts-out docs/iq3/results/11-quality-iq3_s.md
```

### Contesto lungo — IQ3_S, nessun OOM

| lunghezza | token reali | prefill | decode | parola-codice | 3060 libera |
|---|---|---|---|---|---|
| ~32k | **32.135** | 183.0 tok/s (175.6 s) | 58.2 tok/s | **TROVATA** | 93 MiB |
| ~64k | **62.703** | 182.9 tok/s (342.9 s) | 54.2 tok/s | **TROVATA** | 93 MiB |

Il prefill è **identico** alla baseline (183.0 e 182.9 contro ~181 tok/s). Il decode a 32k/64k è
**più veloce** che sul prompt corto (58.2 e 54.2 contro 44.1): con più contesto le draft del
verifier vengono accettate di più. 128k non è stato provato: 32k e 64k tenevano e non serviva
spingersi oltre.

**Il dato che conta per la produzione**: con IQ3_S la VRAM libera sulla 3060 scende da 396 MiB al
load a **93–95 MiB** durante la generazione non appena la cache si riempie, e ci resta. A 32k e 64k
non va in OOM, ma il margine è di ~90 MiB, non di ~400. Con IQ2_XS resta 1943 MiB.

### Soak — IQ3_S, 10 minuti

73 run da 300 token: **media 37.5, mediana 37.5**, stdev 1.0, min 34.5, max 39.7. Primo run 36.9
(freddo), ultimo 35.7 (**deriva −1.2 tok/s**). Primo terzo 37.6, ultimo terzo 37.5: **nessun
degrado progressivo**. VRAM libera costante a 93 MiB per tutti i 73 run.

**AER.** Prima: solo `0000:09:11.0` (bridge PLX 8747, non una GPU) a **6**, tutte le GPU a **0**.
Dopo: lo stesso bridge a **8**, GPU ancora tutte a **0**. Le 2 unità in più sono tutte `RxErr`
(errori di ricezione PCIe correggibili): **il contatore si è mosso durante la finestra**, cosa che
nella finestra dell'Iterazione 3 non era successa (3→3). Nessun OOM e nessun errore applicativo
associato; è un'osservazione da seguire, non una spiegazione.

### Gate e produzione

| controllo | IQ3_S | IQ2_XS |
|---|---|---|
| gate PASSED | **8/8** | **8/8** |
| campioni | **56** (7 combinazioni × 8 die) | **24**, identici alla baseline 0.1.51 |
| esperti/die | 2853–2854 | 2838 ×6, 2837 ×2 |
| GiB/die | 5.42–5.45 | 3.80–3.82 |
| slot cache | 1749 | 1874 |
| MiB liberi al load | 396 | 2242 |
| load | 213 s (SSD) | 640 s (HDD) |

Produzione ripristinata: `active`, PID 20630, health OK, **8/8 gate PASSED**, 1874 slot,
**2246 MiB** liberi, 0 errori, e le **24 righe di gate identiche alla baseline 0.1.51**.

→ **Iterazione 5: criterio ≥55 FAIL sull'IQ3_S (46.8 media, 46.2 mediana).** IQ3_S è ~20% più
lento dell'IQ2_XS su città+codice e lascia ~90 MiB liberi sulla 3060 invece di ~400. La produzione
resta su IQ2_XS.

---

## Iterazione 6 — diagnosi: potenza, profilo per-layer, margine VRAM (protocollo, prossima finestra)

Iterazione 5 è chiusa e misurata: l'IQ3_S costa il 19,7% in più per token e non viene promosso.
Questa iterazione è **solo diagnosi**: una variabile per volta, **nessuna ottimizzazione dei kernel**.
Le cinque misure dell'operatore (potenza, profilo per-layer, margine VRAM, qualità, AER) servono a
decidere *dove* intervenire, non a intervenire.

### Cosa è già pronto (compilato e testato, non ancora in finestra)

| strumento | cosa fa |
|---|---|
| `tools/iq3_power.py` | campiona `nvidia-smi` (`power.draw`, `power.limit`, `clocks.sm`, `clocks.mem`, `temperature.gpu`, `clocks_throttle_reasons.active`) a 1 s, salva il CSV e stampa media/mediana/max, i motivi di throttle come **quota** dei campioni e il totale contro la somma dei limiti. `--parse` riassume un CSV già scritto. Non tocca clock né limiti. |
| `tools/iq3_profile.py` | legge le righe `MWPROF` del log e separa **down-20 (IQ4_NL) da down-42 (Q2_0)**, per scheda e pesate sui layer-span. |
| `src/core/maxwell_experts.cpp` | **solo contatori** (inerti senza `STRATA_MW_LAYERPROF`): nuova riga `MWPROF m10type CUDAk down20\|down42 …`. |
| `tools/test_iq3_power.py`, `tools/test_iq3_profile.py` | 30 test offline in totale con quelli del banco; verdi in locale e sul server (3.11.2). |

La riga `m10type` viene stampata **solo se entrambi i bucket hanno span**: sul pack canonico Q2_0
(IQ2_XS, tutti i layer down-42) l'output del motore resta **identico** a prima. Misurato in
compilazione: `[100%] Built target strata`, ospite e test verdi.

La mappa dei layer è stata **verificata sul pack** (`native_experts.txt`), non copiata dal design:
**39 layer down-20 + 9 layer down-42 = 48**. I 9 down-42 sono i layer
`1, 8, 10, 12, 13, 20, 21, 30, 36` (il riepilogo delle iterazioni precedenti diceva «40/8»: era
sbagliato, il pack dice 39/9 e il tool usa quello).

### Due correzioni al punto 3, misurate prima della finestra

Il punto 3 dell'operatore («cache fissa a 1600, gli esperti in più vanno sulle M10, obiettivo
≥1 GB libero») si basa su due premesse che **il codice smentisce**. Meglio dirlo ora che scoprirlo
in finestra.

**(a) `--expert-cache N` non è «N slot».** Su un pack nativo N è il budget di **N blob più grandi**;
il motore cammina il profilo e riempie finché ci sta:

```
budget = N × max_blob            (max_blob IQ3_S = 2.662.400 B = 2.539 MiB)
slot   = i primi pair del profilo che entrano nel budget, ognuno col blob del SUO layer
```

Il modello riproduce **esattamente** il load della finestra 5 (chiedendo 1357 → 1749 slot,
3,363 GiB, 2853–2854 esperti/die, 5,424–5,450 GiB/die contro 5,42–5,45 del log): quindi è affidabile.

| `--expert-cache` | slot veri | cache | vs adesso | esperti/die | M10 GiB/die |
|---|---|---|---|---|---|
| **1600** | **2065** | 3,967 GiB | **+619 MiB** | 2814 | 5,35–5,38 |
| 1400 | 1805 | 3,470 GiB | +110 MiB | 2846 | 5,41–5,44 |
| *auto (1357, finestra 5)* | *1749* | *3,363 GiB* | *0* | *2853* | *5,42–5,45* |
| 1242 | 1599 | 3,078 GiB | −292 MiB | 2872 | 5,46–5,48 |
| 1100 | 1414 | 2,726 GiB | −652 MiB | 2895 | 5,50–5,53 |
| **988** | **1267** | **2,448 GiB** | **−937 MiB** | 2914 | 5,54–5,56 |
| 900 | 1155 | 2,232 GiB | −1159 MiB | 2928 | 5,60–5,63 |

→ **1600 è nella direzione sbagliata**: chiede *più* cache di adesso. Il budget di 1600 unità è
3,967 GiB, e in ogni caso il risultato è **≥ 3,363 GiB** — mai meno di adesso, quindi non libera
niente. Il tetto dello spazio libero (≈3,69 GiB, stimato dai termini della riga `auto`: 4,59 GiB
liberi − 700 riservati − 220 del draft) la taglia a ~1.919 slot, cioè **+334 MiB**: una stima, ma
comunque sopra i 3,363 GiB attuali. Per l'obiettivo «≥1 GB libero sulla 3060» il punto è **≈988**
(2,448 GiB di cache → 94 + 937 ≈ **1.031 MiB liberi**), oppure 1100 se si vuole restare più larghi
sulla cache (≈746 MiB liberi, sotto il target).

**(b) Gli esperti «in più» non vanno sulle M10: ci sono già.** La lista delle M10 è definita come
*«i pair che la cache della 3060 non contiene»*, in ordine di profilo, distribuiti a rotazione sulle
8 schede (`generate.cpp`, `maxwell_cand`). E i numeri del load lo confermano al byte:

```
22.827 esperti residenti sulle M10 + 1.749 slot della cache = 24.576 = 48 × 512  (tutto il pool)
```

La cache non è una copia ridondante: è **l'altra metà della partizione**. Ridurla non «sposta»
esperti sulle M10 — glieli aggiunge: a 988 i die passano da 5,424–5,450 a **5,539–5,563 GiB**
(+~118 MiB per die, aritmetica del modello validato). Le M10 hanno ~2,3–2,5 GiB liberi per die,
quindi lo assorbono senza OOM; il prezzo è che i pair meno caldi escono dalla cache della 3060 e
passano al percorso M10. Il commento del motore dà ~50 µs/espte sul hit della 3060 contro ~17 µs
aggregati sulle M10 (numeri del motore, non di questa campagna): se regge, il trade-off potrebbe
essere **favorevole** — ma è un'ipotesi da misurare, non un risultato.

**Cosa chiedo all'operatore**: confermare il valore. La finestra misura **988** (l'unico che
soddisfa «≥1 GB»), e se preferisce 1600 lo misura lo stesso — servirà a dimostrare che peggiora.

### Le cinque fasi, in quest'ordine

**Fase 0 — stato e AER prima.** Produzione `active`, 8 tier, AER di tutte le GPU e del bridge PLX.

**Fase 1 — IQ3_S, cache piena (auto), profilo acceso.** Config derivata da `strata-iq3_s-i3.json`
con `STRATA_MW_LAYERPROF=1` aggiunto in `env` (i contatori non cambiano il servizio). Con questa
singola accensione si fanno **tre** misure:

1. **POWER** — `iq3_power.py` campiona per tutta la durata: banco città+codice, poi soak da 10 minuti
   (la sequenza continua che nella finestra 5 ha dato 37,5 tok/s contro i 44 del banco). Il confronto
   banco↔soak è l'ipotesi principale: se le M10 entrano in `SwPowerCap` (bit `0x2`) solo nello soak,
   la differenza è il tetto di potenza, non il motore.
2. **TIME PROFILE** — dalle righe `m10` e `m10type`: `gup`, `down`, `sync-wait` per layer, **separando
   i 39 layer down-20 dai 9 down-42**, più le fasi di dispatch (`plan/actq/jobs/run`) che dicono
   quanto sta *fuori* dalle M10. Stesso prompt della finestra 5.
3. **QUALITY** — i testi con `--no-thinking` (città e codice): nella finestra 5 erano ragionamento
   troncato, non risposte. Raccolti qui, a cache piena, con `--texts-out`.

**Fase 2 — IQ3_S, cache ridotta (988).** Un secondo load (213 s da SSD) con lo stesso `env` e
`--expert-cache 988`. Si misurano: **MiB liberi sulla 3060 al load e durante la generazione**,
esperti e GiB **per die**, tok/s città+codice, e il campionamento di potenza (la VRAM più stretta
sulla 3060 può cambiare la pressione sulle M10). È il punto che sblocca o non sblocca la produzione.

**Fase 3 — IQ2_XS, riferimento A/B.** Config derivata da `strata-iq2-xs-i3-after.json` con
`STRATA_MW_LAYERPROF=1`. Banco città+codice, profilo, potenza, testi con `--no-thinking`. È il
confronto che rende leggibili i numeri della Fase 1: senza lo stesso profilo sull'IQ2_XS, un
`down-20` più lento non è attribuibile.

**Fase 4 — ripristino.** Produzione su IQ2_XS: `active`, health, **8/8 gate PASSED**, 1874 slot,
VRAM libera, 0 errori, 24 righe di gate identiche alla baseline 0.1.51. AER dopo.

### AER: quando si legge

Prima della finestra, e **dopo ogni fase** (dopo Fase 1, 2, 3, 4), annotando **quale carico era
attivo** quando il contatore è salito. Nella finestra 5 il bridge PLX `0000:09:11.0` è passato
3 → 6 → 8 (solo `RxErr`) mentre le 9 GPU restavano a 0: va seguito, non spiegato qui.

### Potenza: cosa questa iterazione non fa

I limiti di potenza sono **solo dell'operatore**. Le M10 sono a **30 W** (default 53, min 26,5, max
53), la 3060 a 120 W, somma dei limiti **360 W**. Lo script previsto `/usr/local/sbin/nvidia-limits.sh`
**non esiste** su aidrogen (cercato su tutto il filesystem: nessuno script con `nvidia-smi -pl`).
Se il campionamento mostra `SwPowerCap` sulle M10, il protocollo **riferisce** e l'operatore decide:
l'assistente non cambia un limite, nemmeno a 35 W.

La misura di potenza è read-only (`--query-gpu`), non scrive clock e non tocca lo stato.

### Regola di stop

Come nelle finestre precedenti: se un gate fallisce, se c'è un errore o un OOM, ci si ferma, si
salva il log, si riavvia la produzione e si riporta. **Nessuna variante al volo**, nessun secondo
tentativo con parametri diversi dentro la stessa fase.

Vale anche per il punto 3: se a 988 la 3060 va in OOM o i tok/s crollano oltre il rumore, la
conclusione è «il margine non si compra con la cache» e la produzione resta com'è.

### Comandi per aidrogen (finestra, dopo l'OK)

```sh
PY=/root/Strata-maxwell/.venv/bin/python
CFG3=/root/Strata-maxwell/strata-iq3_s-i6-prof.json
CFG3S=/root/Strata-maxwell/strata-iq3_s-i6-c988.json
CFG2=/root/Strata-maxwell/strata-iq2-xs-i6-prof.json

# --- 0. AER e stato prima ---
bash /root/aer.sh            # helper della finestra: una riga per device, solo TOTAL_ERR_COR
systemctl is-active strata

# --- 1. IQ3_S a cache piena, profilo acceso ---
systemctl stop strata
bash /root/down.sh
bash /root/up.sh "$CFG3" 2400
$PY /root/Strata-iq3/tools/iq3_power.py --label iq3_s-full-bench --minutes 12 \
    --out /root/i6-power-iq3_s-full-bench.csv &
$PY /root/Strata-iq3/tools/iq3_bench.py --label iq3_s --runs 3 --labels city,code \
    --no-thinking --engine-log /root/Strata-maxwell/strata-iq3_s-i6-prof.log \
    --api-key-json "$CFG3" --texts-out /root/Strata-iq3/docs/iq3/results/17-quality-iq3_s.md \
    --json /root/i6-bench-iq3_s-full.json
wait
$PY /root/Strata-iq3/tools/iq3_power.py --label iq3_s-full-soak --minutes 11 \
    --out /root/i6-power-iq3_s-full-soak.csv &
$PY /root/Strata-iq3/tools/iq3_bench.py --label iq3_s --soak-minutes 10 \
    --engine-log /root/Strata-maxwell/strata-iq3_s-i6-prof.log \
    --api-key-json "$CFG3" --json /root/i6-soak-iq3_s-full.json
wait
$PY /root/Strata-iq3/tools/iq3_profile.py --log /root/Strata-maxwell/strata-iq3_s-i6-prof.log \
    > /root/Strata-iq3/docs/iq3/results/18-profile-iq3_s.txt
# AER dopo la fase 1

# --- 2. IQ3_S a cache 988 ---
bash /root/down.sh
bash /root/up.sh "$CFG3S" 2400
$PY /root/Strata-iq3/tools/iq3_power.py --label iq3_s-c988 --minutes 8 \
    --out /root/i6-power-iq3_s-c988.csv &
$PY /root/Strata-iq3/tools/iq3_bench.py --label iq3_s --runs 3 --labels city,code \
    --engine-log /root/Strata-maxwell/strata-iq3_s-i6-c988.log \
    --api-key-json "$CFG3S" --json /root/i6-bench-iq3_s-c988.json
wait
# VRAM per die a cache piena e durante la generazione: nvidia-smi --query-gpu=index,memory.used,memory.free
# AER dopo la fase 2

# --- 3. IQ2_XS, riferimento ---
bash /root/down.sh
bash /root/up.sh "$CFG2" 2400
$PY /root/Strata-iq3/tools/iq3_power.py --label iq2_xs --minutes 8 \
    --out /root/i6-power-iq2_xs.csv &
$PY /root/Strata-iq3/tools/iq3_bench.py --label iq2_xs --runs 3 --labels city,code \
    --no-thinking --engine-log /root/Strata-maxwell/strata-iq2-xs-i6-prof.log \
    --api-key-json "$CFG2" --texts-out /root/Strata-iq3/docs/iq3/results/19-quality-iq2_xs.md \
    --json /root/i6-bench-iq2_xs.json
wait
$PY /root/Strata-iq3/tools/iq3_profile.py --log /root/Strata-maxwell/strata-iq2-xs-i6-prof.log \
    > /root/Strata-iq3/docs/iq3/results/20-profile-iq2_xs.txt
# AER dopo la fase 3

# --- 4. produzione ---
bash /root/down.sh
systemctl start strata
systemctl is-active strata && curl -sf http://127.0.0.1:8080/health && echo
# AER dopo la fase 4
```

I tre config si creano **sul server** copiando quelli della finestra 5 e aggiungendo la chiave in
`env` (contengono `api_key`: non si committano e non si stampano):

```sh
$PY - <<'PY'
import json
for src, dst, extra in (
    ("strata-iq3_s-i3.json",      "strata-iq3_s-i6-prof.json",   {"STRATA_MW_LAYERPROF": "1"}),
    ("strata-iq3_s-i3.json",      "strata-iq3_s-i6-c988.json",   {"STRATA_MW_LAYERPROF": "1"}),
    ("strata-iq2-xs-i3-after.json","strata-iq2-xs-i6-prof.json", {"STRATA_MW_LAYERPROF": "1"}),
):
    c = json.load(open("/root/Strata-maxwell/" + src))
    c["env"].update(extra)
    c["log"] = dst.replace(".json", ".log")
    a = c["args"]
    if "c988" in dst:
        a[a.index("--expert-cache") + 1] = "988"
    json.dump(c, open("/root/Strata-maxwell/" + dst, "w"), indent=1)
    print(dst, "->", c["log"], "cache", a[a.index("--expert-cache") + 1])
PY
```

### Cosa questa iterazione non fa

Non cambia i kernel, non cambia il criterio, non promuove l'IQ3_S. Non cambia i limiti di potenza.
Non riduce la cache in produzione: se il margine VRAM non si compra senza perdere velocità, la
risposta è «IQ3_S non va in produzione», che è già la conclusione dell'Iterazione 5.

### Stato della preparazione (misurato, prima della finestra)

| controllo | esito |
|---|---|
| compilazione del motore | `[100%] Built target strata`, dai sorgenti allineati al commit `5eb4944a` |
| test offline (potenza, profilo, banco) | **30/30 verdi** in locale e sul server (Python 3.11.2) |
| worktree del server `/root/Strata-iq3` | `5eb4944a`, **pulito**; il remoto configurato è `nuke1984`, il branch è stato allineato con un fetch esplicito da `BitMakerMan` (i due alberi coincidono file per file, `LOG.md` escluso finché non è arrivato col commit) |
| `tools/iq3_profile.py --dump` | 48 layer, **39 down-20 + 9 down-42**, mappa verificata sul pack |
| produzione IQ2_XS | `active`, health OK, **2.247 MiB liberi** sulla 3060, 4.0–4.1 GiB usati per die sulle M10 |
| AER di partenza | bridge `0000:09:11.0` a **8** (`RxErr`), **tutte le 9 GPU a 0** |

Nota operativa: `/root/aer.sh` **non esisteva**; è stato creato in preparazione (legge
`aer_dev_correctable` e stampa una riga per device, più le classi non nulle delle GPU). È read-only.

---