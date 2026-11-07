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

## Iterazione 3 — design: down IQ4_NL nativo (da rivedere prima del codice)

**Stato: DESIGN, nessun codice scritto.** Questa sezione esiste perché l'operatore la riveda prima
che parta l'Iterazione 3. Non è stata eseguita nessuna modifica oltre all'Iterazione 2.

### Perché serve

Il censimento dice: 39 layer su 48 dell'IQ3_S hanno `down = 20` (IQ4_NL, ggml type 20). Oggi
`MaxwellExperts::open` rifiuta ogni layer con `d_type != 42`, quindi il pack IQ3_S non si apre.
L'Iterazione 3 è la variabile che sblocca il caricamento: **un solo formato**, il down IQ4_NL,
sostituendo il down Q2_0 planare nel kernel S2.

### Il formato e la sua geometria

`block_iq4_nl` (ggml-common.h): `QK4_NL = 32`, `ggml_half d` + `uint8_t qs[16]` = **18 byte** per
32 pesi. Il down ha `H = 2560` righe, ognuna con `FF = 640` pesi = **20 blocchi** = 360 byte/riga,
quindi 921.600 byte per esperto contro i 460.800 del Q2_0: **il blob IQ3_S cresce di 460.800 byte
per esperto** (2.176.000 → 2.662.400 per 18/20, come da censimento). Su 2838 esperti per die sono
+1,22 GiB per die: **il vincolo di memoria è il primo rischio del design**, non il kernel.

Il decode: `valore = d * kvalues_iq4nl[nibble]`, con `kvalues_iq4nl` tabella `int8 × 16` (la stessa
che l'Iterazione 2 già stage in shared per il gate/up 23). A differenza del Q2_0 non c'è
`expand4_s2` né termine `min`: è un **dot con tabella**, come `vec_dot_iq4_nl_q8_1` di
`iq_kernels.cu`.

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

Il blob resta "gate/up verbatim + down convertito", con la stessa struttura di offset di oggi:

```
[ o_gate = 0                    ] gate: 640 righe × gu_row, blocchi GGUF grezzi
[ o_up = up_off                 ] up:   640 righe × gu_row, blocchi GGUF grezzi
[ d_codes_off = down_off        ] down: 2560 righe × 20 blocchi × {d(fp16) | qs(16 B)}
[ d_scales_off = down_off + ... ] (solo percorso Q2_0)
```

Per IQ4_NL **non c'è de-interleaving**: `block_iq4_nl` è già contiguo a 18 byte e il kernel lo legge
così com'è. Quindi la conversione è una `memcpy` del blocco, non una separazione codes/scales come
per il Q2_0. Questo è un guadagno: la funzione di conversione per 20 non fa lavoro per blocco, e
`d_row_stride` = 360 B, `d_scale_stride` = 0 (non usato).

**Il blob deve dichiarare il proprio formato di down.** Oggi `expert_maxwell_geo::BLOB` è una
costante (1.382.400 B) e `MaxwellExperts::open` la usa come stride; con il formato 20 lo stride
dipende dal layer. Serve quindi `blob_bytes` per layer, che `NativeFmt::bytes` **già fornisce**
(è per-layer: `up_off*2 + d_row*n_embd`). Il lavoro è far sì che il percorso GPU usi
`f.bytes`/il campo `blob_bytes` del blocco `g` invece di `BLOB`, come il percorso i-quant fa già
per `blob_stride` del gup. **Questo è il punto di rischio numero due**: è una modifica al percorso
di indirizzamento dei blob, non al kernel, e va verificata con lo stesso confronto SASS dell'Iterazione 2.

### Kernel down: la forma

Nome proposto `iq4nl_expert_maxwell_down_batch_async`, simmetrico a `s2_expert_maxwell_down_batch_async`:

- **Geometria invariata**: `SPLIT = 2`, `THREADS = 128`, `PER_BLOCK = 64` → una riga di `H = 2560`
  per 40 blocchi, come l'S2. La riduzione finale (`sh[e][threadIdx.x]`, somma sequenziale `s = 0..1`)
  è **la stessa**, così il contratto di arrotondamento verso il chiamante non cambia.
- **Ingresso**: `faq/fasc/fahx` prodotti dal gup (già bit-identici, verificato in Iterazione 2) →
  il kernel li consuma senza toccarli.
- **Per ogni (riga, blocco)**: `int4` di 16 code byte → `get_int_from_table_16(q4, kvalues_sh)` su
  metà basse/alte (esattamente `vec_dot_iq4_nl_q8_1`), poi `dp4a_s32` contro gli `int4` della riga
  quantizzata di `faq`.
- **Epilogo**: `acc = d * Σ (dot)` e `corr = d * Σ hx`, poi `out = acc - corr`, come l'S2. Il termine
  `corr` esiste perché il contratto con la CPU è `dot + hx` con `hx` per chunk da 32; va **ricalcato
  identico** (stessa somma, stesso ordine) o il round-trip gate fallisce.
- **`kvalues_iq4nl` in shared**: 16 byte, staged una volta per blocco, come il gate/up 23.
- **Multi-token**: nella prima versione **no**. Si parte dalla variante per-token (una riga per
  gruppo di lane), si misura, e solo se il down domina il tempo si aggiunge la variante MT
  (hoist del blocco + `apply` su E attivazioni), esattamente come è stato fatto per il gup.

### Il q8_1 epilogo

Il down consuma `faq` (int8) + `fasc`/`fahx` (float) per token, **non** i `block_q8_1` del gup.
Quindi "epilogo q8_1" qui significa: **lo stesso epilogo di quantizzazione che il gup 23 già usa**
per produrre `faq/fasc/fahx` (l'albero di shuffle, `scc = amax/127`, `hx = scc * Σ qi`). Non va
riscritto: è già bit-identico e verificato. Il punto aperto è solo che il down IQ4_NL deve
consumare quel contratto senza reintrodurre una seconda quantizzazione.

### Guard e arming

- Il guard di `open()` (~riga 173) passa da `d_type != 42` a `d_type != 42 && d_type != 20`, e il
  messaggio d'errore elenca entrambi. **È l'unica riga che si tocca del guard**, ed è il momento in
  cui il pack IQ3_S può finalmente caricare.
- `iq_supported`/whitelist e il tool di parity accettano `iq4_nl/q2_0` e `iq3_s/iq4_nl`.
- Il round-trip gate di `open()` deve coprire la combinazione `21/20` (down nuovo): `ti` passa da 7
  a 8 valori, e il gate va eseguito su un layer reale 21/20 (17, 25, 27, 28, 29, 35, 40, 44, 45 in
  IQ3_S).

### Verifica prevista

1. **Parity** `--synthetic iq3_s/iq4_nl`, `iq3_xxs/iq4_nl`, `iq2_s/iq4_nl`, `iq4_xs/iq4_nl`: il down
   IQ4_NL contro i dot di produzione del 3060 (`iq4_nl` è un tipo che `native_expert.cpp` già
   conosce). Gate 1e-5 come in Iterazione 2.
2. **Pesi reali**: i 39 layer con down 20 diventano leggibili. I candidati migliori per la parity
   sono i layer 21 (21/20) e 47 (23/20), che coprono anche il gate/up nuovo su pesi veri.
3. **SASS**: le istanze 22/16/29 devono restare identiche (il percorso di produzione non cambia) e
   l'istanza S2 Q2_0 pure. Se il kernel down nuovo è separato, questo è verificabile come in
   Iterazione 2.
4. **Memoria**: +1,22 GiB per die su 8 die. Con 2246 MiB liberi sul 3060 e 3,80–3,82 GiB/die, il
   pack IQ3_S a 8 die **potrebbe non entrare**: il numero va misurato prima di dichiarare il
   caricamento riuscito, ed è il motivo per cui questa iterazione è a rischio di "gate rosso per
   memoria" e non per correttezza.

→ **Nessun risultato: è un design in attesa di revisione.** L'Iterazione 3 non è iniziata.