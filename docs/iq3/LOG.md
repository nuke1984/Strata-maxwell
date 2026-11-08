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
- [Iterazione 6b — diagnosi 3060: quale causa di throttle (misurato)](#iterazione-6b--diagnosi-3060-quale-causa-di-throttle-misurato)
- [Iterazione 7 — design: down IQ4_NL planare](#iterazione-7--design-down-iq4_nl-planare-da-rivedere-prima-del-codice)
- [Iterazione 7 — limite noto: gli application clock delle M10 non sono impostabili](#iterazione-7--limite-noto-gli-application-clock-delle-m10-non-sono-impostabili)
- [Iterazione 7 — protocollo: la 3060 a 140 W, A/B](#iterazione-7--protocollo-la-3060-a-140-w-ab-una-variabile)

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
(+~118 MiB per die, aritmetica del modello validato). Il margine per die non è stato misurato in
questa campagna: con 8 GiB per scheda e 5,45 GiB di esperti resta **almeno** il reserve di 250 MiB
più il resto, quindi ~118 MiB in più per die sono assorbiti senza OOM — ma la finestra lo verifica
con `nvidia-smi` per die, non lo assume. Il prezzo è che i pair meno caldi escono dalla cache della
3060 e passano al percorso M10. Il commento del motore dà ~50 µs/espte sul hit della 3060 contro
~17 µs aggregati sulle M10 (numeri del motore, non di questa campagna): se regge, il trade-off
potrebbe essere **favorevole** — ma è un'ipotesi da misurare, non un risultato.

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

## Iterazione 6 — diagnosi misurata (finestra 2026-11-07, cache 988)

Finestra eseguita con l'OK dell'operatore, che ha **escluso 1600** («lo dimostra già l'aritmetica») e
ha chiesto **solo 988**. Ordine: potenza+profilo con IQ3_S auto, poi IQ3_S a 988, poi IQ2_XS di
riferimento, poi ripristino. **I limiti di potenza non sono stati toccati**: M10 a 30 W e 3060 a
120 W prima, durante e dopo.

### Due bug nei miei tool, trovati e corretti in finestra

Vanno registrati perché la prima misura di potenza è andata persa:

1. **`iq3_power.py` passava `-lms 1000` come un solo argv.** Il driver risponde
   `Option -lms 1000 is not recognized`, esce, e il sampler riporta **0 campioni** — che sembrava
   «macchina ferma». Ora sono due argv (`"-lms", "1000"`), e **zero campioni è un errore esplicito**
   (exit 2 con lo stderr del driver), non un riassunto silenzioso.
2. **`iq3_profile.py` stampava una riga per finestra**: 496 righe per scheda, illeggibile. Ora
   aggrega per scheda e per tipo di down, pesando sui layer-span.

Il banco della prima esecuzione resta valido (è misura del motore); la potenza è stata rimisurata.
I test sono passati da 30 a **34**, con due casi che avrebbero colto entrambi i bug.

**Nota metodologica.** `STRATA_MW_LAYERPROF=1` **costa il 4,7% di decode** (confronto diretto con la
finestra 5, stessa config: 46,8 → 44,6 sulla media città+codice). Non è gratis: i numeri di questa
finestra vanno letti *contro il riferimento IQ2_XS misurato nella stessa finestra*, non contro la
finestra 5. Il costo viene da 15.621 righe di `MWPROF` scritte su stderr durante il banco.

### 1. Potenza — **corretto in Iterazione 7**: le M10 hanno un application clock, la 3060 è power-capped

> **ERRATA (Iterazione 7).** Questa sezione, come tutto il LOG fino all'Iterazione 6, usava una
> **tabella dei bit sbagliata** in `tools/iq3_power.py`: dal bit `0x2` in su i nomi erano sfalsati di
> uno, quindi `ApplicationsClocksSetting` veniva letto come `SwPowerCap` e il vero `SwPowerCap` della
> 3060 come `HwSlowdown`. La tabella è corretta e fissata da test che citano le letture sul campo
> (`test_iq3_power.BitMap`). Le conclusioni che cambiano sono due, e sono a favore della chiarezza:
> **le M10 non sono power-capped**, la 3060 sì. Sotto, i numeri ricalcolati dai CSV originali con la
> mappa corretta.

Campioni, solo a GPU attive (bit `GpuIdle` escluso), intervallo 1 s:

| fase | M10 attive | draw medio | % del limite | **`ApplicationsClocksSetting`** | clock SM medio | max |
|---|---|---|---|---|---|---|
| banco auto | 820 | 22,2 W | 74% | **95%** | 1007 MHz | 1032 |
| soak auto | 4.881 | 23,3 W | 78% | **99%** | 1027 MHz | 1032 |
| banco 988 | 874 | 22,2 W | 74% | **94%** | 997 MHz | 1032 |
| IQ2_XS | 714 | 21,0 W | 70% | **93%** | 994 MHz | 1032 |

**Il bit che le M10 accendono è `ApplicationsClocksSetting`, non `SwPowerCap`**: le schede stanno a
**1032 MHz esatti e costanti** (il valore misurato è sempre 1032, mai un valore intermedio) contro un
massimo NVML di 1202 MHz, e la maschera dice che è un *application clock*, cioè un clock configurato,
non un tetto di potenza raggiunto. Il draw attivo è 21–23 W su 30 W (**70–78%**), e il contatore
`SW Power Capping` della M10 **non è cresciuto di un microsecondo** durante un banco che ha fatto
salire quello della 3060 di 7,5 s. Quindi: **le M10 non vengono mai limitate dalla potenza.** Il
"crollo banco→soak" non è spiegabile con la potenza per questa ragione strutturale, non solo
statistica.

Perché 1032 e non 1202: 1032 è l'ultimo scalino *sotto* il massimo nella tabella degli application
clock (1306…1202) e il driver riporta `Applications Clocks: N/A` con `Clock Policy` non impostata —
**non ho una spiegazione del perché quel valore sia attivo**. È un fatto misurato, non interpretato:
va verificato con l'operatore, e potrebbe dipendere dal VBIOS delle M10 o da un profilo del driver.
Di certo **non è il limite di 30 W**: alzarlo non tocchherebbe un application clock.

Il **3060**, con la mappa corretta, è quello che il limite lo tocca davvero: **`SwPowerCap` sull'81%
dei campioni attivi al banco e sull'89% nello soak**, 103 W medi (max 119,3) e 112 W (max 119,9) su
120 W, 1795–1810 MHz su 2100 max, e il contatore cresciuto di 7,5 s in 30 s di banco. È **potenza**,
non temperatura: i bit termici sono a 0 (vedi B, Iterazione 7).

Nota di metodo: i bit termici/brake sono **0%** in tutte le fasi; il riepilogo del sampler dava
percentuali più basse (53%/83%) perché le calcolava su *tutti* i campioni, idle compresi.

### 2. Profilo — il down IQ4_NL è il costo, e il gate/up dell'IQ3_S è *più veloce*

Span misurato dalla scheda (eventi CUDA sulla sua stream), per layer, mediato sulle 8 schede:

| | IQ3_S | IQ2_XS | rapporto |
|---|---|---|---|
| kernel totale | **403 µs/L** | 282 µs/L | **1,43×** |
| gate/up | **140 µs/L** | 151 µs/L | **0,93×** (IQ3_S più veloce) |
| down | **263 µs/L** | 131 µs/L | **2,01×** |
| MB/L | 4,10 | 3,10 | 1,32× |
| sync-wait host | 63 µs/L | 39 µs/L | 1,62× |

(Medie pesate sui layer-span di tutte e 8 le schede, non i soli layer down-20: quelli sono nella
tabella seguente.)

Separando per tipo di down **dentro l'IQ3_S**:

| down | layer | MB/L | kernel | gate/up | down | sync |
|---|---|---|---|---|---|---|
| **20 (IQ4_NL)** | 39 | 4,286 | **437 µs/L** | 140 | **296** | 69 |
| **42 (Q2_0)** | 9 | 3,286 | **258 µs/L** | 141 | **117** | 36 |

**Il gate/up è identico (140 vs 141 µs/L): la differenza è tutta nel down.** A parità di byte il
down IQ4_NL costa **1,94×** il Q2_0 (296/4,286 = 69 µs per MB contro 117/3,286 = 36 µs per MB).
Questo risponde alla domanda dell'operatore: **non è l'IQ3 del gate/up, è il down IQ4_NL**.

Il confronto col riferimento IQ2_XS va letto con attenzione: anche l'IQ2_XS ha un down (Q2_0), ma
il suo gate/up è Q2_0/IQ2_XXS, più lento del gate/up IQ3_S. Il vantaggio dell'IQ3_S sul gate/up
(−7%) è **reale ma piccolo**; il costo del down (**+101%**, 263 contro 131 µs/L) lo mangia. Il
residuo è la parte che l'IQ2_XS paga sul gate/up e che l'IQ3_S non paga.

`MWPROF layer` / `vlayer` (lato host, medie su 567 finestre): dispatch `run` 429 µs/layer, span 551;
verifier `ring` 65, **`pool` 551**, `post` 1,5. `pool` è la finestra in cui il verifier attende le
M10, e vale **quanto l'intero span di dispatch** (100,1%): dentro il dispatch il tempo è attesa delle
schede, non lavoro host. Il `ring` (copia in/out) è il 12% dello span. Lo span host (551) supera
quello misurato sulla scheda (403) di ~148 µs/L, e `sync-wait` (63 µs/L, misurato per scheda) non lo
spiega da solo: **non ho separato queste due quote in questa finestra**, quindi non attribuisco il
residuo all'host né alla sincronizzazione.

### 3. VRAM — 988 centra l'obiettivo, e il costo in velocità è 4,3%

Il modello preparato prima della finestra è stato **confermato al byte** dal motore:

| | auto (1357 unità) | **988 unità** |
|---|---|---|
| slot veri | 1749 | **1267** |
| cache | 3,36 GiB | **2,45 GiB** |
| esperti/die | 2853–2854 | **2913–2914** |
| GiB/die | 5,43–5,45 | **5.54–5.56** |
| **3060 libera al load** | 397 MiB | **1333 MiB** |
| **3060 libera in generazione** | **95 MiB** | **1029 MiB** |

Previsione: 1267 slot, 2,448 GiB, 2914/die, 5,54–5,56 GiB/die. Misura: identica. E la partizione è
confermata: 22.827 + 1.749 = 24.576 (auto), 23.309 + 1.267 = 24.576 (988) — **sempre il pool intero**
(`Maxwell pairs in the 3060 cache: 0 of 23309`).

**Obiettivo «≥1 GB libero» raggiunto: 1029 MiB durante la generazione**, contro 95 MiB. Le M10
assorbono +60 esperti/die (+~118 MiB) senza OOM, con 2,28–2,30 GiB ancora liberi per die.

**Il costo:** 44,6 → **42,7 tok/s** sulla media città+codice (**−4,3%**). Meno di quanto si potesse
temere, ma non gratis.

**64k di controllo a 988:** prompt 62.703 token, prefill 176,7 tok/s, decode 49,9 tok/s,
**needle FOUND**, 1027 MiB liberi. Il contesto lungo tiene anche con la cache ridotta. Il testo della
risposta è ragionamento troncato (`We need answer user's request: ...`) perché questo controllo è
stato eseguito **senza** `--no-thinking`, come nella finestra 5: il needle è stato trovato, quindi il
contesto è stato letto, ma il testo non è una risposta pulita. Non è un problema del motore.

I risultati grezzi di questa finestra sono in `docs/iq3/results/21-…` – `35-…` (15 file, 910 KiB):
banchi, soak, campionamenti di potenza, testi di qualità e profili dei due motori.

### 4. Velocità — il criterio resta FAIL, e il divario è strutturale

| | media città+codice | mediana | primo run | criterio ≥55 |
|---|---|---|---|---|
| IQ3_S auto (profilo) | 44,6 | 44,8 | 40,4 | **FAIL** |
| IQ3_S 988 (profilo) | 42,7 | 42,1 | 40,1 | **FAIL** |
| **IQ2_XS riferimento** | **57,2** | **57,2** | 51,2 | **PASS** |

**IQ3_S è il 22,0% più lento dell'IQ2_XS** a parità di profilazione (44,6 vs 57,2). Coerente con
l'Iterazione 5 (−19,7%): la differenza di 2,3 punti è compatibile col rumore e col fatto che i due
motori scrivono un numero diverso di righe di profilo (l'IQ3_S ha due bucket `m10type` per scheda,
l'IQ2_XS uno solo). Non è un peggioramento del motore, ma **non posso dimostrarlo con questa
finestra**: servirebbe un banco senza profilo, che l'operatore non ha chiesto.

Soak: IQ3_S **35,2 tok/s** (68 run, mediana 35,1, drift −0,9) contro 37,5 dell'Iterazione 5. Il −6%
è **compatibile** col −4,7% di costo della profilazione misurato sul banco (ipotesi: non ho uno soak
senza profilo per dimostrarlo). Il punto strutturale resta un altro: **banco→soak scende di 9,4 tok/s**
(44,6→35,2), come nella finestra 5 scendeva di 8,1 (46,8→37,5). **Il crollo banco→soak non è la
potenza e non è il profilo**: nella finestra 5 non c'era profilazione e il crollo c'era lo stesso,
con la stessa forma.

Draft acceptance sostanzialmente identica (65,8% IQ3_S, 66,2% IQ2_XS): il divario è **costo per
token accettato**, non qualità delle draft.

### 5. Qualità — i testi ora sono risposte, non ragionamento troncato

Con `--no-thinking` entrambi i motori producono testi completi e corretti:

- **IQ3_S città**: prosa piena (Firenze, Medici, Brunelleschi, Uffizi, Ponte Vecchio). 238 token.
- **IQ3_S codice**: `merge_intervals` completa, type hint, docstring, **due casi di test**, e la
  logica tratta correttamente l'adiacenza (`current[0] <= last[1]`). 432 token.
- **IQ2_XS città**: equivalente per contenuto, struttura simile.
- **IQ2_XS codice**: equivalente, con esempi nel docstring invece che come assert.

**Nessuna differenza di qualità osservabile a occhio nei due testi.** Il che è coerente col fatto che
il divario è di *costo*, non di capacità — ma un confronto di qualità serio richiederebbe un banco
di valutazione, che questa iterazione non ha.

### 6. AER — solo il bridge, le GPU tutte a 0

| momento | bridge `0000:09:11.0` | GPU |
|---|---|---|
| prima della finestra | 8 | tutte 0 |
| dopo fase 1 (IQ3_S auto) | 10 | tutte 0 |
| dopo fase 2 (IQ3_S 988) | 10 | tutte 0 |
| dopo fase 3 (IQ2_XS) | **11** | tutte 0 |
| dopo fase 4 (produzione) | 11 | tutte 0 |

Il contatore è salito **+3** in finestra: 8→10 durante la fase 1 (due load IQ3_S: auto e 988), poi
10→11 durante la fase 3 (load IQ2_XS da HDD). Sempre `RxErr` sul bridge PLX 8747, **mai su una GPU**.
Il pattern resta «sale durante i load», non durante il calcolo. Da seguire, non da spiegare qui.

### Ripristino produzione — verificato

| controllo | esito |
|---|---|
| servizio | `active`, `NRestarts=0`, health OK |
| tier | **8** Maxwell: GPU 1–6 a **2838** esperti, GPU 7–8 a **2837** |
| cache | **1874 slot**, 2,53 GiB |
| gate | **8/8 PASSED** (worst rel 1,12e-02 – 1,26e-02) |
| 3060 libera | **2247 MiB** |
| M10 libere | 4068–4084 MiB per die |
| limiti di potenza | **M10 30 W, 3060 120 W — invariati** |
| AER | bridge 11, GPU tutte 0 |

### Conclusione dell'iterazione

**La diagnosi ha una risposta netta e una condizione.**

1. **Il costo dell'IQ3_S è nel down IQ4_NL**, non nel gate/up (che è addirittura più veloce
   dell'IQ2_XS). Nei layer down-20 il down costa **296 µs/L** contro **117** dei layer Q2_0, a parità
   di byte **1,94×** (69 contro 36 µs/MB). Sull'intero modello il down pesa 263 µs/L contro 131
   dell'IQ2_XS. Questo è il punto su cui un'eventuale Iterazione 7 dovrebbe lavorare, **se** si vuole
   insistere sull'IQ3_S.
2. **Il margine VRAM si compra**: 988 porta la 3060 da 95 a 1029 MiB liberi, tiene 64k, e costa il
   4,3%. È un trade-off **difendibile** — ma non cambia il criterio: a 988 l'IQ3_S fa 42,7 contro
   57,2 dell'IQ2_XS, ed è **FAIL**.
3. **La potenza non è la leva** — e in Iterazione 7 si è scoperto *perché*, correggendo la mappa dei
   bit di questo stesso LOG: quello che le M10 accendono è `ApplicationsClocksSetting` (stanno a
   1032 MHz di application clock, 21–23 W su 30 W, contatore `SW Power Capping` fermo), **non**
   `SwPowerCap`. Le M10 non sono limitate dalla potenza, quindi alzare il limite non dà clock. La
   scheda che il limite lo tocca è la **3060** (`SwPowerCap` all'81–89%, bit termici a zero). Il
   crollo banco→soak esiste anche senza profilazione (finestra 5) e non è attribuibile alla potenza.

**IQ3_S non viene promosso.** La produzione resta su IQ2_XS, ripristinata e verificata. La sola
strada con un numero a favore è il down IQ4_NL; qualsiasi altra (potenza, cache, contesto) o non
sposta il criterio o lo peggiora.

---

## Iterazione 6b — diagnosi 3060: quale causa di throttle (misurato)

**Obiettivo.** L'Iterazione 6 ha misurato `HwSlowdown` sull'81–89% dei campioni attivi della 3060.
Quel bit però non dice *quale* meccanismo: termico, power-brake o potenza. L'operatore ha chiesto di
distinguere la causa, **senza cambiare nulla** (ventole e cavi sono hardware: decide lui).

**Come.** `tools/iq3_power.py --deep` è nuovo e campiona, oltre alla maschera aggregata, i singoli
campi `clocks_throttle_reasons.{hw_slowdown, hw_power_brake_slowdown, hw_thermal_slowdown,
sw_power_cap, sw_thermal_slowdown}`, più `temperature.gpu`, `temperature.memory`, `fan.speed`. La
misura è stata fatta **sulla produzione**, senza fermarla: un banco città+codice contro l'endpoint IQ2_XS
già attivo (2341 campioni, 500 ms), e i campioni sono riportati **solo a GPU attiva**, perché una
scheda a 210 MHz non è in throttle termico, è addormentata.

**Misura (3060, 139 campioni attivi):**

| grandezza | valore |
|---|---|
| `HwThermalSlowdown` | **0%** |
| `HwPowerBrakeSlowdown` | **0%** |
| `SwThermalSlowdown` | **0%** |
| `HwSlowdown` | **0%** |
| **`SwPowerCap`** | **73%** |
| temperatura GPU | media 58,6 °C, **max 67 °C** |
| temperatura memoria | **non esposta** dal driver (`N/A` su ogni campione) |
| ventola | media 36,8%, max 43% |
| draw | media 100,4 W, **max 120,3 W** su 120 W |
| clock SM | media 1789, max 1912 su 2100 |

**Risposta: è potenza, non temperatura.** Nessun bit termico è mai attivo, la GPU sta a 67 °C massimi
con la ventola al 43% (ha margine termico, non sta scaldando), e il draw tocca il limite di 120 W. Il
throttle della 3060 è **`SwPowerCap`**, esattamente come per un limite di potenza raggiunto.

**Ma la misura ha scoperto un errore più importante, che corregge l'Iterazione 6.** La maschera
aggregata e i campi individuali si contraddicevano: `mask=0x4` con `sw_power_cap=Active`. La tabella
dei bit in `tools/iq3_power.py` era **sfalsata di uno da `0x2` in su**, quindi:
- quello che il LOG dell'Iterazione 6 chiamava `SwPowerCap` sulle M10 era in realtà
  **`ApplicationsClocksSetting`** (bit `0x2`);
- quello che chiamava `HwSlowdown` sulla 3060 era in realtà **`SwPowerCap`** (bit `0x4`).

La mappa è ora corretta, fissata da test che citano le letture sul campo (`BitMap` in
`tools/test_iq3_power.py`), e i CSV dell'Iterazione 6 sono stati **ricalcolati**. Le due conclusioni che
cambiano sono nella §1 dell'Iterazione 6 (con errata) e qui: **le M10 non sono power-capped**, la 3060
sì. Verifica indipendente: il contatore cumulativo `SW Power Capping` della M10 **non è cresciuto** in
un banco in cui quello della 3060 è cresciuto di 7,5 s.

**Cosa non dico.** Perché le M10 stiano a 1032 MHz (application clock) e non a 1202 non è spiegato da
questa misura: il driver riporta `Applications Clocks: N/A` e nessuna `Clock Policy` impostata. È un
fatto misurato, non interpretato. Non ho toccato né ventole, né limiti, né cavi.

**Risultati:** `docs/iq3/results/36-power-iq2_xs-deep.csv`, `37-bench-iq2_xs-diag.json`. Il banco di
controllo ha fatto **58,3 tok/s** (mediana 56,8), coerente con i 57,2 dell'Iterazione 6.

---

## Iterazione 7 — design: down IQ4_NL planare (da rivedere prima del codice)

**Stato: design. Nessuna riga di codice scritta, nessun file toccato.** Come chiesto dall'operatore:
prima il design, poi la revisione, poi l'implementazione.

### Obiettivo

Togliere al down IQ4_NL (ggml tipo 20) il difetto che gli impedisce di usare letture vettoriali: nel
blob esso è **grezzo**, quindi i codici di ogni blocco cominciano a `+2` (dopo il `d` fp16) e sono
allineati a **2 byte, mai a 16**. Al `open()` la regione down viene riorganizzata in forma
**planare** — codici di ogni riga contigui e a 16 byte, scale fp16 in un piano separato — **con gli
stessi byte, senza costo di VRAM, senza perdita di informazione**. Il kernel legge i codici con
`int4` (128 bit).

### La misura da battere

Dal profilo dell'Iterazione 6 (`results/27-profile-iq3_s-i6.txt`), per layer per scheda:

| gruppo | MB/L | kernel µs/L | gup | **down** | sync | down GB/s |
|---|---|---|---|---|---|---|
| down-20 (IQ4_NL, 39 layer) | 4,286 | 437 | 140 | **296** | 69 | **14,5** |
| down-42 (Q2_0/S2, 9 layer) | 3,286 | 258 | 141 | **117** | 36 | **28,1** |

Il confronto è onesto perché è lo stesso kernel-host, le stesse schede, la stessa finestra: **lo
stesso lavoro per byte costa il doppio quando i codici sono a 2 byte invece che planari**. Il tetto
teorico della M10 è ~83 GB/s (colonna `ideal` del profilo). Lo span down-20 per layer, comprese le
fasi host, è **553,6 µs** (plan 3,5 + actq 107 + jobs 6 + run 437); quello down-42 è **374,6 µs**
(run 258).

### L'ipotesi, e il numero che serve

Se il divario 14,5 vs 28,1 GB/s è **il disallineamento** (non i byte, che sono identici), il down-20
deve avvicinarsi al tasso del Q2_0. Quanto serve per il criterio? L'IQ3_S a 988 fa **42,7 tok/s** e
il criterio è **55**, cioè **+28,8%**, cioè **−22,4% di tempo per token**.

Il conto, sulla sola misura (nessuna previsione, solo proporzioni):

| grandezza | valore | fonte |
|---|---|---|
| tempo per token a 42,7 tok/s | **23.400 µs** | banco I6 |
| down-20: 39 layer × 296 µs/L | **11.544 µs = 49,3% del tempo per token** | profilo I6 × calcolo |
| down-42: 9 layer × 117 µs/L | 1.053 µs | profilo I6 × calcolo |

Metà del tempo per token è il down IQ4_NL: **è la leva, e non ce n'è una seconda**. Togliendo `S` µs
a ciascuno dei 39 layer down-20, `tok/s = 42,7 · 23.400 / (23.400 − 39·S)`. Per arrivare a 55:
**S = 134 µs/L**, cioè il down-20 da **296 a 161 µs/L** (14,5 → **26,6 GB/s**).

E qui il numero è scomodo, meglio dirlo adesso: 161 µs/L è il **106% del tasso del Q2_0** applicato
ai byte del down-20 (152,5 µs a 28,1 GB/s). Il criterio chiede quindi **più di tutto** il dividendo
della vettorializzazione — un **1,83×** sul down — non una parte:

| down-20 µs/L | GB/s | tok/s attesi | criterio 55 |
|---|---|---|---|
| 296 (oggi) | 14,5 | 42,7 (misurato) | FAIL |
| 250 | 17,1 | 46,2 | FAIL |
| 220 | 19,5 | 48,9 | FAIL |
| 182 | 23,5 | 52,7 | FAIL |
| 161 | 26,6 | 55,1 | **PASS (limite)** |
| 152,5 (= tasso Q2_0) | 28,1 | 56,1 | PASS |
| 107 (se il tetto fosse 40 GB/s) | 40,1 | 62,3 | PASS con margine |

Lettura per l'operatore: questa iterazione **non ha margine di sicurezza**. Se la vettorializzazione
dà metà del dividendo (down ~224, ~19 GB/s) il modello resta intorno a 48 tok/s, sotto il criterio, e
lo si saprà con un numero. È anche il motivo per cui la LUT in shared resta una variabile separata e
pronta: potrebbe essere il pezzo che manca dopo questo.

Una nota sulla decomposizione, perché i numeri vadano letti per quello che sono: la somma degli span
per layer del profilo (39 × 553,6 + 9 × 374,6 = **24.962 µs**) è **6,6% sopra** il tempo per token
misurato (23.400 µs). Le due cose non sono la stessa grandezza — lo span per scheda comprende attese
che sul percorso si sovrappongono — quindi il conto qui sopra usa **la misura** come base e il
profilo solo per la **quota** del down (49,3%). **Ma la sensibilità è alta, e va dichiarata**: sulla
base del profilo lo stesso criterio chiederebbe **S = 174 µs/L**, cioè un down-20 a **122 µs/L**,
sotto il tasso del Q2_0. In altre parole: **il criterio cade esattamente al bordo del massimo
teorico di questa leva**, e se la base vera fosse quella del profilo questa leva da sola non
basterebbe. È il motivo per cui la verifica (5) misura i µs/MB *prima* di promettere il banco: se il
down non scende sotto ~160 µs/L, il banco non può dare 55 e non serve spenderlo.

Labelled explicitly: **ipotesi** = il disallineamento è la causa e la vettorializzazione recupera il
divario; **misura** = le due grandezze della prima tabella e il 42,7 tok/s, che vengono dal profilo e
dal banco dell'Iterazione 6.

### Il layout

Per riga (FF = 640 valori = 20 blocchi da 32):

| piano | contenuto | byte/riga | allineamento |
|---|---|---|---|
| codici | 20 × 16 B di nibble (gli stessi `qs` di oggi) | **320** | 16 B (320 % 16 == 0) |
| scale | 20 × fp16 (gli stessi `d` di oggi) | **40** | 2 B |
| **totale** | | **360** | **= 20 × 18, identico al grezzo** |

Per esperto (H = 2560 righe): codici 2560 × 320 = **819.200 B (0,781 MiB)**, scale 2560 × 40 =
**102.400 B (0,098 MiB)**, totale **921.600 B (0,879 MiB)** — **esattamente i byte di oggi**. Il blob
non cambia dimensione: `f.bytes` resta valido e il controllo `f.bytes != blob` di
`expert_layout_load` continua a passare.

Dentro il blob, la regione down resta dove è oggi:

```
[ 0                    ] gate: 640 righe × gu_row, blocchi GGUF grezzi (INVARIATI)
[ f.up_off             ] up:   640 righe × gu_row, blocchi GGUF grezzi (INVARIATI)
[ f.down_off           ] codici: 2560 × 320 B          <- d_codes_off, d_row_stride = 320
[ f.down_off + 819200  ] scale:  2560 × 40 B           <- d_scales_off, d_scale_stride = 40
[ ... 921600 B totali  ] nessun padding, nessun byte aggiunto
```

**Perché i codici prima e le scale dopo** (e non il contrario): così `d_codes_off` resta
`f.down_off`, che è già il valore provato, e il piano delle scale cade a `+819.200`, anch'esso
multiplo di 16, senza padding. L'ordine è anche quello che l'S2 usa già per il Q2_0
(`d_scales_off = down_off + H*ROW_D // S2 planar: codes then scales`): una sola convenzione nel tier.

**L'allineamento a 16 byte è dimostrabile, non sperato.** `native_fmt` calcola
`down_off = 2 · gu_row · n_ff` con `n_ff = FF = 640`, e `gu_row = (H/256) · block_bytes = 10 · bs`.
Quindi `down_off = 2 · 10·bs · 640 = 12.800 · bs`, e **12.800 = 16 · 800** → `down_off % 16 == 0` per
**qualunque** `bs` intero, non solo per quelli di questo pack. Verificato anche sul pack reale:
**48/48 layer con `down_off % 64 == 0`** — i blocchi in gioco sono sei, `bs` ∈ {62 IQ1_M, 82 IQ2_S,
86 IQ2_XXS, 98 IQ3_XXS, 110 IQ3_S, 136 IQ4_XS}, tutti pari (e 86 non lo sarebbe stato per un 32: è
la ragione per cui il multiplo che conta è 16, non 32). Il design aggiunge comunque un **guard a
runtime in `open()`** che rifiuta con un messaggio esplicito se `d_codes_off % 16 != 0`: un'invariante
non controllata è un'invariante che un giorno si rompe in silenzio.

### Perché `int4` è lecito qui (e non lo era prima)

`get_int_b2(qs, i)` non legge 2 byte: legge **due** `uint16_t` (`x16[2i]`, `x16[2i+1]`) e li compone
in una parola da 32 bit. Per blocco sono 4 chiamate → **8 load da 16 bit**. Il compilatore non può
fonderli, perché dal suo punto di vista il puntatore è a 2 byte (`blk + 2`, dentro una struct da 18 B
senza `alignas`), e una `LDG.128` su un indirizzo non allineato è un fault, non un rallentamento: è la
ragione per cui sia `vec_dot_iq4_nl_q8_1` sia `Split<20>` usano `get_int_b2` e non un cast vettoriale.

**Questo non è un'argomentazione: è misurato.** `results/38-down-iq4nl-sass-before.txt` conta le
istruzioni di memoria dei tre kernel IQ4_NL sull'oggetto del branch (`43c84922`, `cuobjdump -sass`,
sm_50):

| kernel | istruzioni | `LDG.128` | **`LDG.U16`** | `LDG.U8` |
|---|---|---|---|---|
| `expert_iq4nl_down_batch<2,false>` | 540 | 29 | **8** | 3 |
| `expert_iq4nl_down_batch<2,true>` | 564 | 29 | **8** | 3 |
| `expert_iq4nl_down_mt<2,128,4>` | 804 | 31 | **8** | 4 |

Gli **8 `U16` per corpo di loop** sono esattamente `get_int_b2` × 4 × 2 half-word: il percorso dei
codici. Le `LDG.128` già presenti **non** sono i pesi — nel corpo non esiste altro `int4` che la copia
di `faq` in `iq4nl_stage_act` (`dstq[i] = srcq[i]`, globale→shared) — e le `U8` sono `read_scale`,
che legge i due byte della fp16 **separatamente** (`scales[b*2]`, `scales[b*2+1]`). Quindi oggi un
blocco costa **10 load**: 8 × U16 + 2 × U8.

Con il piano planare la base della riga è `d_codes_off + row·320`, con entrambi i termini multipli di
16, e il blocco `b` sta a `+ b·16`: **un solo `LDG.E.128` per blocco**. I quattro `int32` che ne
escono sono **gli stessi quattro** che `get_int_b2` componeva, nello stesso ordine, e alimentano lo
stesso `get_int_from_table_16`: il valore di `p[0..7]` è identico bit per bit, cambia solo come
arrivano i byte.

Il piano delle scale rende **provabile** anche la seconda metà: `d_scales_off + row·40 + b·2` è a 2
byte, quindi la fp16 si legge con **una** `LDG.U16` invece di due `U8` — stessi bit, un load. Questo
usa un helper nuovo (`read_scale_u16`) e **non** modifica `read_scale`, che serve i kernel S2 di
produzione: toccarlo cambierebbe il SASS del percorso IQ2_XS, che il gate (4) vieta.

Conto per riga: **200 → 40 load** (10 → 2 per blocco), cioè **8 × U16 + 2 × U8 → 1 × 128 + 1 × U16**.
La riduzione di istruzioni è 5×; quella dei soli codici è 8×, ed è quella che il design rivendica.

**Il contratto d'indice, da non sbagliare.** Nel corpo di oggi il peso si indicizza con
`(b - blk_base) * 18` mentre l'attivazione con `b * 32` (il loop parte da `b = blk_base + part`): le
due cose coincidono solo perché `blk_base` è **sempre 0** per il tipo 20 — il tensor split lo rifiuta
(`maxwell_experts.cpp:257`) e le tre chiamate passano `0, kDBlocksIq4nl`. Il design **conserva questa
forma esattamente**: codici a `+ (b - blk_base)·16`, scale a `+ (b - blk_base)·2`, attivazioni a
`+ b·32`. Non è un dettaglio teorico: scrivere `b·16` per i pesi darebbe risultati giusti finché
`blk_base == 0` e sbagliati il giorno in cui qualcuno abilita lo split — ed è il tipo di errore che
nessun test attuale vedrebbe, perché tutti passano `blk_base = 0`.

### Cosa cambia, file per file

| file | modifica |
|---|---|
| `include/strata/kernels/cuda/expert_maxwell.hpp` | le 3 firme `iq4nl_expert_maxwell_down_{batch,batch_f16,mt}_async` prendono `d_scales_off` e `d_scale_stride`; nuove costanti di geometria `kIq4nlCodesRow = FF·4/8 = 320`, `kIq4nlScalesRow = (FF/32)·2 = 40`; commento d'uso aggiornato |
| `src/kernels/cuda/expert_maxwell.cu` | `iq4nl_load_block` prende un `int4` invece di un puntatore; nuovo `read_scale_u16` accanto a `read_scale` (che resta intatto, è dei kernel S2); i tre corpi leggono `dc = blob + d_codes_off + row·d_row_stride` come `int4` e la scala da `d_scales_off + row·d_scale_stride + b·2`; **il launcher MT smette di cablare `d_row_stride = 20·18`** (riga 1918) e usa il parametro |
| `src/core/maxwell_experts.cpp` | `geo` per `d_type == 20`: `d_codes_off = f.down_off`, `d_row_stride = 320`, `d_scales_off = f.down_off + H·320`, `d_scale_stride = 40`; la copia al `open()` sostituisce il `memcpy` verbatim (riga 493) con la conversione planare; guard di allineamento; commento di testa (righe 68–73) riscritto |
| `include/strata/core/maxwell_experts.hpp` | solo il commento di `LayerGeo`: il tipo 20 non è più "raw" |
| `src/kernels/cuda/expert_maxwell_iq_parity.cpp` | costruisce il blob **planare** invece di copiare le righe grezze, chiamando **la stessa funzione di conversione di `open()`**; nuovo flag `--dump <file>` per la bit-identità contro HEAD (contratto, punto 3) |

La conversione è una funzione host condivisa (`iq4nl_planarize(src_raw, dst_planar, rows)`), così il
test di parity esercita **esattamente** il codice che la produzione esegue al load: se il layout è
sbagliato, il test lo vede; se il test lo vede solo lui, la produzione è salva.

### Il contratto di verifica

1. **La conversione è una permutazione di byte, e si verifica byte per byte**: per ogni riga `r` e
   blocco `b` di un esperto reale, `planar.codes[r·320 + b·16 .. +16] == raw[r·360 + b·18 + 2 .. +16]`
   e `planar.scales[r·40 + b·2 .. +2] == raw[r·360 + b·18 .. +2]`, su tutte le 2560 × 20 coppie.
   Poiché i byte sono gli stessi e l'aritmetica del kernel non cambia, questo è più forte di un
   round-trip: non c'è nulla da invertire.
2. **Parity 1e-5 contro `iq_mmvq` tipo 20** sui pesi reali (layer 0, 17, 2, 47 — le quattro
   combinazioni 18/20, 21/20, 22/20, 23/20) e su sintetico, come in Iterazione 2/3:
   `maxwell_iq_parity`, check (4) e (4b). Il numero atteso non è "sotto 1e-5": deve restare
   **lo stesso di oggi** (9,1–9,6e-08).
3. **Bit-identità vecchio vs nuovo, misurata e non argomentata**: `maxwell_iq_parity` acquisisce un
   flag `--dump <file>` che scrive il vettore di uscita del down (nt × H float, esadecimale). Si
   esegue il binario **di HEAD** (blob grezzo, kernel di oggi) e quello **del branch** (blob planare,
   kernel nuovo) sugli **stessi pesi, le stesse `faq`/`fasc`, lo stesso seed**, e si confrontano i due
   dump: **0 float diversi**, non 1e-5. È la prova che il layout non ha cambiato i valori, e non
   richiede di tenere in vita il vecchio kernel (che sarebbe codice morto in produzione). Se questa
   passa e la (2) peggiora, l'errore è nel layout; se questa fallisce, l'errore è nel load.
   Il check (4b) del tool copre già la stessa proprietà fra batch, f16 e MT: se lo stride dell'MT
   restasse cablato a 360, l'MT divergerebbe dal batch **dentro il tool** e il check fallirebbe.
4. **SASS**: le istanze dell'S2 (`expert_s2_stage2_batch_v2` 5, `expert_s2_stage2_mt_batch` 1,
   `expert_s2_stage1` 7, `expert_s2_stage2` 8) e le 66 delle 22/16/29 **bit-identiche** allo stesso
   metodo dell'Iterazione 2 (`cuobjdump -sass`, normalizzando solo l'hash del namespace anonimo).
   Il kernel IQ4_NL **deve** cambiare: è la variabile. Sul kernel nuovo, contro la baseline
   `results/38-down-iq4nl-sass-before.txt`: **`LDG.U16` sul percorso dei codici = 0** (oggi 8; le
   U16 che restano sono le 20 scale, una per blocco) e **`LDG.128` nel corpo dei pesi presente** (oggi
   il corpo non ne ha nessuna). Se il compilatore non emette la vettoriale, il layout da solo non
   serve e va detto prima di misurare.
5. **µs/MB del down con `STRATA_MW_LAYERPROF=1`**, prima/dopo, **nella stessa finestra** (la
   profilazione costa il 4,7%: un confronto fra finestre diverse non è un confronto). Si leggono
   `down20 down us/L`, `MB/L` e il `kernel` per scheda, come nella tabella della misura da battere.
6. **Banco città+codice A/B nella stessa finestra**, 3 run per label, senza LAYERPROF, contro il
   criterio: media delle 6 run ≥ **55 tok/s**. Le due configurazioni sono **due binari** (HEAD e
   branch, stessa config 988) alternati nella stessa finestra, come in Iterazione 6.

### Risultato atteso (dichiarato come ipotesi)

- **Da 296 a ≤161 µs/L** il down-20 (14,5 → ≥26,6 GB/s): è il minimo per il criterio, e non c'è
  margine — è il 106% del tasso del Q2_0.
- Se il disallineamento era l'unica causa e si raggiunge il tasso del Q2_0: 152,5 µs/L, ~56 tok/s.
  Se il tetto vero fosse 40 GB/s (107 µs/L), ~62 tok/s.
- Se resta a 296: la causa non era il load. Il codice resta comunque in uno stato in cui la prossima
  variabile (la LUT in shared) si misura da sola.
- **Soglia di stop dichiarata prima di misurare**: se il down-20 non scende sotto **~200 µs/L**
  (misura 5), il banco (6) non può dare 55 e non si spende. Se scende sotto ~161, il banco si fa.

### Cosa questo design non fa

- **Non tocca la LUT `get_int_from_table_16`**: la variante con `kvalues_iq4nl` in shared come
  `uint32_t` è una **variabile separata**, da valutare dopo, quando questo layout sarà chiuso. Qui la
  LUT è già in shared (`shl`, 16 B) e la funzione non cambia.
- **Non tocca il percorso di produzione Q2_0**: né i kernel S2, né la loro geometria, né il blob
  canonico. Il gate è il SASS (punto 4).
- **Non tocca** l'aritmetica, l'ordine di riduzione, il numero di thread/blocchi, lo staging delle
  attivazioni (`int4`, già così), l'arrotondamento fp16 di `fasc`, `GPUFLAG`, `PERSIST`, il tensor
  split (che già rifiuta il tipo 20), la cache 988, la potenza.
- **Non tocca `read_scale`**, che serve i kernel S2 di produzione: la lettura a `uint16_t` della fp16
  è un helper nuovo usato solo dai kernel IQ4_NL, così il SASS del percorso IQ2_XS resta identico.
- **Non aggiunge VRAM**: 921.600 B per esperto prima e dopo.

### Rischi aperti

| # | rischio | mitigazione |
|---|---|---|
| R1 | `d_codes_off` non a 16 byte su un pack futuro | dimostrato `12.800·bs`; guard a runtime in `open()` che rifiuta |
| R2 | il launcher MT cabla ancora `20·18` (riga 1918) | è nella tabella dei file: se sfugge, l'MT legge a stride sbagliato e il test (3) lo prende |
| R3 | il tool di parity copia righe grezze | usa la funzione condivisa, non una copia del layout |
| R4 | altri lettori della regione down | verificato: `mwdown::{batch,batch_f16,mt}` sono gli unici, tutti via `g.d_*`; `PERSIST` declina il tipo 20, `GPUFLAG` lo rifiuta, il tensor split lo rifiuta |
| R5 | il piano delle scale a 40 B/riga non è 16-allineato | non serve: la fp16 si legge a 2 byte con **una** `LDG.U16` (`read_scale_u16`, nuovo), e il `d` è copiato **verbatim**. `read_scale` resta intatto per non toccare il SASS dell'S2 |
| R6 | il compilatore non emette `LDG.128` | controllo esplicito nel SASS del kernel nuovo (punto 4), prima di misurare |
| R7 | indice dei pesi scritto come `b·16` invece di `(b - blk_base)·16` | innocuo finché `blk_base == 0` (sempre, oggi) e sbagliato se un domani lo split si abilita: il design conserva la forma, e il commento nel kernel lo dice |

---

## Iterazione 7 — limite noto: gli application clock delle M10 non sono impostabili

**Domanda chiusa.** L'Iterazione 6 — e la sua §1 in particolare — lasciava aperto *perché* le M10
stiano a 1032 MHz invece che ai 1202 riportati da NVML. **Non è più una domanda aperta.** L'operatore
ha provato a impostare gli application clock con `nvidia-smi`, sul host, sulla die 0. La risposta del
driver è stata:

```
Setting applications clocks is not supported
```

I clock sono stati ripristinati subito dopo. Quindi:

| grandezza | valore | stato |
|---|---|---|
| clock delle M10 | **~1032 MHz** | **base clock**, non un tetto raggiunto |
| application clock | **non impostabili** su questo driver | limite del driver, chiuso |
| massimo NVML | 1202 MHz | teorico, non ottenibile per questa via |

Va letto per quello che è: **un limite noto del sistema**, non un esperimento da rifare e non una leva
disponibile. Conseguenze pratiche, da qui in avanti:

- il bit `ApplicationsClocksSetting` sulle M10 (93–99% dei campioni attivi) **non è un sintomo da
  curare**: è il driver che riporta il clock base con cui le schede girano;
- **alzare il limite di potenza delle M10 non può dare clock**, perché non è la potenza a fissare i
  1032 MHz (confermato dal draw 21–23 W su 30 W e dal contatore `SW Power Capping` fermo);
- ogni stima che assuma «M10 a 1202 MHz» va buttata: il tetto di calcolo delle M10 è quello che si
  misura, non quello che NVML dichiara.

Resta **la 3060** l'unica scheda su cui la potenza agisce — ed è l'oggetto della finestra seguente.

---

## Iterazione 7 — protocollo: la 3060 a 140 W, A/B (una variabile)

**Stato: protocollo. Nessuna misura fatta, nessun limite toccato.** I limiti di potenza sono
dell'operatore: **i 140 W sulla 3060 li imposta lui**, non l'assistente. Questo documento dice solo
cosa misurare e come leggerlo.

**Obiettivo.** Stabilire se i 140 W sulla 3060 spostano i tok/s dei due modelli, e di quanto. È la
prima delle due cose che l'operatore ha messo in fila: **prima** questa, **poi** il kernel planare
dell'Iterazione 7. Una sola variabile: **il limite della 3060, da 120 a 140 W**. Motore invariato,
kernel invariati, cache invariata, nessun'altra manopola.

### Perché questa misura ha senso solo con un numero dichiarato prima

La 3060 è `SwPowerCap` al 73–89% dei campioni attivi: è **l'unica scheda che il limite lo tocca**
(Iterazione 6b). Ma non è l'unica scheda che lavora: le M10 fanno il percorso Maxwell e la 3060 fa
quello denso. Se la 3060 è una frazione piccola del tempo per token, +20 W non sposta nulla e lo si
deve sapere **prima** di leggere il wattmetro, altrimenti si attribuisce al rumore quello che è
aritmetica.

Il profilo dell'Iterazione 6 dà la quota host del roundtrip, non quella della 3060. Quello che si può
dire con i numeri già in mano:

| grandezza | IQ3_S 988 | IQ2_XS | fonte |
|---|---|---|---|
| tok/s | 42,7 | 57,2 | banco I6 |
| tempo per token | 23.400 µs | 17.483 µs | 1e6 / tok/s |
| span misurato sulla scheda | 403 µs/L | 282 µs/L | profilo I6 |
| span host dispatch | 551 µs/L | 377 µs/L | profilo I6 (`MWPROF layer`) |
| **residuo host non attribuito** | **~148 µs/L** | **~95 µs/L** | span host − span scheda |

Il residuo è **l'unico posto in cui può stare la 3060**: lo span misurato sulla scheda è quello delle
M10 (eventi CUDA sulla loro stream), e il `sync-wait` (63 µs/L IQ3_S, 39 IQ2_XS) non lo spiega.
**Ma non è attribuito**: l'Iterazione 6 lo dichiara esplicitamente non separato. Quindi la quota della
3060 è **un'incognita dichiarata**, con un tetto superiore: **al più** tutto il residuo, cioè 148 µs/L
su 553,6 (IQ3_S) e 95 su 377 (IQ2_XS).

Il conto del caso migliore, che è il motivo per cui il protocollo esiste:

- se la 3060 fosse **tutto** il residuo e i 140 W dessero un +10% di clock effettivo sulla parte
  power-limited, si guadagnerebbero 148 · 0,10 ≈ **15 µs/L**, cioè **+2,6%** sui 553,6 µs/L di span
  (IQ3_S da ~43 a ~44 tok/s, IQ2_XS da ~57 a ~58,5);
- se la 3060 fosse metà del residuo, **+1,3%**;
- se i 140 W non dessero clock (plausibile: 140/120 = +17% di budget, ma il clock dipende dal carico e
  dalle tensioni, non linearmente), **0%**.

**Nessuno di questi numeri raggiunge il criterio di 55 tok/s per l'IQ3_S.** Il criterio non è in
discussione in questa finestra, e va detto prima: **questa è una misura di margine, non una leva per
il criterio.** Se anche desse il caso migliore, l'IQ3_S resterebbe a ~44 tok/s.

Dichiarato come **ipotesi**, non come misura: il residuo contiene la 3060, e il suo peso è ignoto.
Il punto (3) del protocollo è lì per ridurre quell'incognita a un numero misurato.

### Il problema che il banco a 3 run non può risolvere

Prima di scrivere i comandi va detto un numero che, se scoperto dopo, avrebbe bruciato la finestra.
La dispersione **run-to-run misurata** del banco città+codice, dai tre banchi IQ2_XS già in repo:

| banco | etichetta | run | media | cv |
|---|---|---|---|---|
| `32-bench-iq2_xs-i6` | city | 51,2 / 54,2 / 54,5 | 53,3 | 3,4% |
| `32-bench-iq2_xs-i6` | code | 62,3 / 60,8 / 60,0 | 61,0 | 1,9% |
| `37-bench-iq2_xs-diag` | city | 54,6 / 55,8 / 57,9 | 56,1 | 3,0% |
| `37-bench-iq2_xs-diag` | code | 65,6 / 55,3 / 60,7 | 60,5 | 8,5% |
| `13-bench-iq2_xs` | city | 53,5 / 58,5 / 58,4 | 56,8 | 5,0% |
| `13-bench-iq2_xs` | code | 59,5 / 60,6 / 59,4 | 59,8 | 1,1% |

**cv pooled = 3,8%.** Con 3 run, l'errore standard della media è 2,2%, e per dichiarare un effetto
con 2 sigma su **due** etichette serve superare **~6,3%**. L'effetto atteso è **1–3%**: il banco a 3 run
**non può vedere l'effetto che questa finestra cerca**. Non è un'opinione, è aritmetica sulla
dispersione già misurata, e va dichiarata prima.

Cosa fare, in ordine di preferenza:

1. **Endpoint primario = il clock.** `clocks.sm` della 3060 è il numero a varianza quasi nulla: se i
   140 W danno clock, il clock **sale e si vede subito** (in I6 era 1789 medio, 1912 max, stabile a
   500 ms). Un salto di clock è la prova del meccanismo, e non ha bisogno del banco.
2. **Endpoint secondario = banco a 6 run per etichetta**, non 3: porta la soglia da 6,3% a **4,4%**.
   Costa ~2× il tempo del banco, che a 600 token e ~50 tok/s è ~12 s per run: **~5 min in più per
   fase**. È poco, e rende la finestra capace di vedere il caso migliore.
3. Se l'operatore vuole tenere 3 run, la finestra si dichiara **inconcludente sul tok/s in partenza** e
   risponde solo sul clock. Meglio saperlo ora.

Il protocollo qui sotto usa **6 run** e il clock come endpoint primario.

### Il banco, e perché A/B nella stessa finestra

`tools/iq3_bench.py` con `--labels city,code`, **6 run per etichetta**, `--max-tokens 600`,
`--no-thinking`, `--vram-gpu 4` (la 3060 è l'indice NVML 4 — verificato dai CSV: è l'unica a 120 W).
Si legge il decode tok/s **dal log del motore**, non dal client. Il clock SM della 3060, campionato a
500 ms per tutta la fase, è **l'endpoint primario**.

L'ordine è **A/B/A**, non A poi B, perché il tempo e la temperatura della stanza cambiano:

```
A1 (120 W)  ->  B (140 W)  ->  A2 (120 W di nuovo)
```

A1 e A2 devono coincidere entro il rumore: se non coincidono, **il rumore è più grande dell'effetto** e
la finestra non conclude (vedi regola di stop). Il confronto utile è **B contro la media di A1 e A2**,
con la differenza A1−A2 come barra d'errore. **Non si confronta con l'Iterazione 6**: quella finestra
aveva il profilo acceso, che costa il 4,7% (dichiarato in I6).

### Le tre fasi, in quest'ordine

1. **Stato e AER prima.** `systemctl is-active strata`, `/root/aer.sh`, e i limiti correnti letti con
   `nvidia-smi --query-gpu=index,power.limit --format=csv` (**sola lettura**: nessun `-pl` da parte mia).
2. **A1, IQ2_XS a 120 W.** Produzione fermata, motore IQ2_XS su con `up.sh`, banco città+codice 3 run,
   `iq3_power.py --deep` in parallelo per tutta la durata. Poi **A1, IQ3_S cache 988** nella stessa
   posizione A: stesso banco, stesso sampler.
3. **B, a 140 W.** L'operatore alza il limite **sulla sola 3060** e lo conferma a voce; io verifico con
   una lettura che il valore sia 140 e che le M10 siano ancora a 30 (se una M10 è cambiata, **stop**:
   sarebbe una seconda variabile). Si ripetono esattamente A1: IQ2_XS poi IQ3_S 988.
4. **A2, di nuovo a 120 W.** L'operatore riporta il limite; verifica della lettura; si ripete il banco.
5. **Produzione.** `down.sh`, `systemctl start strata`, health, AER, limiti riletti e riportati nel LOG.

### Potenza: cosa si campiona accanto al banco

`tools/iq3_power.py --deep --interval-ms 500` per ogni fase, in parallelo al banco (come in 6b, dove ha
funzionato senza disturbare la produzione). Si riportano, per la 3060, **solo i campioni attivi**:

| campo | perché |
|---|---|
| `power.draw` media/max | i 140 W vengono davvero usati? |
| `clocks.sm` media/max | **è qui l'effetto atteso**: se il clock non sale, +20 W non ha dato nulla |
| `SwPowerCap` % | se scende dal 73–89%, il limite morde meno — e questo è il meccanismo |
| `temperature.gpu` max | margine termico ai 140 W |
| `fan.speed` | se la ventola satura, il margine è finito |
| contatore `SW Power Capping` | secondi cumulati: deve **crescere meno** a 140 W |

Se `SwPowerCap` resta al 73–89% **e** il clock SM non sale, la conclusione è che i 140 W non sono la
leva — e si chiude lì, senza Iterazione 8 sulla potenza. Nota: `temperature.memory` è `N/A` su questo
driver, non lo si aspetta.

### Il wattmetro

Lo legge **l'operatore**, alla presa, e lo riporta per fase (A1, B, A2) insieme all'ora di lettura.
Serve a una cosa sola: **verificare che i 140 W siano davvero 140 W alla presa** e che il consumo
totale (9 schede) non abbia margine di alimentatore. Non entra nel confronto tok/s.

### Lettura del risultato

| esito | lettura |
|---|---|
| **clock SM sale** e `SwPowerCap` scende | il limite mordeva: **meccanismo provato**, e si quantifica col resto |
| clock SM piatto dentro il rumore dei 500 ms | **i 140 W non danno clock**: la potenza non è la leva, si chiude |
| banco: B − media(A) > 2 × \|A1 − A2\| | effetto reale sul tok/s, e **quantificato** |
| banco: B dentro 2 × \|A1 − A2\| | **nessun effetto misurabile** a questa risoluzione |
| \|A1 − A2\| grande quanto B − media(A) | **finestra inconcludente**: rumore dominante, si rifà o si chiude |

L'ordine di lettura è deliberato: **prima il clock**, poi il banco. Se il clock non si muove, il banco
non ha nulla da misurare e non serve interpretarlo — è il caso in cui questa finestra costa poco e
chiude una domanda.

In ogni caso **non si promuove l'IQ3_S**: il criterio è 55 tok/s e questa leva, nel caso migliore
dichiarato sopra, ne dà ~1–3%.

### Regola di stop

- Se una **M10** cambia limite insieme alla 3060: **stop immediato** (seconda variabile).
- Se la 3060 supera **80 °C** o la ventola va al **100%** a 140 W: si riporta a 120 W e si chiude.
- Se A1 e A2 non coincidono entro il 2%: la finestra è inconcludente, non si fanno altre fasi.
- La produzione si ferma solo con l'OK dell'operatore, e si ripristina **prima** di qualunque altra cosa.

### Cosa questa finestra non fa

Non tocca i kernel (il planare è la fase **dopo**, separata). Non tocca la cache 988. Non tocca i limiti
delle M10, che restano 30 W. **Non accende la profilazione** (`STRATA_MW_LAYERPROF` resta spenta: il suo
4,7% sarebbe una seconda variabile). Non cambia il criterio e non promuove l'IQ3_S. Non spiega il crollo
banco→soak. **Non è una leva per i 55 tok/s**: è la misura che dice se la potenza della 3060 conta
qualcosa, prima di spendere il banco sul kernel.

### Comandi per aidrogen (finestra, dopo l'OK dell'operatore)

**I config.** Non si riusano quelli dell'Iterazione 6: hanno `STRATA_MW_LAYERPROF=1` nell'`env`, che
costa il **4,7%** di decode (misurato in I6) e qui sarebbe una seconda variabile. Si creano due config
**senza** profilazione, partendo dagli stessi sorgenti e togliendo la chiave:

```sh
PY=/root/Strata-maxwell/.venv/bin/python
$PY - <<'PY'
import json
for src, dst in (("strata-iq3_s-i3.json",   "strata-iq3_s-i7w.json"),
                 ("strata-iq2-xs-i3-after.json", "strata-iq2-xs-i7w.json")):
    c = json.load(open("/root/Strata-maxwell/" + src))
    c["env"].pop("STRATA_MW_LAYERPROF", None)      # one variable: no profile overhead
    c["log"] = dst.replace(".json", ".log")
    a = c["args"]
    if "iq3_s" in dst:
        a[a.index("--expert-cache") + 1] = "988"  # the measured reference config
    json.dump(c, open("/root/Strata-maxwell/" + dst, "w"), indent=1)
    print(dst, "->", c["log"], "cache", a[a.index("--expert-cache") + 1],
          "prof", "STRATA_MW_LAYERPROF" in c["env"])
PY

CFG3S=/root/Strata-maxwell/strata-iq3_s-i7w.json
CFG2=/root/Strata-maxwell/strata-iq2-xs-i7w.json

# --- 0. stato, limiti (SOLA LETTURA), AER ---
systemctl is-active strata
nvidia-smi --query-gpu=index,power.limit,power.default_limit,power.max_limit --format=csv
bash /root/aer.sh

# --- 1. A1 @120 W: IQ2_XS, poi IQ3_S 988 ---
systemctl stop strata
bash /root/down.sh
bash /root/up.sh "$CFG2" 1800
$PY /root/Strata-iq3/tools/iq3_power.py --deep --label a1-iq2_xs --minutes 8 \
    --interval-ms 500 --out /root/i7-a1-power-iq2_xs.csv &
$PY /root/Strata-iq3/tools/iq3_bench.py --label iq2_xs --runs 6 --labels city,code \
    --no-thinking --vram-gpu 4 --engine-log /root/Strata-maxwell/strata-iq2-xs-i7w.log \
    --api-key-json "$CFG2" --json /root/i7-a1-bench-iq2_xs.json
wait
bash /root/down.sh
bash /root/up.sh "$CFG3S" 1800
$PY /root/Strata-iq3/tools/iq3_power.py --deep --label a1-iq3_s --minutes 8 \
    --interval-ms 500 --out /root/i7-a1-power-iq3_s.csv &
$PY /root/Strata-iq3/tools/iq3_bench.py --label iq3_s --runs 6 --labels city,code \
    --no-thinking --vram-gpu 4 --engine-log /root/Strata-maxwell/strata-iq3_s-i7w.log \
    --api-key-json "$CFG3S" --json /root/i7-a1-bench-iq3_s.json
wait
# AER dopo A1

# --- 2. B @140 W (l'operatore alza SOLO la 3060), poi identico ad A1 ---
#   nvidia-smi -i 4 -pl 140     <-- LO FA L'OPERATORE, non l'assistente
nvidia-smi --query-gpu=index,power.limit --format=csv     # deve dire 140 sulla 4, 30 sulle altre
#   ... stessi quattro blocchi di A1 con --label b-... e --out /root/i7-b-...
# AER dopo B

# --- 3. A2 @120 W (l'operatore riporta), poi identico ad A1 ---
#   nvidia-smi -i 4 -pl 120     <-- LO FA L'OPERATORE
nvidia-smi --query-gpu=index,power.limit --format=csv     # deve dire 120 sulla 4, 30 sulle altre
#   ... stessi quattro blocchi con --label a2-... e --out /root/i7-a2-...
# AER dopo A2

# --- 4. produzione ---
bash /root/down.sh
systemctl start strata
systemctl is-active strata && curl -sf http://127.0.0.1:8080/health && echo
nvidia-smi --query-gpu=index,power.limit --format=csv     # 120 sulla 4, 30 sulle altre
# AER dopo il ripristino
```

**Nota.** A2 esiste per una ragione sola: senza, il confronto A/B sarebbe contro un numero di
un'altra ora. Con A1 e A2 si ha la barra d'errore della stessa finestra.

**Nota sul sampler.** `iq3_power.py` ha `--minutes` fisso: non sa quando il banco finisce. Con **6 run
per etichetta** su due etichette il banco dura ~4–5 min a modello (dipende dai tok/s), quindi i 8
minuti coprono la fase con margine. Se il banco finisse prima, i campioni in più sono a scheda quasi
idle e **non entrano nel riepilogo** (`active_only`, come in 6b). Se durasse di più, il sampler taglia
la coda: il clock viene dal **CSV di potenza**, non dal log del motore, quindi in quel caso si rifà
**solo il sampler** sulla fase già finita (costa minuti, non tocca il banco) oppure si accetta il buco
e lo si dichiara.

Il tempo totale stimato: A1 + B + A2 = 3 fasi × 2 modelli × (load + ~5 min di banco) ≈ **50–60 minuti**,
più i cambi di limite dell'operatore. Se l'operatore vuole accorciare, si taglia prima A2 (dichiarando
che il confronto perde la barra d'errore interna), **non** i run: 3 run rendono la finestra cieca
sull'endpoint del tok/s (vedi sopra).

---