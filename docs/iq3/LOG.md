# IQ3 — Qwen3.8-Flash-Next IQ3_S sul tier Maxwell (Tesla M10)

Questo LOG registra **una voce per iterazione**: obiettivo, ipotesi, modifiche, comandi esatti per
aidrogen, risultato atteso e risultato misurato. Le voci sono in italiano; codice, commenti e messaggi
di commit sono in inglese, nello stile della repo.

Regole: una sola variabile per iterazione; niente numeri dichiarati senza misura; il percorso IQ2_XS
di produzione resta invariato; nessun `api_key` nella repo (i config del server restano fuori).

Indice:

- [Iterazione 0 — base `iq3` e baseline IQ2_XS](#iterazione-0--base-iq3-e-baseline-iq2_xs)
- [Iterazione 0b — `STRATA_MW_FASTCOMPLETE=1` (esperimento separato)](#iterazione-0b--strata_mw_fastcomplete1-esperimento-separato)

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
- AER: 0 durante il test.

**Risultato misurato.** (2026-11-07, aidrogen; output completo in `docs/iq3/results/00-baseline.txt`)

```
data:                         2026-11-07
8 Maxwell tier(s)?            SI
gate PASSED su tutti gli 8?   SI (8/8, worst rel 1.12e-02 .. 1.26e-02)
esperti e GiB per die:        2838 (CUDA1-6), 2837 (CUDA7-8); 3.80-3.82 GiB
slot 3060 / MiB liberi:       1874 slot / 2246 MiB
prefill (tok/s):              173.8 / 178.6 / 178.7   (media 177.0)
decode (tok/s):               dipende dal prompt, vedi sotto
AER:                          2 correctable, entrambi sul PCIe switch PLX 8747 (09:11.0), 0 sulle GPU
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
dall'accettazione delle draft, quindi dal testo generato: nel run di riferimento del 6 novembre
(1801 tok, 68.6 tok/s) le draft accettate erano **1273 su 1687 (75,5%)**; con i prompt di oggi
sono **727 su 1007 (72,2%)**. I run brevi da 300-600 token danno 45-49 tok/s **anche sul motore
0.1.47**, quindi il dato storico non è riproducibile senza la richiesta esatta del PI agent.
Il valore di riferimento per le iterazioni successive è la colonna 0.1.51 di questa tabella.

**Nota di metodo sulla build.** Il checkout di produzione (`/root/Strata-maxwell`, branch `mw8`,
motore 0.1.47) è stato lasciato intatto: la build è stata fatta in un worktree separato
`/root/Strata-iq3` (detached su `8b15183`), con `third_party/llama.cpp` in symlink alla copia
già presente. La `llama.cpp` sul server corrisponde esattamente al pin `3cf03257` richiesto da
0.1.51 (hash della regione `mmq.cuh` identico). Build pulita in ~10 min, 18 warning (tutti
preesistenti: `-Wrestrict`, `-Wunused-variable`, `-Wsign-compare`), 0 errori.

→ Ipotesi: **CONFERMATA** sulla capacità (8 tier, 8 gate, tempi di caricamento, prefill nel
range). **SMENTITA** nella forma "decode 65-68,7 entro il rumore": il decode è ~47-55 tok/s con
i prompt usati, su **entrambi** i motori. La causa è l'accettazione delle draft, non il codice.

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