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
# Individua il config di produzione IQ2_XS usato dal servizio:
systemctl cat strata | grep -E 'ExecStart|--config'
# Poi impostane il percorso qui:
CFG_IN=/root/Strata-maxwell/strata-maxwell.json        # <-- correggere se il nome è diverso
python3 - "$CFG_IN" <<'PY'
import json, sys
src, dst = sys.argv[1], "/root/Strata-maxwell/strata-iq2-xs-baseline.json"
c = json.load(open(src))
c["exe"] = "engine-iq3/strata"
c["log"] = "strata-iq2-xs-baseline.log"
with open(dst, "w") as f:
    json.dump(c, f, indent=2)
print("wrote", dst)
print("exe:", c["exe"])
print("maxwell:", [a for a in c["args"] if a.startswith("--maxwell")])
PY
# Atteso: exe engine-iq3/strata e --maxwell-devices 1,2,3,4,5,6,7,8

# --- 4. avvia il motore nuovo in tmux (NON usare il servizio systemd) ---
tmux new -s iq3-baseline
cd /root/Strata-maxwell
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=4,0,1,2,3,5,6,7,8
export STRATA_EMBED_VRAM=1
export STRATA_MW_MT=1
export STRATA_MW_MT_T=256x4
export STRATA_MW_NTM=256
export STRATA_MW_TWOPHASE=1
export STRATA_MW_AD2H=1
# NON impostare STRATA_MW_FASTCOMPLETE in questa baseline (è l'esperimento 0b)
python3 serve/server.py --engine strata --config strata-iq2-xs-baseline.json --port 8088
```

Da un'altra shell, attendere il caricamento (4–6 minuti dall'SSD) e verificare il tier:

```sh
curl -sf http://127.0.0.1:8088/health && echo " health OK"
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

**Risultato misurato.** *(da compilare da parte dell'operatore o del revisore)*

```
data:
8 Maxwell tier(s)?            SI / NO
gate PASSED su tutti gli 8?   SI / NO
esperti e GiB per die:
slot 3060 / MiB liberi:
prefill (tok/s):
decode (tok/s):
AER:
note:
```

→ Ipotesi: **CONFERMATA / SMENTITA** (da decidere sul risultato sopra).

---

## Iterazione 0b — `STRATA_MW_FASTCOMPLETE=1` (esperimento separato)

**Obiettivo.** Dopo che la baseline 0a è chiusa, misurare l'unica variabile
`STRATA_MW_FASTCOMPLETE=1` sullo stesso motore e con lo stesso protocollo.

**Ipotesi.** Il fork riporta +2,8% di decode con i quattro complete concorrenti; su 8 die il guadagno
può cambiare. I gate di qualità e il round-trip devono restare identici: le righe delle carte sono
disgiunte e i thread si uniscono prima dell'upload.

**Modifiche.** Nessuna al codice: solo la variabile d'ambiente nel lancio.

**Comandi per aidrogen.** Ripetere il passo 4 dell'iterazione 0 aggiungendo:

```sh
export STRATA_MW_FASTCOMPLETE=1
```

e usare un nome di log distinto, per esempio `strata-iq2-xs-baseline-fastc.log`. Stesse misure,
stesso protocollo, poi salvare in `docs/iq3/results/00b-fastcomplete.txt`.

**Risultato atteso / criterio di successo.** Decode ≥ baseline 0a; prefill e gate invariati. Se il
decode peggiora o un gate cambia, la variabile non entra nella produzione.

**Risultato misurato.** *(da compilare)* → Ipotesi: **CONFERMATA / SMENTITA**.