> Historical document: the brief the campaign started from on 2026-11-07, kept unchanged. Some premises were superseded during the work; the LOG is authoritative. The operator named BitMakerMan is Craicek.

# BRIEF — Qwen3.8-Flash-Next IQ3_S sul tier Maxwell (Tesla M10) di Strata-maxwell

Repo di lavoro: https://github.com/BitMakerMan/Strata-maxwell (fork di nuke1984/Strata-maxwell, `main` = 6175fa0)
Proprietario e operatore: BitMakerMan.
Lingua: questo brief e il LOG sono in italiano. Codice, commenti e messaggi di commit sono in inglese, nello stile della repo.

---

## 0. Ruoli: la triangolazione

| Chi | Cosa fa | Cosa non fa |
|---|---|---|
| **Agente di coding (tu)** | Legge il codice, scrive le modifiche, compila se può, fa commit e push sul branch di lavoro, scrive il LOG di ogni iterazione | Non dichiara risultati che non ha misurato. Non tocca la produzione |
| **BitMakerMan (operatore)** | Fa `git pull` sul server aidrogen, compila, lancia i test, riporta gli output (senza chiavi API) | — |
| **Claude (revisore)** | Legge la repo aggiornata e i risultati, verifica le ipotesi, propone il passo successivo | Non fa push |

**Ciclo di ogni iterazione:**
1. **Agente:** fa **una** modifica con uno scopo solo, poi commit e push sul branch `iq3`. Aggiunge una voce in `docs/iq3/LOG.md` con: obiettivo, file toccati, ipotesi, **comandi esatti da lanciare su aidrogen** e risultato atteso.
2. **Operatore:** lancia i comandi su aidrogen e salva gli output in `docs/iq3/results/<NN>-<nome>.txt`, oppure li passa all'agente.
3. **Revisore:** legge diff e risultati e conferma o smentisce l'ipotesi.
4. Si passa all'iterazione successiva **solo** se quella attuale è chiusa.

---

## 1. Regole di metodo (non negoziabili)

1. **Metodo scientifico.** Ogni affermazione è **misurata** o marcata come **ipotesi**. Una lettura del codice si dichiara come "analisi statica, non confermata".
2. **Una variabile alla volta.** Ogni iterazione ha uno scopo solo. Niente refactoring "già che ci sono".
3. **Il percorso IQ2_XS di produzione deve restare identico bit per bit.** Le nuove strade si attivano **solo** per i formati nuovi, e il controllo all'avvio (*guard*) si allenta **solo** per le combinazioni che hanno superato il test di parità.
4. **Il guard esiste per un motivo.** Un formato letto con la geometria sbagliata produce *righe sbagliate ma plausibili, in silenzio* (commento in `src/core/maxwell_experts.cpp`, prima del controllo `gu_type`/`d_type`). Si allenta solo dopo la parità.
5. **Nessun segreto nella repo.** I config del server contengono `api_key`: non si committano mai, e nei risultati la chiave si sostituisce con `***`.
6. **Si seguono `AGENTS.md` e `CONTRIBUTING.md`** del fork, sia per lo stile sia per l'obbligo di misura. Le PR verso upstream sono un passo **eventuale e finale**.

---

## 2. L'hardware (aidrogen)

- **HP Z440:** Xeon E5-1620 v4 (4 core e 8 thread, AVX2, niente AVX-512), Proxmox VE 9 (kernel 6.17.13-21-pve). Strata gira nel **container LXC 101** (Debian 12, privilegiato) con **24 GB di RAM** e 220 GB su SSD NVMe. I modelli di riserva sono su un HDD montato in `/mnt/backup`.
- **GPU** (driver **580.196.07**, l'ultimo ramo con supporto Maxwell; **CUDA 12.9**, perché CUDA 13 non supporta più sm_50):
  - **RTX 3060 12 GB** (sm_86): `nvidia-smi` indice 4, bus 07:00.0. È il dispositivo principale, limitato a 120 W.
  - **2× Tesla M10**, cioè **8 die GM107** (sm_50) da **8 GB** ciascuno: `nvidia-smi` indici 0–3 e 5–8, limite 30 W per die. Gli slot sono **forzati a PCIe Gen1** per un problema di compatibilità elettromagnetica (errori AER risolti così). La sonda del motore misura comunque 11.3 GB/s host→device.
- **Ordine dei dispositivi nel motore:** `CUDA_DEVICE_ORDER=PCI_BUS_ID`, `CUDA_VISIBLE_DEVICES=4,0,1,2,3,5,6,7,8`. Così CUDA0 è la 3060, CUDA1–8 sono le M10, e `--maxwell-devices 1,2,3,4,5,6,7,8`.

---

## 3. Cosa abbiamo già fatto (stato misurato)

### 3.1 La patch `mw8`: fino a 8 dispositivi Maxwell
Sul server, branch `mw8`, commit `892d8df0`, basato su un checkout **più vecchio** di `main` (il motore riporta 0.1.47, mentre `main` è a 0.2.6 con motore 0.1.51). Il file della patch è `0001-mw8-up-to-8-Maxwell-devices-arrays-prefill-tier-cap-.patch` (6887 byte, generato con `git format-patch`). L'operatore lo copierà nella repo. Le modifiche:

- **`include/strata/core/expert_source.hpp`:** `maxwell[4]`→`[8]`, `mw_partial[4]`→`[8]`.
- **`src/program/generate.cpp`:**
  - `std::array<strata::core::MaxwellExperts, 4>`→`8`;
  - `maxwell_devs.size() > 4`→`> 8`, e il messaggio "1-4"→"1-8";
  - dopo `const bool e1_on = ...`, un guard: se `e1_on && maxwell_count > 4`, errore "at most 4 Maxwell devices" e `return 1`.
- **`src/core/expert_source.cpp`:** `int order[4]`→`int order[8] = {0,1,2,3,4,5,6,7}`, `int64_t cnt[4]`→`cnt[8] = {}`.
- **`src/prefill/prefill.cpp`:**
  - `core::MaxwellExperts* mw[4]`→`[8]`;
  - in `set_maxwell`, `if (count > 4) count = 4;`→`8`;
  - dopo `if (m.mw_count == 0) return true;`, se `mw_count != 4` e `STRATA_MW_DRED` è impostata, stampa un warning e fa `unsetenv("STRATA_MW_DRED")`.
- **Restano a 4 apposta,** perché servono solo alla modalità tensor: `maxwell_meta.cpp` `rank`, `h_tp[4]`, il loop `m<4` in `generate.cpp` (circa riga 9493 nel nostro checkout).

### 3.2 La produzione attuale: IQ2_XS, 3060 + 8 die M10
- **Motore:** `engine-mw8`.
- **Argomenti:** `--expert-cache 1800 --prefill auto --spec 6 --spec-min-p 0.5 --mtp <mtp/rt> --max-context 128000 --kv k8v4 --mmap-experts --maxwell-devices 1,2,3,4,5,6,7,8 --maxwell-reserve-mib 250 --spec-split`.
- **Variabili d'ambiente:** `STRATA_EMBED_VRAM=1 STRATA_MW_MT=1 STRATA_MW_MT_T=256x4 STRATA_MW_NTM=256 STRATA_MW_TWOPHASE=1 STRATA_MW_AD2H=1`. Nota: **`STRATA_MW_FASTCOMPLETE=1` non è attiva** e non sappiamo se il nostro checkout vecchio ce l'abbia.
- **Misure del 5–6 novembre:**
  - prefill **181 tok/s**;
  - decode **65–68.7 tok/s** (68.7 su 1801 token da PI agent);
  - ogni die M10 ha circa 2838 expert, **3.81 GiB**, round-trip gate **PASSED**;
  - la 3060 ha 1874 slot. 1874 + 22702 = 24576, cioè **tutti gli expert sono in VRAM**;
  - restano 2246 MiB liberi sulla 3060;
  - soak di 10 minuti stabile, AER 0.
- **Output a temperatura 0 non identico bit per bit** già prima di qualsiasi modifica, anche con un solo die. Ipotesi: tempistiche delle ottimizzazioni asincrone. Quindi **la qualità si verifica con i test di parità dei kernel**, non confrontando le risposte.

### 3.3 Perché IQ3_S non parte (misurato il 6 novembre)
Il motore si ferma in `MaxwellExperts::open` con questo messaggio:
```
MaxwellExperts (CUDA1): layer 0 keeps its experts as gate/up ggml type 18 and down type 20;
this tier takes Q2_0 (42), and IQ2_S (22) / IQ2_XXS (16) / IQ1_M (29) gate/up over a Q2_0 down
```
Il guard è un `return false` fatale e controlla tutti i layer. Non c'è una modalità che salti il singolo layer.

**Formati di IQ3_S visti nell'output di `tools/iq_pack.py`.** È un campione: il censimento completo è il compito dell'iterazione 1.

| layer | gate/up | down | byte del blob per expert |
|---|---|---|---|
| 0, 16 | IQ3_XXS (18) | IQ4_NL (20) | 2 176 000 |
| 8 | IQ2_S (22) | Q2_0 (42) | 1 510 400 |
| 24, 32 | IQ2_S (22) | IQ4_NL (20) | 1 971 200 |
| 40 | IQ3_S (21) | IQ4_NL (20) | 2 329 600 |

`experts.bin` di IQ3_S pesa **46.84 GiB** (48 layer), contro i 35.45 GB di IQ2_XS. Il blob IQ2_XS pesa 1 510 400, 1 305 600 o 1 177 600 byte.
**Stima della VRAM:** tolta la quota della 3060, restano circa 5.5 GiB per die, contro gli 8 disponibili. Dovrebbe entrare, ma è un'ipotesi da misurare.

**Riferimento dei tipi ggml:** 16 IQ2_XXS · 18 IQ3_XXS · 20 IQ4_NL · 21 IQ3_S · 22 IQ2_S · 29 IQ1_M · 42 Q2_0 (lo "S2" del tier).

### 3.4 Cosa c'è già nel codice (analisi statica su `main` 6175fa0)
- **`src/kernels/cuda/expert_maxwell.cu` (2022 righe):**
  - `dp4a_s32` è **emulata su sm_50** con `vmad` (righe 56–64), quindi i prodotti scalari IQ in stile llama.cpp si portano così come sono.
  - I formati sono descritti da `template <int TY> struct IqFmt` con specializzazioni solo per 22, 16 e 29 (righe 970–994): `qk`, `ipb`, `step`, `grid_bytes`, `grid_dev()`, `dot()`.
  - `IqV2qSmem` (circa riga 1020) dimensiona la shared memory con `grid = (TY == 16 ? 2048 : 8192)`.
  - Gli `switch` sui tipi sono alle righe circa 1276, 1626 e 1956. C'è anche una variante multi-token, `IqMt` (circa riga 1666), e un kernel persistente (`maxwell_persist_kernel`, circa riga 1216).
  - **Il down oggi è solo S2 (Q2_0):** il kernel dello stadio 2 (`expert_s2_stage2_*`) e l'epilogo dello stadio 1 producono `faq`/`fasc`/`fahx` nel formato S2.
- **`src/kernels/cuda/iq_kernels.cu`** (il percorso della 3060) ha già `vec_dot_iq3_xxs_q8_1` (circa riga 197), `vec_dot_iq3_s_q8_1` (circa 223) e `vec_dot_iq4_nl_q8_1` (circa 283), con `Fmt<18>`, `Fmt<20>` e `Fmt<21>` (circa righe 722–736).
- **`src/core/maxwell_experts.cpp` (1497 righe):** il guard è intorno alla riga 170. Circa alla riga 281 c'è `if (f.gu_type != 42)`, dove il down viene convertito in S2 al caricamento. Circa alla riga 553 c'è la mappa `ti`, che decide l'indice del kernel dal `gu_type`.
- **`src/kernels/cuda/expert_maxwell_iq_parity.cpp`** è lo strumento di parità: confronta il kernel M10 con i prodotti scalari di produzione della 3060, con soglia 1e-5 relativa, più il controllo dell'act image bit per bit e dell'epilogo. Ha anche la modalità `--synthetic <gu>/<down>`, con pesi casuali quantizzati da ggml. Il target CMake `maxwell_iq_parity` è a circa riga 2193 di `CMakeLists.txt`.

---

## 4. Obiettivo e criteri di accettazione

**Obiettivo:** far girare Qwen3.8-Flash-Next **IQ3_S** (ISTA-DASLab GSQ-RCO) sulla 3060 + 8 die M10, con gli expert sulle M10, **senza toccare** il percorso IQ2_XS.

**Accettazione (tutto misurato su aidrogen):**
1. `maxwell_iq_parity` **PASSED** per ogni combinazione gate/up e down presente in IQ3_S, sia con `--synthetic` sia sulle righe reali dello shard 1.
2. Il motore carica IQ3_S con "8 Maxwell tier(s)" e round-trip gate **PASSED** su tutti gli 8 die.
3. **Decode ≥ 55 tok/s** con PI agent / API, e prefill misurato con lo stesso protocollo (prompt da circa 2.2k token).
4. Soak di 10 minuti senza errori. VRAM per die e AER sotto controllo.
5. **IQ2_XS**, ricompilato con il nuovo codice, ha velocità e round-trip gate invariati rispetto alla baseline dell'iterazione 0.
6. Se il criterio 3 fallisce: si documenta perché (M10 limitate dalla banda di memoria o CPU limitata dal single-thread) e si valuta IQ3_XXS. Il lavoro resta comunque utile.

---

## 5. Le iterazioni

### Iterazione 0: base della repo e baseline
1. Crea il branch `iq3` da `main` (6175fa0).
2. Applica la patch `mw8` (`git am`). Se `main` è cambiato rispetto al nostro checkout vecchio, risolvi i conflitti **seguendo la descrizione del §3.1**, ricontrolla con `grep` ogni `[4]` e `> 4` che riguarda Maxwell, e documenta nel LOG cosa è cambiato.
3. Crea `docs/iq3/LOG.md` e `docs/iq3/results/`, poi `.gitignore` per `strata*.json` (i config contengono la chiave API).
4. Scrivi nel LOG i comandi per aidrogen:
   - compilazione nella cartella `build-iq3` (vedi §6);
   - avvio di **IQ2_XS** con il config di produzione e il nuovo motore;
   - misura del decode e del prefill.
   Questa è **la nuova baseline**, perché `main` è più recente del nostro motore 0.1.47.
5. Solo dopo, come esperimento separato con una variabile sola, prova `STRATA_MW_FASTCOMPLETE=1` (secondo l'autore vale +2.8% di decode).

### Iterazione 1: censimento dei formati
Scrivi `tools/mw_layer_census.py`, usando il `gguf-py` incluso in `third_party/llama.cpp/gguf-py` o `tools/gguf_reader.py`. Per ogni layer deve stampare: tipo ggml di `ffn_gate_exps`, `ffn_up_exps` e `ffn_down_exps`, le forme, i byte per expert, e il **totale per combinazione**.
L'operatore lo lancia sullo shard 1 di IQ3_S (`/root/Strata-data/models/IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf`) e committa l'output in `docs/iq3/results/01-census.txt`.
**Da qui esce l'elenco esatto dei formati da implementare.** Ipotesi da verificare: gate/up ∈ {18, 21, 22}, down ∈ {20, 42}, forse altri.

### Iterazione 2: gate/up IQ3_XXS (18) e IQ3_S (21) sulle M10
- Porta `vec_dot_iq3_xxs_q8_1` e `vec_dot_iq3_s_q8_1` in `expert_maxwell.cu` usando `dp4a_s32`, con le griglie `iq3xxs_grid` e `iq3s_grid` disponibili su sm_50.
- Aggiungi `IqFmt<18>` e `IqFmt<21>`, e correggi `IqV2qSmem::grid`, che oggi è scritta a mano per 16 contro "tutto il resto"; meglio derivarla da `IqFmt<TY>::grid_bytes`.
- Aggiungi i `case` in **tutti** gli `switch`: il batch v2q, il multi-token `IqMt` e il persistente, oltre alla mappa `ti` in `maxwell_experts.cpp`.
- Estendi `expert_maxwell_iq_parity` per i formati nuovi.
- Test: `--synthetic iq3_xxs/q2_0` e `--synthetic iq3_s/q2_0`. Sono combinazioni artificiali che isolano il solo gate/up.
- **Il guard non si tocca ancora.**

### Iterazione 3: down IQ4_NL (20) nativo sulle M10 (il cuore del progetto)
- Oggi il down è sempre S2: viene convertito al caricamento e passa per l'epilogo S2. IQ4_NL **non** va convertito in S2, perché si perderebbe la qualità che motiva tutto il progetto.
- Progetta il percorso: le righe del down IQ4_NL restano native nel blob; l'epilogo dello stadio 1 quantizza l'uscita SwiGLU in **q8_1** (come fa il percorso della 3060); un **nuovo kernel per lo stadio 2** fa il prodotto con `vec_dot_iq4_nl_q8_1`.
- Mantieni la stessa geometria a batch e lo stesso ordine di somma della 3060, dove serve per la parità.
- In `maxwell_experts.cpp` aggiorna la disposizione del blob (offset e stride del down per layer) e la scelta del kernel per layer secondo `d_type`.
- Estendi lo strumento di parità con `--synthetic iq3_xxs/iq4_nl`, `iq2_s/iq4_nl` e `iq3_s/iq4_nl`.
- **Prima di scrivere codice**, metti nel LOG il progetto (la struttura dati e i kernel), così il revisore lo può valutare.

### Iterazione 4: parità sulle righe reali e allentamento del guard
- `maxwell_iq_parity <shard1 IQ3_S> <layer ...>`, con almeno un layer per ogni combinazione del censimento.
- Allenta il guard **aggiungendo solo** le combinazioni verificate. Il messaggio di rifiuto resta per tutto il resto.
- Il round-trip gate al caricamento deve coprire anche i formati nuovi: oggi campiona iq2_s, iq2_xxs e iq1_m.

### Iterazione 5: integrazione e misure
- Config IQ3_S con 8 M10. Base: `strata-iq3_s-8mw.json`, che c'è già sul server (`--expert-cache auto` per partire, `--max-context 128000`).
- Verifica dal log: "8 Maxwell tier(s)", gate PASSED, numero di expert e GiB per die, slot della 3060, VRAM libera.
- Misura il decode (PI agent e API) e il prefill (`/root/prefill_test.py`: 3 prompt da circa 2220 token), poi soak di 10 minuti.
- Verifica che IQ2_XS non sia regredito (criterio 5).
- Taratura successiva, una variabile alla volta: `--expert-cache`, `--maxwell-reserve-mib`, `STRATA_MW_*`.

### Iterazione 6: documentazione
`docs/iq3/README.md` con il risultato, i numeri, i limiti e come riprodurlo. Poi, se ha senso, una PR verso nuke1984 secondo `CONTRIBUTING.md`, con titolo `[Engine]`, misure e limiti dichiarati.

---

## 6. Compilazione e test su aidrogen (container LXC 101, `/root/Strata-maxwell`)

```bash
cd /root/Strata-maxwell
git fetch https://github.com/BitMakerMan/Strata-maxwell iq3 && git checkout -B iq3 FETCH_HEAD
export PATH=/opt/cmake-venv/bin:$PATH   # OBBLIGATORIO: il cmake 3.25.1 di sistema non supporta il dialetto CUDA20 con nvcc 12.9 (misurato)
cmake -S . -B build-iq3 -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.9/bin/nvcc -DCMAKE_CUDA_ARCHITECTURES=86 \
  -DSTRATA_ENABLE_CUDA=ON -DSTRATA_EXPERIMENTAL_SM60=ON \
  -DSTRATA_GGML_DIR=/root/Strata-maxwell/third_party/llama.cpp \
  -DSTRATA_NATIVE_EXPERTS=ON -DSTRATA_Q6K_EXPERTS=ON -DSTRATA_MMQ_KQUANTS=ON \
  -DSTRATA_MMQ_TQ_OVERLAY=ON -DGGML_STATIC=ON
cmake --build build-iq3 -j 8 --target strata
mkdir -p engine-iq3 && cp build-iq3/strata build-iq3/libstrata_kq.so engine-iq3/
```

- **Cosa sappiamo:** con `STRATA_BUILD_TESTS` attivo alcuni target di test **non linkano** (`cuStreamWriteValue32_v2`: manca la libreria del driver CUDA, `-lcuda` / `CUDA::cuda_driver`). Per compilare `maxwell_iq_parity` può servire sistemare il link in `CMakeLists.txt`. Fallo con un commit separato e documentalo.
- **Lo strumento di parità vuole tutti i dispositivi visibili:** si lancia **senza** `CUDA_VISIBLE_DEVICES`.
- **La produzione non si tocca:** il servizio systemd `strata` usa `engine-mw8` e il config IQ2_XS. I test girano in tmux, dopo `systemctl stop strata`. Alla fine si riavvia la produzione.
- **Percorsi:**
  - IQ3_S: `/root/Strata-data/models/IQ3_S/*.gguf` e il pack `/root/Strata-data/packs/iq3_s`;
  - IQ2_XS: lo shard 1 e il pack su `/mnt/backup/strata-iq2_xs/`;
  - MTP: `/root/Strata-data/mtp/rt`;
  - profilo: `data/expert-profile.bin`.
- **Il caricamento dall'HDD dura circa 13 minuti**, contro i 4–6 dall'SSD. Gli shard IQ3_S sono sull'SSD.

---

## 7. Formato di ogni voce del LOG (`docs/iq3/LOG.md`)

```
## Iterazione N — <titolo>   (commit <hash>)
Obiettivo: ...
Ipotesi: ...  (cosa ci aspettiamo e perché)
Modifiche: file:funzione — cosa e perché
Percorso IQ2_XS: invariato? come lo garantisco
Comandi per aidrogen: (blocco copiabile)
Risultato atteso: ... / Criterio di successo: ...
Risultato misurato: (lo compila l'operatore o il revisore) → ipotesi CONFERMATA / SMENTITA
```

## 8. Da non fare
- Togliere o indebolire il guard per "vedere se va".
- Convertire IQ4_NL o IQ3 in S2 (Q2_0): riduce la qualità e fa perdere il senso del progetto.
- Modificare i default del percorso IQ2_XS, o il comportamento in modalità tensor (`tp_cards_ > 1`), che resta fuori dal progetto.
- Dichiarare velocità o parità senza averle misurate.
- Committare config, log o output che contengano `api_key`. 
