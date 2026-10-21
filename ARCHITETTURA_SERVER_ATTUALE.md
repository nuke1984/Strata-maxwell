# Architettura del server attuale e percorso del token

**Rilevazione:** 19 ottobre 2026.

**Aggiornamento successivo della stessa giornata:** è stato attivato il test
`STRATA_EMBED_VRAM=1`, con configurazione locale `strata-iq2xs-embed-vram.json`.
La tabella token embedding è ora nella VRAM della 3060 (322,07 MiB); distribuzione
degli esperti, contesto 256k, KV k8v4 e spec 4 sono invariati. Il corpo del documento
conserva la fotografia precedente al test, quando l'embedding era host-mapped.
Verifiche e misure del cambiamento sono in
[`bench/results/2026-10-19-embedding-vram/README.md`](bench/results/2026-10-19-embedding-vram/README.md).

Documento ricostruito dal processo attivo, dal suo ambiente, dal log dell'ultimo avvio, dai metadati GGUF e dal codice locale. Durante l'analisi il server non è stato modificato né riavviato. I numeri descrivono la configurazione rilevata, non necessariamente avvii successivi.

> Oggi la 3060 esegue il percorso principale del modello; le quattro M10 sono un insieme di acceleratori che eseguono esperti MoE interi. Non costituiscono una GPU virtuale tensor-parallel.

## 1. Il server attivo

Ci sono due processi:

- **Server Python**, PID `87067`: espone l'API sulla porta **8088**, prepara le richieste, gestisce tokenizzazione e risposte.
- **Motore C++/CUDA**, PID `87071`: mantiene il modello caricato ed esegue l'inferenza.

La configurazione è `/tmp/opencode/spec-sweep.json`; il binario è `/home/toor/Strata/engine/strata`.

| Impostazione | Valore |
|---|---|
| Motore | Strata **0.1.45** |
| Modello | `Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS` |
| Contesto configurato | **256.000 token** |
| KV cache | **k8v4** |
| Speculazione | **4** |
| Soglia delle proposte MTP | **0,5** |
| GPU principale | RTX 3060, device **0** |
| Tier esperti Maxwell | M10, device **1–4** |

Nel processo non sono attivi `--maxwell-split tensor`, `STRATA_MW_GPUFLAG` o `--pipeline-windows`.

L'ambiente del motore contiene `CUDA_VISIBLE_DEVICES=0,1,2,3,4` e `STRATA_SPEC_DEPTH=1`. `/health` conferma che il modello è caricato e il server è operativo.

## 2. Come è fatto il modello

È un modello **ibrido**, non una semplice successione di layer di attenzione tradizionale.

### Struttura principale

- **48 layer**.
- Vettore nascosto di **2.560 elementi**.
- Vocabolario di **248.320 token**.
- **Quattro flussi residuali**, gestiti dalle *hyper-connections*.

I layer seguono questo schema ripetuto:

```text
GDN → GDN → GDN → QSA
```

Quindi:

- **36 layer GDN**: elaborazione ricorrente con convoluzione e stato persistente.
- **12 layer QSA**: attenzione con KV cache e indexer che seleziona le posizioni da consultare.

Nei layer QSA ci sono:

- **24 teste query**.
- **2 teste KV**.
- Dimensione delle teste **256**.
- Indexer con **4 teste da 128**, budget di selezione **2.048**.

### La parte MoE, presente in ogni layer

Ogni layer contiene:

- **512 esperti selezionabili**.
- Un router che ne sceglie **10 per token**.
- **Uno shared expert**, eseguito indipendentemente dalla selezione dei dieci.

Ogni esperto ha tre matrici:

```text
                      ┌─ gate: 2560 → 640 ─ SiLU ─┐
input da 2560 elementi│                           × → down: 640 → 2560
                      └─ up:   2560 → 640 ────────┘
```

Complessivamente ci sono **48 × 512 = 24.576 esperti routed**.

“Esperto” significa una coppia **layer + identificativo**: l'esperto 7 del layer 0 è diverso dall'esperto 7 del layer 1.

### Quantizzazione reale

Il nome `IQ2_XS` non significa che tutti i tensori usino lo stesso formato.

Per gli esperti routed:

| Matrici | Formato |
|---|---|
| Gate/up di 34 layer | IQ2_S |
| Gate/up di 11 layer | IQ2_XXS |
| Gate/up di 3 layer | IQ1_M |
| Down di tutti i 48 layer | Q2_0 |

Gli altri componenti usano formati diversi: BF16, F32 e altre quantizzazioni.

## 3. Dove si trovano pesi e stati

### RTX 3060

La 3060 contiene ed esegue:

- Proiezioni dell'attenzione e dei blocchi GDN.
- Router.
- Hyper-connections e normalizzazioni.
- Shared expert di ogni layer.
- Stati ricorrenti GDN.
- KV cache e strutture dell'indexer.
- Proiezione finale sul vocabolario.
- Drafter MTP.
- **1.874 esperti routed**, scelti dal profilo di frequenza.

Questi 1.874 esperti occupano **2,53 GiB**.

La configurazione chiede `--expert-cache 1800`, ma il codice lo traduce in un **budget di memoria**. Poiché gli esperti hanno dimensioni differenti, in quel budget ne entrano effettivamente **1.874**.

La cache viene riempita all'avvio con gli esperti più frequenti del file:

`/home/toor/Strata/data/expert-profile.bin`

**La collocazione attuale è statica:** non avviene una promozione continua degli esperti dalle M10 alla 3060.

### Le quattro M10

Gli altri **22.702 esperti** vengono distribuiti fra le M10:

| GPU | Esperti interi | Memoria dei blob riportata al caricamento |
|---|---:|---:|
| M10 device 1 | 5.676 | 7,63 GiB |
| M10 device 2 | 5.676 | 7,63 GiB |
| M10 device 3 | 5.675 | 7,62 GiB |
| M10 device 4 | 5.675 | 7,61 GiB |

Il caricatore prende la lista degli esperti non presenti sulla 3060 e li assegna **a rotazione alle quattro schede**.

**Non divide le matrici di un esperto fra quattro GPU:** ciascun esperto risiede integralmente su una sola M10. Ogni M10 contiene esperti appartenenti a molti layer.

Il log conferma:

- **Nessun esperto duplicato** tra cache 3060 e tier M10.
- **Tutti i 24.576 esperti routed sono coperti dalle GPU.**

Sulle M10 gate/up rimangono nei rispettivi formati quantizzati. La matrice down viene riorganizzata nel layout S2 planare usato dai kernel Maxwell: **non viene espansa in FP16**.

### RAM e disco

Non è corretto dire che **l'intero modello** sia esclusivamente in VRAM.

Ci sono almeno due eccezioni importanti:

- La tabella di **token embedding**, circa **322 MiB**, è in memoria host mappata, leggibile dalla GPU.
- La grande tabella **PLE** rimane sul disco ed è consultata tramite il percorso `direct`.

Nel GGUF attuale la PLE è un tensore IQ4_NL da **28,8 GB**, con 320.001.536 righe da 160 elementi. Viene usata per costruire un'iniezione basata sugli n-grammi nel layer 1, contando da zero.

Dunque: **tutti gli esperti routed sono in VRAM; non tutti i dati necessari al modello lo sono.**

## 4. Come viene caricato

L'avvio corrente procede, in sostanza, così:

1. Legge configurazione, pack e GGUF.
2. Prepara embedding host e pesi principali della 3060.
3. Prepara PLE e drafter MTP.
4. Alloca gli stati della sessione, inclusa la capacità KV prevista.
5. Riempie i **1.874 posti della cache esperti sulla 3060**.
6. Distribuisce e carica il complemento sulle quattro M10.
7. Verifica su ogni M10 una catena esperto completa contro un riferimento numerico.
8. Collega le M10 anche al percorso di prefill.
9. Prepara e cattura i grafi CUDA per diverse dimensioni di finestra.

**Il caricamento delle quattro M10, nel percorso attuale, è sequenziale.**

Dopo il caricamento, i pesi degli esperti rimangono sulle rispettive GPU: **non vengono trasferiti dalla RAM a ogni token**.

## 5. Il percorso del token durante la generazione

Partiamo dal caso concettuale di un token; la sezione successiva distingue la finestra speculativa.

### A. Embedding e stato iniziale

L'identificativo del token viene passato al motore.

Un kernel sulla 3060 recupera il suo embedding dalla tabella host mappata e costruisce la rappresentazione iniziale nei quattro flussi residuali.

### B. Elaborazione del layer sulla 3060

Per ciascuno dei 48 layer:

1. Le hyper-connections ricavano l'ingresso del blocco.
2. La 3060 esegue:
   - il blocco ricorrente, se è un layer **GDN**;
   - il blocco di attenzione con KV e indexer, se è un layer **QSA**.
3. La rappresentazione risultante viene preparata per il blocco MoE.
4. Il router sulla 3060 calcola **i dieci esperti scelti e i rispettivi pesi**.

Nel layer 1 interviene anche il contributo PLE, preparato tramite letture host/disco e poi elaborato sulla 3060.

### C. Distribuzione del lavoro MoE

La 3060 pubblica in memoria host condivisa:

- Il vettore di attivazione.
- Gli identificativi degli esperti scelti.
- Un segnale che indica al thread host che il layer è pronto.

Il thread host consulta le tabelle di residenza:

```text
Esperto scelto dal router
          │
          ├─ residente sulla 3060 → calcolo sulla 3060
          │
          └─ residente su una M10 → calcolo su quella M10
```

Per le M10 l'host quantizza le attivazioni e prepara un trasferimento contenente **attivazioni + elenco dei lavori**.

Avvia prima la scheda con più lavori assegnati per quel layer/finestra.

**Le chiamate di avvio sono emesse in sequenza dall'host, ma il lavoro CUDA è asincrono:** questo non significa che una M10 debba terminare prima di avviare la successiva.

### D. Calcolo sulle M10

Ogni M10 interessata esegue:

1. Gate e up degli esperti di sua proprietà.
2. SiLU, prodotto e quantizzazione dell'intermedio.
3. Proiezione down.

Nel percorso decode attuale, il kernel down scrive le righe risultanti direttamente nella **RAM pinned mappata**.

L'host attende il completamento dei tier tramite sincronizzazione degli stream. La modalità senza sincronizzazione host tramite flag GPU **non è attiva** nel processo corrente.

Il percorso dei dati è quindi:

```text
3060
  │ attivazioni e routing
  ▼
RAM host pinned → quantizzazione/preparazione CPU
  │
  ├─ H2D → M10 1 ─┐
  ├─ H2D → M10 2 ─┤ risultati scritti
  ├─ H2D → M10 3 ─┤ nella RAM pinned
  └─ H2D → M10 4 ─┘
                   │
                   ▼
                  3060
```

Non è un trasferimento diretto GPU↔GPU.

### E. Ricomposizione sulla 3060

Nel frattempo la 3060 può eseguire:

- Gli esperti presenti nella propria cache.
- Lo shared expert, su uno stream separato.

Quando tutti i contributi necessari sono disponibili, importa le righe prodotte dalle M10 e calcola:

```text
uscita MoE =
    somma dei 10 risultati esperti × rispettivo peso del router
    + risultato dello shared expert
```

Le hyper-connections reinseriscono il risultato nello stato residuale. Si prosegue con il layer successivo.

**Non c'è un AllReduce fra le quattro M10:** producono risultati di esperti differenti, che vengono combinati sulla 3060.

### F. Uscita finale

Dopo il layer 48:

1. Combinazione finale delle hyper-connections.
2. Proiezione sul vocabolario sulla 3060.
3. Scelta del token secondo i parametri della richiesta.
4. Invio dell'identificativo al server Python.
5. Decodifica in testo e restituzione al client.

## 6. Perché si elaborano finestre, non sempre un token solo

È attivo il drafter **MTP**, residente sulla 3060, con un proprio blocco e i propri esperti.

Con `spec 4`, il percorso ordinario può verificare una finestra di:

```text
[token corrente, proposta 1, proposta 2, proposta 3]
```

La finestra può essere più corta, anche per la soglia di confidenza **0,5**.

Il modello principale verifica le posizioni insieme. Per ogni layer si possono quindi avere fino a:

**4 token × 10 selezioni = 40 incarichi esperto.**

Quaranta incarichi **non significano necessariamente quaranta esperti diversi**.

Il motore accetta il prefisso delle proposte che concorda con le proprie scelte, conserva gli stati corrispondenti e scarta il resto. Poi prepara la finestra successiva.

`STRATA_SPEC_DEPTH=1`, presente nell'ambiente, **abilita il rapporto diagnostico sull'accettazione**, non imposta la speculazione a un token.

## 7. Il prompt segue un percorso distinto

La lettura iniziale del prompt, o **prefill**, elabora blocchi di token con kernel e buffer dedicati.

Anche qui le M10 calcolano gli esperti che possiedono, ma il trasporto differisce dal decode:

- Le attivazioni vengono scaricate e preparate dall'host.
- Le M10 lavorano in sotto-batch fino a 64 token.
- I risultati passano da buffer locali delle M10 all'host.
- Un trasferimento riporta il blocco di risultati sulla 3060.

Quindi **non bisogna attribuire automaticamente al prefill il protocollo del decode**. Il motore dispone inoltre di un meccanismo di condivisione CPU per alcuni prefill; il messaggio generico di avvio non dimostra, da solo, quanto lavoro CPU venga effettivamente svolto in una determinata richiesta.

## 8. Sintesi

La fotografia attuale è: **un percorso principale sulla 3060, con una distribuzione statica degli esperti interi fra cinque GPU e un coordinamento host a ogni layer**.

Le M10:

- Non ricevono layer completi.
- Non mantengono la KV del modello principale.
- Non eseguono quattro copie suddivise dell'intero grafo.
- Calcolano gli esperti interi assegnati loro e restituiscono i risultati alla 3060.

Questa descrizione non costituisce una valutazione delle prestazioni di un'architettura alternativa, né una conclusione sulla fattibilità della tensor parallelism descritta in `SPLIT_MODE_TENSOR.md`.

## Fonti della rilevazione

- Argomenti e ambiente dei processi attivi, tramite `/proc`.
- Configurazione `/tmp/opencode/spec-sweep.json` (credenziali non riportate).
- Endpoint `/health` del server sulla porta 8088.
- Log `/home/toor/Strata/strata-iq2xs.log`, avvio del 19 ottobre 2026 alle 19:46:15.
- Metadati e directory dei tensori dei due shard GGUF in `/mnt/DATASET/STRATA_MODELS/models/iq2_xs/`, letti con `tools/gguf_reader.py`.
- `include/strata/core/geometry.hpp`.
- `src/program/generate.cpp`.
- `src/core/verify.cpp`.
- `src/core/expert_source.cpp`.
- `src/core/maxwell_experts.cpp`.
- `src/core/mtp.cpp`.
- `src/prefill/prefill.cpp`.
