# Token embedding in VRAM: RTX 3060 + quattro M10

## Cambiamento

`STRATA_EMBED_VRAM=1` carica `token_embd.weight` nella VRAM della GPU corrente
invece che nella RAM host pinned/mappata. I blocchi GGUF, il formato IQ4_XS,
lo stride delle righe e i kernel di gather rimangono invariati. Nessuna
dequantizzazione permanente della tabella.

- Dimensione reale: **337.715.200 byte = 322,07 MiB**.
- La scelta è opt-in, letta al caricamento. Se la richiesta esplicita di VRAM
  fallisce, il caricamento segnala errore, senza ripiego silenzioso sulla RAM.
- Senza la variabile (o con `=0`) rimane il percorso precedente, compresi i
  ripieghi VRAM già esistenti quando la memoria host non è utilizzabile.
- La stessa tabella è usata da prefill, verifier e drafter tramite `NativeEmbed`.
- L'allocazione precede il dimensionamento della cache esperti e viene quindi
  conteggiata nel budget di memoria esistente.

Configurazione del test: Strata 0.1.45, IQ2_XS, RTX 3060 12 GB, quattro Tesla M10,
contesto **256000**, KV **k8v4**, spec **4**, soglia MTP **0,5**. Nessun cambio
alla collocazione o ai kernel degli esperti. Driver NVIDIA 580.178.04.

La nuova configurazione locale, esclusa da git perché contiene credenziali, è
`strata-iq2xs-embed-vram.json`: copia della configurazione standard con la sola
aggiunta di `"STRATA_EMBED_VRAM": "1"` nell'oggetto `env`.

## Verifica della numerica

Il target standalone `native_embed_placement_test` carica contemporaneamente la
tabella host e la copia VRAM e confronta bit per bit tutte le righe dequantizzate
dagli stessi kernel:

```text
PASS: all 248320 vocabulary rows x 2560 F32 values bit-identical;
gather_one PASS; 322.07 MiB in VRAM
```

Confronto di **635.699.200 valori F32** nel gather batched, più il controllo
del gather singolo su tre token (primo, centrale, ultimo). Il test fallisce se
la presunta baseline host è invece ricaduta in VRAM.

Riproduzione dalla radice del repository, su CUDA0 e con VRAM sufficiente:

```bash
cmake --build build --target strata native_embed_placement_test -j4
build/native_embed_placement_test /path/to/model-00001-of-00002.gguf
```

Il target di test è `EXCLUDE_FROM_ALL` e non richiede un cambio del default del
motore. Build CUDA riuscito; toolchain HIP/SYCL non disponibili per questa verifica.

## Protocollo prestazionale

Baseline prima del riavvio, poi stesso protocollo con embedding VRAM: tre prompt
brevi, tre ripetizioni ciascuno, temperatura 0, limite 512 token. Velocità decode
estratta dal log del motore; il wall HTTP include anche il prefill e non è usato
come velocità decode. Nessun profiler CUPTI.

La generazione può variare fra richieste anche sulla baseline: confronti di
lunghezza o testo generato non sono una prova di parità numerica. Il primo prompt
story della baseline ha riusato 53 token; i tempi di prefill dei prompt brevi non
sono quindi un confronto controllato.

### Baseline: embedding host mappato

| Prompt | Tok/s delle tre ripetizioni | Media |
|---|---|---:|
| tell me a story about AI | 37,9 / 37,9 / 39,8 | 38,53 |
| explain Fibonacci | 43,7 / 44,9 / 44,8 | 44,47 |
| write an example code in python, medium difficulty | 43,7 / 45,8 / 45,6 | 45,03 |

Media complessiva: **42,68 tok/s**, 4.344 token generati.

### Embedding in VRAM

| Prompt | Tok/s delle tre ripetizioni | Media |
|---|---|---:|
| tell me a story about AI | 37,1 / 39,0 / 39,9 | 38,67 |
| explain Fibonacci | 42,3 / 45,2 / 45,0 | 44,17 |
| write an example code in python, medium difficulty | 45,0 / 45,7 / 45,7 | 45,47 |

Media complessiva: **42,77 tok/s**, 4.457 token generati.

**Delta: +0,21%. Nessun miglioramento end-to-end dimostrabile:** il delta è molto
più piccolo della dispersione delle singole richieste. Screening 3×3 in ordine
host→VRAM, non una campagna alternata ABBA con intervalli di confidenza. La baseline
usa il binario precedente; la candidata ricompilata cambia solo collocazione e
messaggio diagnostico dell'embedding, senza nuovi kernel o cambi di aritmetica.
Non è stata ripetuta una misura serving della candidata con l'opzione disattivata.

Le osservazioni individuali sono in [baseline.json](baseline.json) e
[vram.json](vram.json). La verifica di parità delle righe è separata dai tok/s.

## Caricamento VRAM e controlli del serving

Il log conferma `token embedding IQ4_XS in VRAM (322 MiB)` e la cache mantiene
**1.874 esperti / 2,53 GiB**, come nella baseline. Anche il complemento M10 è
invariato: 5.676 / 5.676 / 5.675 / 5.675 esperti, zero sovrapposizioni con la 3060.

- Quattro round-trip gate M10 al caricamento: **PASSED**.
- `/health`: loaded/ok, contesto **256000**.
- **391**, **9999**, **391 × 8 richieste concorrenti**: tutti corretti.
- Il wrapper ha raggiunto lo stato pronto dopo **452,4 secondi**. Questo tempo
  non è un confronto controllato della velocità di caricamento.
- Prompt aggiuntivo da **1.509 token**, zero riuso del prefisso: risposta **391**,
  69 token generati, 22,606 s HTTP. Verifica funzionale del prefill, non un A/B
  della sua velocità.
- Dopo il decode, `nvidia-smi`: **11.359 MiB usati, 551 MiB liberi** sulla 3060.
  Le M10 mantengono i loro consumi precedenti (7.931 / 7.930 / 7.918 / 7.916 MiB).
- Dopo il prompt aggiuntivo, lettura finale: **11.401 MiB usati, 509 MiB liberi**
  sulla 3060; `/health` ancora loaded/ok a 256000.

Il controllo della capacità a 256k riguarda l'allocazione della sessione: non è
una prova di ingestione di un prompt completo da 256k token.

## Stato finale e ripristino

Embedding **lasciato in VRAM come richiesto**, tramite la configurazione opt-in
`/home/toor/Strata/strata-iq2xs-embed-vram.json`; nessun cambio del default globale.
Il server resta sulla porta 8088. La configurazione standard
`/tmp/opencode/spec-sweep.json` non è stata modificata: usarla al successivo
riavvio riporta l'embedding al percorso host anche con il nuovo binario.

Il nuovo `engine/strata` coincide con `build/strata`, SHA256
`cd0ce43dceec9b44911355628c3bb7ddebd30762e9bb8c18d184a0b84766f433`.
Backup del binario precedente: `engine/strata.bak-embed-host-20261019`, SHA256
`171c4035ab503220fd08463205192edb9b70ee52740d9ce975d5deb649636384`.

Nessun commit o push. Non viene raccomandato il nuovo posizionamento come default
più veloce: costa 322 MiB di VRAM e in questo screening il guadagno non si distingue
dal rumore. Nessuna conclusione sulla velocità di un kernel embedding isolato o
sull'effetto a contesti lunghi, che qui non sono stati misurati.
