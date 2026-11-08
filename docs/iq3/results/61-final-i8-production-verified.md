# Iterazione 8 — IQ3_S in produzione: il passaggio, verificato

Data: 2026-11-08, finestra 11:00–11:33 CET (10:00–10:33Z). Host `strata` (LXC su Proxmox).
Autorizzazione dell'operatore: promuovere IQ3_S in produzione dopo il verdetto dell'Iterazione 7
(criterio 55 **FAIL formale a 54,30**, ma dentro il rumore: SE ~0,80 tok/s, con decisione esplicita
dell'operatore di promuovere). Le ottimizzazioni — gate/up 134 µs/L, LUT in shared — **restano per
dopo e non bloccano**. Ordine eseguito: 1→7 come da brief, senza varianti al volo.

## 1. Cosa è in produzione adesso

| | valore |
|---|---|
| modelli | **`qwen3.8-flash-next-iq3_s`**, `loaded: true`, `max_context` 128000 |
| motore | `/root/Strata-maxwell/engine-iq3-e1de86e8/strata`, **md5 `0f5effb3d86b2251cd80439079a50575`** = commit `e1de86e8` |
| libreria | `libstrata_kq.so` accanto al binario, md5 `b7b1d93d8e7ef763034a098604a7c586` |
| config | `/root/Strata-maxwell/strata-iq3_s-prod.json`, **modo 0600**, chiave mai stampata |
| cache 3060 | **1267 slot**, 2,45 GiB, pre-filled 1267/1267 in 4,4 s (600 MB/s), slot 0 verificato |
| partizione | `Maxwell pairs in the 3060 cache: 0 of 23309` — il pool intero, come da modello |
| tier Maxwell | **8**, 2914×5 + 2913×3 esperti, **5,54–5,56 GiB** per die |
| gate round-trip | **8/8 PASSED**, worst rel 1,18e-02 … 1,29e-02, 23 315 blob convertiti |
| prefill | `8 Maxwell tier(s) compute their experts' rows of every prompt chunk` |
| VRAM 3060 | **1332 MiB** liberi al load, **1033 MiB** a regime dopo il soak |
| servizio | `active`, MainPID 2435, **NRestarts=0**, health OK in 251 s |
| limiti di potenza | 3060 **120,00 W**, M10 × 8 **30,00 W** — letti, mai scritti |

Il motore è **lo stesso binario** di tutte le misure dell'Iterazione 7: il down planare che ha fatto
271 → 118 µs/L è quello che serve adesso. `engine-mw8` (motore 0.1.47, pre-Iterazione 8) è rimasto
intatto su disco come riserva.

## 2. Test a 128k — PASS

Prompt da **119 275 token** (haystack della repo + needle a metà), generazione 64 token.

| | misura |
|---|---|
| prompt | **119 275 token**, 0 riusati, tutti letti |
| prefill | **178,4 tok/s** (668 680 ms) |
| decode | **44,7 tok/s** |
| needle | `meadow-copper-504` → **FOUND**, exit 0 |
| VRAM libera 3060 | **1035 MiB** durante/dopo il test |
| errori | **0** OOM, 0 CUDA, 0 allocazioni fallite (word-boundary sul log intero) |
| AER | contatore fermo a **4** prima e dopo; **tutte le 9 GPU a 0** |

**`contextWindow` per PI: 128000.** Non è stato necessario scendere a 96k: il contesto pieno regge,
con la needle trovata e ~1 GB libero sulla 3060. Il valore è già quello che il motore dichiara in
`/health` (`"max_context": 128000`).

Nota di forma: la risposta è ragionamento (`We need answer user's question: …`) perché il controllo
gira **con thinking attivo**, come il controllo 64k dell'Iterazione 7 — il needle è stato trovato,
quindi il contesto è stato letto. Non è un difetto del motore.

## 3. Soak di 10 minuti in produzione

84 run consecutivi, prompt città, 300 token ciascuno, **thinking attivo**.

| | valore |
|---|---|
| run | **84** |
| media | **43,51 tok/s** |
| mediana | 43,45 tok/s |
| dev. std | 1,29 tok/s |
| min / max | 39,7 / 46,7 tok/s |
| primo / ultimo | 43,2 / 42,4 tok/s → **drift −0,8 tok/s** |
| VRAM 3060 | **1033 MiB costanti** in ogni run, nessuna deriva |
| AER prima | bridge `0000:09:11.0` a **4**, GPU tutte a 0 |
| AER dopo | bridge `0000:09:11.0` a **4**, GPU tutte a 0 → **nessun evento in finestra** |
| servizio dopo | `active`, NRestarts=0, health OK |

Il drift di −0,8 tok/s su 84 run è dentro la deviazione standard e non è un degrado progressivo: il
plateau VRAM è piatto e il contatore AER non si è mosso. Un soak in produzione misura un carico
diverso dal banco (300 token, non 600; un solo prompt), quindi **43,5 non va confrontato con il
54,30 del banco** dell'Iterazione 7: è il numero di esercizio della produzione, non il criterio.

## 4. La richiesta di prova (città)

Prima del soak, un run città da 600 token sul percorso di produzione: **50,1 tok/s** dal motore
(600 token in 11 981 ms), prompt 74 token a 86,6 tok/s, draft accettate 364/554 (**65,7%** — in linea
con il 65,8% misurato in Iterazione 6), 1077 MiB liberi. È il primo run dopo il load, quindi a freddo:
coerente con il 50,4 del primo run del banco I7.

## 5. File e comandi

### File creati sul server (permessi 0600, contengono la chiave, **non** vanno committati)

| file | contenuto |
|---|---|
| `strata-iq3_s-prod.json` | la produzione: motore nuovo, cache 988, 8 device, reserve 250, 128k, log dedicato |
| `strata-iq2_xs-prod-reserve.json` | la riserva IQ2_XS **con lo stesso motore nuovo**, cache 1800, NON attiva |
| `/root/strata.service.bak.20261108T101754Z` | la unit com'era prima del passaggio |
| `/root/strata-switch.sh` | lo switch in un comando (sotto) |

### Tornare a IQ2_XS in un minuto

Lo script fa backup della unit, riscrive `ExecStart`, `daemon-reload`, scarica le schede, riavvia e
aspetta `/health`. Il tempo di ritorno è il **load** (~250 s), non lo switch: il comando è uno.

```sh
/root/strata-switch.sh iq2        # IQ2_XS con il motore nuovo e1de86e8 (non regressione verificata)
/root/strata-switch.sh iq2-old    # IQ2_XS sul vecchio engine-mw8, com'era prima dell'Iterazione 8
/root/strata-switch.sh iq3        # di nuovo IQ3_S
/root/strata-switch.sh            # dice quale config è attiva, senza toccare nulla
```

Se si preferisce il passaggio a mano, è esattamente questo:

```sh
cp -a /etc/systemd/system/strata.service /root/strata.service.bak.$(date -u +%Y%m%dT%H%M%SZ)
sed -i 's|--config [^ ]*|--config /root/Strata-maxwell/strata-iq2_xs-prod-reserve.json|' \
    /etc/systemd/system/strata.service
systemctl daemon-reload && systemctl restart strata
# attesa ~250 s, poi: curl -s http://127.0.0.1:8080/health
```

### Snapshot ZFS (lo esegue l'operatore sull'host Proxmox)

Il container non vede `zfs`: il filesystem è `rpool/data/subvol-101-disk-0` montato su `/`, e i dati
stanno lì. Dall'**hypervisor**, non dentro il container:

```sh
# snapshot a caldo del rootfs del container (il nome del dataset è quello che mount mostra su /)
zfs snapshot rpool/data/subvol-101-disk-0@iq3s-prod-2026-11-08

# verifica
zfs list -t snapshot -o name,used,creation | grep iq3s-prod

# ripristino (solo se serve: fermare prima il container)
# zfs rollback -r rpool/data/subvol-101-disk-0@iq3s-prod-2026-11-08
```

I 78 GB di modelli e i 49 GB di pack stanno sotto `/root/Strata-data` e `/mnt/backup`: se non sono
nello stesso dataset del rootfs, lo snapshot del solo rootfs **non li copre**. Da verificare sull'host:
`zfs list -r -o name,mountpoint | grep -E 'Strata-data|backup'`. In questa finestra il pack IQ3_S era
già su disco da prima e non è stato modificato.

## 6. Cosa questa finestra non fa

Non tocca i kernel, non cambia il criterio, non tocca i limiti di potenza (`-pl` mai eseguito).
Non promuove nessuna ottimizzazione: gate/up e LUT restano le variabili separate già dichiarate.
Non modifica il pack né i pesi. Non esegue lo snapshot ZFS (è dell'operatore, sull'host).

## 7. Evidenza archiviata

| file | contenuto |
|---|---|
| `results/62-longctx-i8-128k.json` | il test 128k: 119 275 token, needle trovata, 1035 MiB liberi |
| `results/63-bench-i8-city-probe.json` | la richiesta di prova città, 50,1 tok/s, testo incluso |
| `results/64-soak-i8-10min.json` | i **84 run** del soak, riga per riga |
| `results/65-soak-i8-runs.txt` | l'output leggibile del soak (media, mediana, drift) |

Verificati privi di `api_key`/`Bearer`/materiale della chiave: 0 corrispondenze su tutti e cinque i
file di evidenza. I due config restano sul server a 0600 e non entrano nel repo.

## 8. Un dettaglio verificato, per non lasciare un falso allarme

La `.so` K-quant (`libstrata_kq.so`) è **opt-in**: `mmq_kq_loader.cpp` la apre solo con
`STRATA_MMQ_KQ=1`, e la produzione non la arma (0 righe "K-quant prompt kernels" nel log). Quindi
lo zero è **corretto**, non un caricamento fallito. La libreria è comunque accanto al binario e con
l'md5 giusto, così l'eventuale arm futuro non richiede un'altra copia.

## 9. Un solo problema incontrato, e la sua causa

Il primo tentativo di scrittura della config è fallito con `UnboundLocalError` nel mio script Python
(assegnazione a `a` dentro una funzione annidata, che la rende locale). Nessun file è stato scritto:
la config è stata creata al secondo tentativo e verificata (chiave presente, modo 0600, argomenti
attesi). Registrato perché lo script è rimasto nella sessione e il fallimento non ha lasciato traccia
nel repo.