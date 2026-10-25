"""Fail-fast TP boundary checks: no model/GGUF or GPU allocation should be needed."""
import os
from pathlib import Path
import subprocess
import sys

binary = str(Path(sys.argv[1]).resolve())
env = {k: v for k, v in os.environ.items() if not k.startswith(('STRATA_MW_', 'STRATA_S2_', 'STRATA_VERIFY_DEVICE_PLAN'))}
env['STRATA_MW_GPUFLAG'] = '2'
base = ['--serve', '--maxwell-split', 'tensor', '--maxwell-devices', '1,2,3,4']
cases = [
    (['--maxwell-split', 'tensor'], {}, 'requires --serve'),
    (['--maxwell-split', 'unknown'], {}, 'must be expert or tensor'),
    (base + ['--batch', '2'], {}, '--batch is not yet supported'),
    (base + ['--pipeline-windows', '2'], {}, '--pipeline-windows is not yet supported'),
    (base + ['--layer-split', '24'], {}, '--layer-split is not yet supported'),
    (base + ['--spec-split'], {}, '--spec-split is not yet supported'),
    (base + ['--no-capture'], {}, 'requires the token-graph'),
    (base, {'STRATA_MW_GPUFLAG': '0'}, 'needs STRATA_MW_GPUFLAG'),
    (base, {'STRATA_MW_GPUFLAG': '3'}, 'needs STRATA_MW_GPUFLAG'),
    (base, {'STRATA_S2_V2': '0'}, 'needs STRATA_S2_V2'),
    (base, {'STRATA_MW_IQ_SPLIT': '1'}, 'IQ launch geometry'),
    (base, {'STRATA_MW_IQ_SPLIT': '2', 'STRATA_MW_IQ_T256': '1'}, 'IQ launch geometry'),
    (base, {'STRATA_MW_IQ_SPLIT': '2', 'STRATA_MW_IQ_T256': 'yes'}, 'IQ launch geometry'),
    (base, {'STRATA_S2_V2': ''}, 'needs --native and --expert-profile'),
    (base + ['--maxwell-devices', '1,2,2,4'], {}, 'duplicate M10'),
    (base + ['--maxwell-devices', '0,1,2,3'], {}, 'invalid M10'),
    (base + ['--maxwell-devices', '1,2,3'], {}, 'exactly four'),
    (base + ['--maxwell-devices', '1,2,3,4junk'], {}, 'invalid M10'),
]
for name in ['STRATA_MW_ZEROCOPY', 'STRATA_MW_LOCAL_DOWN', 'STRATA_MW_DOWN_LOCAL_TIMING',
             'STRATA_MW_ROWS_F16', 'STRATA_VERIFY_DEVICE_PLAN']:
    cases.append((base, {name: '1'}, name + ' is not supported'))
for args, settings, expected in cases:
    run = subprocess.run([binary] + args, env=env | settings, capture_output=True, text=True, timeout=10)
    text = run.stdout + run.stderr
    if run.returncode != 2 or expected not in text:
        raise AssertionError((args, settings, run.returncode, text))
print(f'PASS: {len(cases)} TP requests rejected before model loading')
