"""Audit completeness, hashes, packet accounting, and checkpoint selection."""
import csv
import hashlib
import json
from pathlib import Path
from pypdf import PdfReader

ROOT=Path(__file__).resolve().parents[1]


def main():
    suite=json.loads((ROOT/'artifacts/scenarios/test.json').read_text())
    ids={r['id'] for r in suite['records']}
    assert len(ids)==400
    files=list((ROOT/'artifacts/results').glob('*.csv'))
    assert len(files)==12, f'expected 12 policy/seed files, got {len(files)}'
    episodes=0
    for path in files:
        with path.open() as f:rows=list(csv.DictReader(f))
        assert len(rows)==400 and {r['scenario_id'] for r in rows}==ids,path
        for r in rows:
            assert r['suite_sha256']==suite['sha256']
            generated=int(r['generated']);delivered=int(r['delivered'])
            assert generated==delivered+int(r['expired'])+int(r['overflow'])
            assert int(r['slots'])==512
            assert abs(float(r['delivery_fraction'])-delivered/generated)<1e-12
            expected=delivered-2*(int(r['expired'])+int(r['overflow']))-.02*int(r['retunes'])-.01*float(r['energy'])
            assert abs(float(r['return_sum'])-expected)<1e-8
        episodes+=len(rows)
    checkpoints=list((ROOT/'artifacts/checkpoints').glob('*.pt'))
    assert len(checkpoints)==6
    for path in checkpoints:
        meta=json.loads(path.with_suffix('.json').read_text())
        assert hashlib.sha256(path.read_bytes()).hexdigest()==meta['checkpoint_sha256']
        logdir=ROOT/'artifacts/training_logs'/path.stem
        vals=[json.loads(line) for line in (logdir/'validation.jsonl').read_text().splitlines()]
        selected=max(vals,key=lambda r:(r['delivery'],-r['energy_per_delivery'],-r['step']))
        assert selected['step']==meta['metadata']['step']
        assert selected['delivery']==meta['metadata']['validation_delivery']
        assert json.loads((logdir/'summary.json').read_text())['steps']==250000
    report=PdfReader(ROOT/'report/report.pdf')
    assert len(report.pages)==3
    assert 'References' in report.pages[2].extract_text()
    for name in ['Ayush Singh','B23EE1008','Tashir Ahmad','B23ME1074','github.com/Ayush-Singh-as/DeadlineHop']:
        assert name in report.pages[0].extract_text(),name
    assert (ROOT/'artifacts/demo/rollout.gif').stat().st_size>0
    checks=dict(evaluated_episodes=episodes,trained_checkpoints=len(checkpoints),training_steps=6*250000,
                test_slots=episodes*512,report_content_pages=2,report_reference_pages=1,
                suite_sha256=suite['sha256'],checks='packet conservation, reward reconciliation, checkpoint hashes, validation-only selection, complete suites, report metadata')
    (ROOT/'artifacts/verification.json').write_text(json.dumps(checks,indent=2))
    print(json.dumps(checks,indent=2))


if __name__=='__main__': main()
