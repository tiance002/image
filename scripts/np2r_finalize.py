"""Verify NP2-R completed artifacts and preservation; never starts inference or training."""
from pathlib import Path
import hashlib
import json
import platform
import subprocess

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'results/np2_revision'


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(4*1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def load(relative):
    return json.loads((OUT/relative).read_text(encoding='utf-8'))


def main():
    manifest_path = OUT/'execution_manifest.json'
    if manifest_path.exists():
        raise FileExistsError(manifest_path)
    preflight = load('preflight.json')
    for name, old in preflight['preexisting_external_probe_files'].items():
        path = ROOT/name; stat = path.stat()
        assert stat.st_size == old['size'] and stat.st_mtime_ns == old['mtime_ns'], ('Old result changed', name)
        if 'sha256' in old:
            assert sha(path) == old['sha256'], name
    current_old_files = {p.relative_to(ROOT).as_posix() for p in (ROOT/'results/np2_external_probe').rglob('*') if p.is_file()}
    assert current_old_files == set(preflight['preexisting_external_probe_files']), 'Historical result tree changed'
    for name, old in preflight['frozen_resources'].items():
        assert sha(ROOT/name) == old['sha256'], ('Frozen file changed', name)
    assert load('histoformer_detect_complete.json')['restored_images'] == 200
    assert load('gpu_diagnostics_complete.json')['completed']
    metadata = load('restored/histoformer_real_fusionfix/metadata.json')
    assert metadata['images'] == 200 and metadata['adaptation']['fusion_version'] == 'np2r-1'
    assert len(list((OUT/'restored/histoformer_real_fusionfix').rglob('*.png'))) == 200
    for filename in ('ap001.json','runtime025.json'):
        rows = load('predictions/histoformer_real_fusionfix/'+filename)['per_image']
        assert len(rows) == len({r['image_id'] for r in rows}) == 200
    supervision = load('supervision_probe.json')
    old = json.loads((ROOT/'results/np2_external_probe/supervision_probe.json').read_text(encoding='utf-8'))
    assert supervision['manifest']['images'] == old['manifest']['images']
    for name, model in supervision['models'].items():
        assert model['gradient']['batches'] == 64 and model['gradient']['all_pass'], name
        assert model['batch_consistency']['batches'] == 16 and model['batch_consistency']['all_batches_pass'], name
        # Native positives must still agree exactly for each original GT in identical fixed batches.
        before = [(r['image'],r['category'],r['positives']) for r in old['models'][name]['per_gt']]
        after = [(r['image'],r['category'],r['positives']) for r in model['per_gt']]
        assert before == after, ('Native assignment changed', name)
    scale = load('scale_order_probe.json')
    assert scale['all_completed'] and scale['sample']['count'] == 24
    assert set(scale['models']) == {'v4','C0','C1'}
    for name, model in scale['models'].items():
        assert model['weights_and_bn_unchanged'] and model['detector_weights_and_bn_unchanged'], name
    quality = load('quality_lpips_fusionfix.json')
    assert quality['status'] == 'completed' and quality['n'] == 200
    assert len(quality['per_image']) == 200 and quality['reference_source_sha256'] == sha(ROOT/'results/np2_external_probe/quality_lpips.json')
    assert load('fusion_identity_check.json')['identity_all_ok']
    assert load('scale_probe_cpu_checks.json')['all_pass']
    assert load('bootstrap.json')['resamples'] == 1000
    assert (OUT/'REVISION_REPORT.md').is_file()
    scripts = ['np2_histoformer_restore.py','np2_fusion_identity_check.py','np2r_probe_tests.py',
               'np2r_supervision_probe.py','np2_scale_order_probe.py','np2r_scale_checks.py','np2r_quality.py',
               'np2r_summarize.py','np2r_revision_report.py','np2r_finalize.py']
    source = {('scripts/'+name): sha(ROOT/'scripts'/name) for name in scripts}
    source['results/np2_revision/tf32_gradient_diagnosis.py'] = sha(OUT/'tf32_gradient_diagnosis.py')
    artifacts = {p.relative_to(OUT).as_posix(): sha(p) for p in OUT.rglob('*')
                 if p.is_file() and p.suffix in {'.json','.md','.csv'} and not p.name.endswith('.progress.json')
                 and p != manifest_path and 'planning' not in p.parts and 'review_sources' not in p.parts}
    record = dict(status='complete_local_revision', review_baseline=preflight['review_baseline'],
                  historical_files_preserved=len(current_old_files), frozen_resources_unchanged=preflight['frozen_resources'],
                  no_training=True, no_optimizer_update=True, no_push=True, no_merge=True, default_weights_replaced=False,
                  no_color_audit_or_generation=True, no_reference_val_prediction_reinference=True,
                  histoformer_val_images=200, aligned_dev_images=200,
                  numerical_precision=supervision['manifest']['precision'],
                  full_supervision=dict(images_per_checkpoint=64, checkpoints=3, stat_batches=48,
                                        gradient_batches=192, backward_calls=1536, original_positives_exact_match=True),
                  scale=dict(images_per_checkpoint=24, checkpoints=3, main='D/A/B FP32 common geometry',
                             quantization_padding_precision_and_actual_rect_separate=True),
                  python=platform.python_version(), source_sha256=source, artifact_sha256=artifacts)
    data = json.dumps(record,ensure_ascii=False,indent=2)+'\n'
    manifest_path.write_text(data,encoding='utf-8',newline='\n')
    assert json.loads(manifest_path.read_text(encoding='utf-8')) == record
    print('Preservation and final artifact verification complete:', len(artifacts),'artifacts')


if __name__ == '__main__':
    main()
