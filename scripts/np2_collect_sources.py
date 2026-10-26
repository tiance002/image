"""NP2：采集外部基线的来源、版本、权重哈希与依赖版本，供报告引用。"""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def pkg_version(name):
    try:
        module = __import__(name)
        return getattr(module, '__version__', 'unknown')
    except Exception:
        return None


def main():
    weights = ROOT / 'third_party/weights'
    data = dict(
        date='2026-10-26',
        python=platform.python_version(),
        platform=platform.platform(),
        packages={k: pkg_version(k) for k in ('torch', 'numpy', 'cv2', 'einops', 'natsort', 'scipy', 'lpips', 'tqdm', 'ultralytics')},
        sources=dict(
            adair=dict(
                repo='https://github.com/c-yn/AdaIR',
                commit='ccb8b98e49614e07badd0641e5163fa7635c2f02',
                download='https://codeload.github.com/c-yn/AdaIR/tar.gz/<commit>',
                license='MIT（仓库 LICENSE）',
                official_weights='https://drive.google.com/drive/folders/1x2LN4kWkO3S65jJlH-1INUFiYt8KFzPH',
                expected_file='ckpt/adair5d.ckpt',
                status='BLOCKED',
                block_reason='本机对 drive.google.com / docs.google.com / www.googleapis.com / drive.usercontent.google.com 全部连接失败（curl 000，CONNECT 被拒或超时）；'
                             '官方仓库 README 只给该 Drive 链接，GitHub 无 release 资源，HF 无 AdaIR 托管（issue #15 请求上传 HF 未被作者采纳），Gitee/ModelScope/HF 搜索都没有可用镜像。',
                prepared=dict(script='scripts/np2_adair_restore.py',
                              architecture_verified=True,
                              params=28784824,
                              note='结构与 strict 加载路径已就绪；缺权重时脚本直接报错，不会随机初始化继续推理'),
                supported_weather=['fog', 'rain', 'night'],
                unsupported_weather='snow（官方五退化不含去雪）'),
            histoformer=dict(
                repo='https://github.com/sunshangquan/Histoformer',
                commit='1f045f06c03551c31504d8042dbe6ff9b9569108',
                download='https://codeload.github.com/sunshangquan/Histoformer/tar.gz/<commit>',
                license='官方 HF 元数据声明 MIT；GitHub 无独立 LICENSE 文件',
                official_hf='https://huggingface.co/sunsean/Histoformer',
                official_hf_revision='f6acde20acaa26ef5e235f2c71ff7637f4d9a945',
                official_hf_url_used='https://hf-mirror.com/sunsean/Histoformer/resolve/main/Allweather/pretrained_models/net_g_real.pth',
                gdrive_mirror='https://drive.google.com/drive/folders/1dmPhr8Z5iPRx9lh7TwdUFPSfwGIxp5l0（不可达）',
                weight_note='官方 HF 只发布 net_g_real.pth；README 的 benchmark 使用 net_g_best.pth，两者是不同权重（339 个张量全部不同，体积同为 66,603,665 字节）'),
        ),
        weights={},
        code_hashes={},
    )

    for name, path in [
        ('histoformer_official_net_g_real', weights / 'histoformer_official_net_g_real.pth'),
        ('histoformer_mirror_net_g_best', weights / 'histoformer_mirror_net_g_best.pth'),
        ('histoformer_mirror_net_g_real', weights / 'histoformer_mirror_net_g_real.pth'),
    ]:
        if path.is_file():
            data['weights'][name] = dict(path=str(path.relative_to(ROOT)), sha256=sha256(path), bytes=path.stat().st_size)

    data['weights']['histoformer_official_hf_lfs_sha256'] = '8ae731d0fd5e2761cec7428322c7602146cba42c73188cc4b0fcc17232065958'
    data['weights']['mirror_provenance'] = dict(
        repo='https://huggingface.co/dronefreak/Histoformer',
        note='第三方镜像，官方未给出 net_g_best.pth 的哈希，无法独立确认版本；'
             '但该镜像的 net_g_real.pth 与官方 HF 字节完全一致（见 histoformer_mirror_net_g_real.sha256），说明它是官方文件的忠实转录。'
             '因此 net_g_best.pth 只能作为受限来源证据，不等同官方 benchmark 权重。')

    for rel in ['third_party/histoformer/Allweather/test_histoformer.py',
                'third_party/histoformer/Allweather/util.py',
                'third_party/histoformer/Allweather/Options/Allweather_Histoformer.yml',
                'third_party/histoformer/basicsr/models/archs/histoformer_arch.py',
                'third_party/adair/net/model.py',
                'third_party/adair/utils/dataset_utils.py',
                'third_party/adair/utils/image_utils.py']:
        path = ROOT / rel
        if path.is_file():
            data['code_hashes'][rel] = sha256(path)

    out = ROOT / 'results/np2_external_probe/external_sources.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(data['weights'], ensure_ascii=False, indent=2))
    print('已写入', out)


if __name__ == '__main__':
    main()
