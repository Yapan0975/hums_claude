# -*- coding: utf-8 -*-
"""
下载 Tier 1 文献 PDF + 抽取文本。
保存到 D:\\_7_sci\\semantic_mapping\\_new_paper\\refs\\
"""
import os, sys, urllib.request, ssl
from pypdf import PdfReader

REFS_DIR = r'D:\_7_sci\semantic_mapping\_new_paper\refs'
os.makedirs(REFS_DIR, exist_ok=True)

# Tier 1: (key, short_name, url)
TIER1 = [
    ('T1.1',  'Khronos_Schmid_RSS24',                 'https://arxiv.org/pdf/2402.13817'),
    ('T1.2',  'Clio_Maggio_RAL24',                    'https://arxiv.org/pdf/2404.13696'),
    ('T1.3',  'GS-LIVO_Hong_TRO25',                   'https://arxiv.org/pdf/2501.08672'),
    ('T1.4',  'ConvBKI_Wilson_TRO24',                 'https://arxiv.org/pdf/2310.16020'),
    ('T1.5',  'nvblox_Millane_ICRA24',                'https://arxiv.org/pdf/2311.00626'),
    ('T1.6',  'S-BKI_Gan_RAL20',                      'https://arxiv.org/pdf/1909.04631'),
    ('T1.7',  'Voxfield_Pan_IROS22',                  'https://www.ipb.uni-bonn.de/wp-content/papercite-data/pdf/pan2022iros.pdf'),
    ('T1.8',  'Hydra_Hughes_RSS22',                   'https://arxiv.org/pdf/2201.13360'),
    ('T1.9',  'Kimera_Rosinol_IJRR21',                'https://arxiv.org/pdf/2101.06894'),  # alternate arxiv
    ('T1.10', 'OpenVox_Deng_arxiv25',                 'https://arxiv.org/pdf/2502.16528'),
    ('T1.11', 'LatentBKI_Wilson_arxiv24',             'https://arxiv.org/pdf/2410.11783'),
    ('T1.12', 'PanopticMultiTSDFs_Schmid_ICRA22',     'https://arxiv.org/pdf/2109.10165'),
]

UA = ('Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
      'AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36')

ssl_ctx = ssl.create_default_context()
ssl_ctx.check_hostname = False
ssl_ctx.verify_mode = ssl.CERT_NONE


def fetch(url: str, out: str) -> int:
    req = urllib.request.Request(url, headers={'User-Agent': UA, 'Accept': 'application/pdf,*/*'})
    with urllib.request.urlopen(req, timeout=60, context=ssl_ctx) as r:
        data = r.read()
    with open(out, 'wb') as f:
        f.write(data)
    return len(data)


def extract(pdf_path: str, txt_path: str) -> int:
    r = PdfReader(pdf_path)
    parts = []
    for i, page in enumerate(r.pages):
        try:
            t = page.extract_text() or ''
        except Exception as e:
            t = f'[ERROR p{i+1}: {e}]'
        parts.append(f'===== PAGE {i+1} =====\n{t}')
    text = '\n\n'.join(parts)
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(text)
    return len(text)


def main():
    ok = 0; fail = 0
    for key, name, url in TIER1:
        pdf = os.path.join(REFS_DIR, f'{key}_{name}.pdf')
        txt = os.path.join(REFS_DIR, f'_{key}_{name}.txt')
        if os.path.exists(txt) and os.path.getsize(txt) > 1000:
            print(f'  SKIP {key} {name} (txt exists)')
            ok += 1
            continue
        try:
            print(f'  DL   {key} {name} ... ', end='', flush=True)
            if not os.path.exists(pdf):
                size = fetch(url, pdf)
                print(f'{size//1024} KB pdf ... ', end='', flush=True)
            chars = extract(pdf, txt)
            print(f'{chars} chars txt OK')
            ok += 1
        except Exception as e:
            print(f'FAIL ({e.__class__.__name__}: {str(e)[:80]})')
            fail += 1
    print(f'\nDONE: {ok} ok / {fail} fail / {len(TIER1)} total')


if __name__ == '__main__':
    main()
