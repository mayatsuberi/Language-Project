"""
residual_spectrum.py
--------------------
Compares the intrinsic dimensionality of the cross-lingual residual across
embedding models.

The residual is the part of a foreign-language embedding that English cannot
linearly explain. How that residual's variance is *distributed* is informative
in itself:

  - a steep spectrum (top few components hold most of the variance) means the
    residual occupies a low-dimensional, structured subspace
  - a flat spectrum means the variance is spread evenly, the signature of
    numerically unstructured leftover

Run as a cell in notebook 05 (DATA_DIR already defined) or standalone.
"""

import os
import json
import numpy as np
from sklearn.decomposition import PCA

try:
    _DEFAULT_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'processed')
except NameError:  # running as a notebook cell, no __file__
    _DEFAULT_DATA_DIR = '../data/processed/'
DATA_DIR = globals().get('DATA_DIR', _DEFAULT_DATA_DIR)

MODES = ['fasttext', 'contextual', 'sliding_window', 'xglm', 'gemmax2']


def find_residual(mode, lang):
    """Locate a residual file, tolerating the two naming conventions in use."""
    long_lang = {'he': 'hebrew', 'ar': 'arabic'}[lang]
    for name in (f'{mode}_{long_lang}_residuals.npy',      # fasttext_hebrew_residuals.npy
                 f'{long_lang}_residuals_{mode}.npy'):     # hebrew_residuals_xglm.npy
        path = os.path.join(DATA_DIR, name)
        if os.path.exists(path):
            return path
    return None


manifest_path = os.path.join(DATA_DIR, 'block_manifest.json')
manifest = {}
if os.path.exists(manifest_path):
    with open(manifest_path) as f:
        manifest = json.load(f)

# Reference: what a perfectly unstructured residual of dimension d would give.
def uniform_ref(d, k):
    return min(k, d) / d

print('=' * 96)
print('RESIDUAL PCA SPECTRA')
print('  top10/50/150 : cumulative share of the residual\'s own variance')
print('  uniform      : what top10 would be if variance were spread evenly (noise reference)')
print('  k(90%)       : components needed for 90% — from block_manifest.json where available')
print('=' * 96)
print(f'{"Model":16s} {"Lang":5s} {"dim":>6s} {"top10":>8s} {"top50":>8s} {"top150":>8s} '
      f'{"uniform":>9s} {"k(90%)":>8s}  shape')
print('-' * 96)

rows = []
for mode in MODES:
    for lang in ['he', 'ar']:
        path = find_residual(mode, lang)
        if path is None:
            print(f'{mode:16s} {lang:5s}  (residual file not found)')
            continue

        R = np.load(path)
        d = R.shape[1]
        k = min(150, d)
        ratios = PCA(n_components=k).fit(R).explained_variance_ratio_
        cum = np.cumsum(ratios)

        top10  = cum[min(9, k - 1)]
        top50  = cum[min(49, k - 1)]
        top150 = cum[k - 1]
        ref10  = uniform_ref(d, 10)

        # steepness: how many times more concentrated than uniform
        concentration = top10 / ref10
        shape = ('STEEP  (structured)' if concentration > 20 else
                 'medium'              if concentration > 5  else
                 'flat   (noise-like)')

        entry = manifest.get(f'{mode}_{lang}', {})
        k90 = entry.get('res_k')
        k90_s = (f'{k90}' + ('*' if entry.get('capped') else '')) if k90 else '-'

        print(f'{mode:16s} {lang:5s} {d:>6d} {top10:>8.1%} {top50:>8.1%} {top150:>8.1%} '
              f'{ref10:>9.1%} {k90_s:>8s}  {shape}')

        rows.append(dict(mode=mode, lang=lang, dim=d, top10=top10, top50=top50,
                         top150=top150, uniform10=ref10,
                         concentration=concentration, k90=k90))

print('-' * 96)
print('* k capped at 150 before reaching 90%')
print()
print('concentration = top10 / uniform10, i.e. how many times more concentrated')
print('the leading 10 components are than an evenly spread residual of the same width.')

try:
    import pandas as pd
    df = pd.DataFrame(rows)
    out = os.path.join(DATA_DIR, 'residual_spectra.csv')
    df.to_csv(out, index=False)
    print(f'\nSaved: {out}')
except ImportError:
    pass
