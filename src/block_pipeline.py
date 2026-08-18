"""
block_pipeline.py
-----------------
Fold-safe feature-block preprocessing for the cross-lingual encoding models.

Why this exists
===============
The original pipeline concatenated feature spaces and then applied a single
PCA to the result. Because the cross-lingual residual is orthogonal to the
English embedding by construction (least-squares normal equations give
E.T @ R = 0), the covariance of [E | R] is block-diagonal. A joint PCA
therefore cannot form mixed components -- it can only *select* between the
two blocks. Under a fixed component budget, residual components displace
English components rather than supplementing them.

This module instead reduces each feature space *within itself*, and does so
inside the cross-validation fold, so nothing is fit on held-out words.

Two transformers are provided:

    BlockPCA          -- ordinary PCA restricted to one block of columns
    EnglishResidual   -- given [E | F], returns F - E @ W, with W fit on the
                         training rows only

Usage
=====
    from block_pipeline import make_block_transformer

    spec = [
        {'kind': 'pca',      'cols': (0, 768),  'k': 150},
        {'kind': 'residual', 'en_cols': (0, 768),
                             'foreign_cols': (768, 1536), 'k': 145},
    ]
    pre = make_block_transformer(spec)
    model = make_pipeline(pre, StandardScaler(), RidgeCV(...))

The design matrix X passed to the pipeline is the raw concatenation
[English | Foreign] -- no residual is precomputed. The residual is derived
inside each training fold.
"""

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline


class EnglishResidual(BaseEstimator, TransformerMixin):
    """
    Given a design matrix whose first `n_en` columns are the English
    embedding and whose remaining columns are a foreign-language embedding,
    return the part of the foreign embedding that English cannot explain.

    W is estimated by least squares on the rows seen during `fit` only, so
    when this sits inside a cross-validated pipeline the projection is
    learned from training words and applied to test words.

    No intercept is fitted, matching the projection used in notebook 05.
    """

    def __init__(self, n_en):
        self.n_en = n_en

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=np.float64)
        E, F = X[:, :self.n_en], X[:, self.n_en:]
        self.W_, *_ = np.linalg.lstsq(E, F, rcond=None)
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=np.float64)
        E, F = X[:, :self.n_en], X[:, self.n_en:]
        return (F - E @ self.W_).astype(np.float32)


def _cols(rng):
    """(start, end) -> explicit integer column list (safe for numpy input)."""
    start, end = rng
    return list(range(start, end))


def make_block_transformer(spec):
    """
    Build a ColumnTransformer that reduces each feature block independently.

    Parameters
    ----------
    spec : list of dict
        Each entry is one output block. Two kinds are supported:

        {'kind': 'pca', 'cols': (start, end), 'k': int or None}
            PCA on those columns. k=None (or k >= block width) passes the
            block through untouched -- used for FastText's native 150d
            English block, which the original pipeline also left unreduced.

        {'kind': 'residual', 'en_cols': (a, b), 'foreign_cols': (c, d),
         'k': int or None}
            Residualise the foreign block against the English block, then
            PCA the residual to k components.

    Returns
    -------
    sklearn.compose.ColumnTransformer
    """
    transformers = []

    for i, block in enumerate(spec):
        kind = block['kind']
        k = block.get('k')

        if kind == 'pca':
            cols = _cols(block['cols'])
            width = len(cols)
            if k is None or k >= width:
                transformers.append((f'block{i}_passthrough', 'passthrough', cols))
            else:
                transformers.append(
                    (f'block{i}_pca', PCA(n_components=k, random_state=0), cols)
                )

        elif kind == 'residual':
            en_cols = _cols(block['en_cols'])
            fo_cols = _cols(block['foreign_cols'])
            cols = en_cols + fo_cols
            n_en = len(en_cols)
            width = len(fo_cols)

            steps = [('residual', EnglishResidual(n_en=n_en))]
            if k is not None and k < width:
                steps.append(('pca', PCA(n_components=k, random_state=0)))
            transformers.append((f'block{i}_residual', Pipeline(steps), cols))

        else:
            raise ValueError(f"unknown block kind: {kind!r}")

    return ColumnTransformer(transformers, remainder='drop')


def describe_spec(spec):
    """One-line-per-block human summary, for logging into the run manifest."""
    lines = []
    for block in spec:
        if block['kind'] == 'pca':
            a, b = block['cols']
            k = block.get('k')
            lines.append(f"pca      cols[{a}:{b}] width={b-a} -> "
                         f"{'passthrough' if k is None or k >= b - a else k}")
        else:
            a, b = block['en_cols']
            c, d = block['foreign_cols']
            k = block.get('k')
            lines.append(f"residual en[{a}:{b}] foreign[{c}:{d}] width={d-c} -> "
                         f"{'passthrough' if k is None or k >= d - c else k}")
    return '\n'.join(lines)