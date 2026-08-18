import mne
import numpy as np
import pandas as pd

# Machine Learning & Stats
from himalaya.backend import set_backend, get_backend
from himalaya.ridge import RidgeCV
from himalaya.scoring import correlation_score

from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.ndimage import uniform_filter1d

from mne_bids import BIDSPath

def process_embeddings(
    raw = None,
    subj="03",
    channel_names_regex="",
    bids_root="",
    embedding_df = None,
    embedding_filename = "podcast_trilingual_embeddings.csv",
    language_mode="en", # Options: "en", "he", "ar", "noise", "en+he", "en+ar", "en+noise", "all"
    random_noise_mode="over all embeds",  # other option: "per embed"
    freq=32,
    tmin=-1.0,
    tmax=1.0,
    use_PCA=False,
    PCA_dim=150,
    feature_blocks=None,   
    ):

    # 1. Backend Selection
    try:
        import torch
        if torch.cuda.is_available():
            set_backend("torch_cuda")
            print("Himalaya backend: Using CUDA (GPU)")
        else:
            set_backend("torch")
            print("Himalaya backend: Using Torch (CPU)")
    except ImportError:
        set_backend("numpy")
        print("Himalaya backend: Using Numpy (CPU)")

    def parse_array(s):
        if not isinstance(s, str): return s
        s = s.replace("\n", " ").strip("[]").strip()
        return np.array([float(x) for x in s.split() if x], dtype=np.float32)

    # 2. Loading Data Sources
    # We load based on the language_mode to save memory
    df = embedding_df
    if df is None:
        df = pd.read_csv(embedding_filename, converters={
            "en_embedding": parse_array,
            "he_embedding": parse_array,
            "ar_embedding": parse_array})

    embeddings_to_concat = []
    en_matrix = np.stack(df['en_embedding'].values)

    if "en" in language_mode or language_mode == "all":
        embeddings_to_concat.append(en_matrix)
    if "he" in language_mode or language_mode == "all":
        embeddings_to_concat.append(np.stack(df['he_embedding'].values))
    if "ar" in language_mode or language_mode == "all":
        embeddings_to_concat.append(np.stack(df['ar_embedding'].values))
    if "noise" in language_mode:
        if random_noise_mode == "per embed":
            row_means = en_matrix.mean(axis=1, keepdims=True)
            row_stds = en_matrix.std(axis=1, keepdims=True)
            noise_matrix = np.random.normal(row_means, row_stds, en_matrix.shape)
        else:
            g_mean, g_std = en_matrix.mean(), en_matrix.std()
            noise_matrix = np.random.normal(g_mean, g_std, en_matrix.shape)
            
        embeddings_to_concat.append(noise_matrix)

    # 3. Final Matrix Assembly
    if len(embeddings_to_concat) > 1:
        final_embeddings = np.concatenate(embeddings_to_concat, axis=1)
    else:
        final_embeddings = embeddings_to_concat[0]

    # 4. Dimensionality Reduction via PCA (skipped when feature_blocks is used)
    if feature_blocks is None and use_PCA is True and final_embeddings.shape[1] > PCA_dim:
        print(f"Before PCA: {final_embeddings.shape}")
        pca = PCA(n_components=PCA_dim)
        final_embeddings = pca.fit_transform(final_embeddings)
        print(f"After PCA: {final_embeddings.shape}")

    # 5. Neural Data Loading
    if raw is None:
        file_path = BIDSPath(root=f"{bids_root}derivatives/ecogprep", 
                        subject=subj, task="podcast", datatype="ieeg", description="highgamma", 
                        suffix="ieeg", extension=".fif")
        
        if not len(bids_root):
            file_path = file_path.basename

        raw = mne.io.read_raw_fif(file_path, verbose=False)
    
    if channel_names_regex != "":
        picks = mne.pick_channels_regexp(raw.ch_names, channel_names_regex)
        raw = raw.pick(picks)

    # 6. Epoching
    events = np.zeros((len(df), 3), dtype=int)
    events[:, 0] = (df['start'] * raw.info['sfreq']).astype(int)

    epochs = mne.Epochs(
        raw, events, tmin=tmin, tmax=tmax,
        baseline=None, proj=False, preload=True,
        event_repeated='drop', verbose=False
    )
    
    epochs = epochs.resample(sfreq=freq, npad='auto', method='fft', window='hamming')
    epochs_data = epochs.get_data(copy=True)          # (n_epochs, n_ch, n_times)

    # Average the neural response over a 200 ms window at each lag.
    # Raises target SNR; adjacent lags become correlated.
    SMOOTH_MS = 200
    w = max(1, int(round(SMOOTH_MS / 1000 * freq)))   # 13 samples at 64 Hz
    if w > 1:
        epochs_data = uniform_filter1d(epochs_data, size=w, axis=-1, mode='nearest')

    n_epochs, n_channels, n_times = epochs_data.shape

    # Flatten neural data for Ridge (n_samples, n_features_neural)
    Y = epochs_data.reshape(n_epochs, -1)
    X = final_embeddings[epochs.selection]

    if "torch" in get_backend().__name__:
        X = X.astype(np.float32)
        Y = Y.astype(np.float32)

    # 7. Encoding Model (RidgeCV)
    alphas = np.logspace(1, 10, 10)
    inner_cv = KFold(n_splits=5, shuffle=False)
    if feature_blocks is None:
        model = make_pipeline(StandardScaler(),
                            RidgeCV(alphas, fit_intercept=True, cv=inner_cv))
    else:
        from block_pipeline import make_block_transformer
        model = make_pipeline(make_block_transformer(feature_blocks),
                            StandardScaler(),
                            RidgeCV(alphas, fit_intercept=True, cv=inner_cv))
    
    epochs_shape = (n_channels, n_times)

    def train_encoding(X_in, Y_in):
        corrs, Yt_pool, Yp_pool = [], [], []
        kfold = KFold(10, shuffle=False)
        for train_idx, test_idx in kfold.split(X_in):
            X_train, X_test = X_in[train_idx], X_in[test_idx]
            Y_train, Y_test = Y_in[train_idx], Y_in[test_idx]

            scaler  = StandardScaler()
            Y_train = scaler.fit_transform(Y_train)
            Y_test  = scaler.transform(Y_test)

            model.fit(X_train, Y_train)
            Y_preds = model.predict(X_test)
            corr = correlation_score(Y_test, Y_preds).reshape(epochs_shape)

            if "torch" in get_backend().__name__:
                corr    = corr.numpy(force=True)
                Y_preds = np.asarray(Y_preds.numpy(force=True))
            corrs.append(corr)
            Yt_pool.append(np.asarray(Y_test));  Yp_pool.append(np.asarray(Y_preds))

        return (np.stack(corrs),
                np.concatenate(Yt_pool, 0).astype(np.float32),
                np.concatenate(Yp_pool, 0).astype(np.float32))

    cv_scores, Y_true_pooled, Y_pred_pooled = train_encoding(X, Y)
    return raw, cv_scores, Y_true_pooled, Y_pred_pooled


def zc(A):
    A = np.asarray(A, np.float64)
    A = A - A.mean(0, keepdims=True)
    return A / (A.std(0, keepdims=True) + 1e-12)

def circular_null(Y_true, Y_pred, shape, exclude=50):
    """
    Exact null over all circular shifts of the word sequence.

    shape : (n_electrodes, n_times)

    Returns dict with
      observed        (n_elec, n_times)  observed r
      p_lagwise       (n_elec, n_times)  p per electrode-lag
      p_electrode     (n_elec,)          Goldstein-style: observed max-across-lags
                                         vs null of max-across-lags-and-electrodes
      null_max        (n_perm,)          the max-statistic null distribution
      q95, q99        (n_elec, n_times)
    """
    n = len(Y_true)
    a, b = zc(Y_true), zc(Y_pred)
    D = np.fft.irfft(np.fft.rfft(b, axis=0) * np.conj(np.fft.rfft(a, axis=0)),
                     n=n, axis=0) / n

    obs   = D[0].reshape(shape)
    valid = D[np.r_[exclude:n - exclude]].reshape(-1, *shape)   # (n_perm, n_elec, n_times)

    null_max    = valid.max(-1).max(-1)                 # max over lags, then electrodes
    obs_max     = obs.max(-1)                           # (n_elec,)
    p_electrode = (null_max[:, None] >= obs_max[None, :]).mean(0)

    return dict(observed=obs,
                p_lagwise=(valid >= obs[None]).mean(0),
                p_electrode=p_electrode,
                null_max=null_max,
                q95=np.percentile(valid, 95, 0),
                q99=np.percentile(valid, 99, 0))