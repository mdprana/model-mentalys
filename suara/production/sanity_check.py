"""Sanity-check both H5 models with the EXACT training preprocessing pipeline.
If the models predict the right emotion, the model files are fine and the
Android-side MFCC pipeline is the bug. If they also predict 'disgust' for
everything, the model itself is broken and needs retraining.
"""
import json, os, sys, numpy as np, librosa, joblib
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf

# Monkey-patch a few Keras-2 / older-Keras-3 kwargs that the current Keras
# version no longer accepts. Must run BEFORE load_model.
def _strip_kwargs(orig_init, drop):
    def patched(self, *a, **kw):
        for k in list(kw):
            if k in drop:
                kw.pop(k, None)
        return orig_init(self, *a, **kw)
    return patched

tf.keras.layers.BatchNormalization.__init__ = _strip_kwargs(
    tf.keras.layers.BatchNormalization.__init__,
    {"renorm", "renorm_clipping", "renorm_momentum"},
)
tf.keras.layers.MultiHeadAttention.__init__ = _strip_kwargs(
    tf.keras.layers.MultiHeadAttention.__init__,
    {"use_gate"},
)
tf.keras.layers.Dense.__init__ = _strip_kwargs(
    tf.keras.layers.Dense.__init__,
    {"quantization_config"},
)

ROOT       = r"D:\1. Skripsi\model\suara\production"
MODELS_DIR = os.path.join(ROOT, "output", "models")
DATASET    = os.path.join(ROOT, "dataset")

LABELS = ["angry","calm","disgust","fearful","happy","neutral","sad","surprised"]
EMOTION_CODE = {"01":"neutral","02":"calm","03":"happy","04":"sad",
                "05":"angry","06":"fearful","07":"disgust","08":"surprised"}

# Hyperparameters used in the notebook
SR, DURATION, OFFSET = 22050, 3.0, 0.5
N_MFCC, HOP_LENGTH   = 40, 512
MAX_LEN              = 130
N_FEATURES           = 120

scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))


def featurize(wav_path: str) -> np.ndarray:
    """Replicate notebook preprocessing exactly."""
    y, sr = librosa.load(wav_path, sr=SR, duration=DURATION, offset=OFFSET)
    mfcc  = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC, hop_length=HOP_LENGTH)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    feat = np.concatenate([mfcc, delta, delta2], axis=0)  # (120, T)
    feat = feat.T  # (T, 120)
    if feat.shape[0] < MAX_LEN:
        feat = np.pad(feat, ((0, MAX_LEN - feat.shape[0]), (0, 0)))
    else:
        feat = feat[:MAX_LEN]
    feat_2d = scaler.transform(feat.reshape(-1, N_FEATURES))
    return feat_2d.reshape(1, MAX_LEN, N_FEATURES).astype(np.float32)


def show(label: str, probs: np.ndarray):
    pairs = [f"{n}={p:.3f}" for n, p in zip(LABELS, probs)]
    top = LABELS[int(np.argmax(probs))]
    print(f"  [{label:8s}] top={top:10s} | " + ", ".join(pairs))


def main():
    samples = []
    actor_dir = os.path.join(DATASET, "Actor_01")
    # Pick one representative file per emotion
    for code, name in EMOTION_CODE.items():
        cands = [f for f in os.listdir(actor_dir) if f.startswith(f"03-01-{code}-")]
        if cands:
            samples.append((name, os.path.join(actor_dir, sorted(cands)[0])))

    # Also test BOTH .tflite files: the one shipped in the app and the one in production/output.
    APP_TFLITE = r"D:\1. Skripsi\app\app\src\main\assets\models\audio_ser.tflite"
    PROD_TFLITE = os.path.join(MODELS_DIR, "audio_ser.tflite")

    # Use the newer LiteRT runtime so we can read tflite files that use
    # opcode versions newer than the bundled tf.lite supports.
    from ai_edge_litert.interpreter import Interpreter as LiteRTInterpreter
    def make_tflite_runner(path):
        try:
            interp = tf.lite.Interpreter(model_path=path)
        except Exception:
            interp = LiteRTInterpreter(model_path=path)
        interp.allocate_tensors()
        in_idx = interp.get_input_details()[0]["index"]
        out_idx = interp.get_output_details()[0]["index"]
        def run(feats):
            interp.set_tensor(in_idx, feats)
            interp.invoke()
            return interp.get_tensor(out_idx)[0]
        return run

    print("Loading H5 models ...")

    # Generic Keras-2 -> Keras-3 compat: strip kwargs that Keras 3 no longer accepts.
    DROP = {"renorm", "renorm_clipping", "renorm_momentum", "use_gate",
            "time_major", "implementation", "kernel_constraint_old"}

    def make_compat(cls):
        class _C(cls):
            def __init__(self, *a, **kw):
                for k in list(kw):
                    if k in DROP:
                        kw.pop(k, None)
                super().__init__(*a, **kw)
        _C.__name__ = cls.__name__
        return _C

    custom = {
        "BatchNormalization": make_compat(tf.keras.layers.BatchNormalization),
        "MultiHeadAttention": make_compat(tf.keras.layers.MultiHeadAttention),
        "LSTM":               make_compat(tf.keras.layers.LSTM),
        "Bidirectional":      make_compat(tf.keras.layers.Bidirectional),
    }
    # Prefer .keras (modern format) over .h5 (legacy loader path); the latter
    # dies on newer Keras kwargs even with shims.
    base = tf.keras.models.load_model(os.path.join(MODELS_DIR, "audio_ser_model.h5"),
                                       compile=False, custom_objects=custom)
    cpu  = tf.keras.models.load_model(os.path.join(MODELS_DIR, "audio_ser_cpu.h5"),
                                       compile=False, custom_objects=custom)

    print("Loading TFLite runners ...")
    app_run  = make_tflite_runner(APP_TFLITE)
    prod_run = make_tflite_runner(PROD_TFLITE)

    print(f"\nSANITY CHECK — librosa preprocessing -> all 4 model artifacts")
    print(f"{'='*92}")
    correct = {"base": 0, "cpu": 0, "app_tflite": 0, "prod_tflite": 0}
    for true_label, path in samples:
        print(f"\nfile={os.path.basename(path)}  ground_truth={true_label}")
        feats = featurize(path)
        for name, runner in (
            ("base",        lambda f: base.predict(f, verbose=0)[0]),
            ("cpu",         lambda f: cpu.predict(f, verbose=0)[0]),
            ("app_tflite",  app_run),
            ("prod_tflite", prod_run),
        ):
            probs = runner(feats)
            show(name, probs)
            if LABELS[int(np.argmax(probs))] == true_label:
                correct[name] += 1

    print(f"\n{'-'*92}")
    print(f"Accuracy on 8 reference clips:  base={correct['base']}/8  "
          f"cpu={correct['cpu']}/8  app_tflite={correct['app_tflite']}/8  "
          f"prod_tflite={correct['prod_tflite']}/8")


if __name__ == "__main__":
    main()