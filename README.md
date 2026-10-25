# Mentalys Models

Training notebooks and exported TensorFlow Lite models behind [Mentalys](https://github.com/mdprana/mentalys), an Android app for early depression-risk screening. The app combines a PHQ-9 questionnaire with two on-device emotion recognizers: one for facial expressions and one for speech.

Built for my undergraduate thesis at Universitas Udayana: *Implementasi Pendekatan Multimodal Deep Learning dalam Rancang Bangun Sistem Skrining Awal Kesehatan Mental (Risiko Depresi) Berbasis Android*.

## Models

| Modality | Architecture | Dataset | Classes | Test accuracy | TFLite size |
|---|---|---|---|---|---|
| Face | VGG-16 + Squeeze-and-Excitation + spatial attention (CBAM-style) | FER-2013 + CK+ | 7 | 68.33% (69.25% with TTA), macro AUC 0.917 | 15.2 MB |
| Voice | CNN-BiLSTM + multi-head self-attention | RAVDESS speech | 8 | 97.22% | 7.3 MB |

Face classes: angry, disgust, fear, happy, neutral, sad, surprise.
Voice classes: angry, calm, disgust, fearful, happy, neutral, sad, surprised.

On a Xiaomi 2405CPCFBG (Android 15), face inference takes about 0.33 s (about 1.0 s including detection and preprocessing). Voice inference takes about 0.02 s (about 0.64 s including MFCC extraction).

## Repository layout

```text
wajah/production/                 Face model
  fer_vgg16_v2.ipynb              Training notebook (EDA, tuning, 4-phase fine-tuning, TTA, export)
  output/models/fer_vgg16_attention_best.tflite
  output/model_metadata.json      Labels, input shape, preprocessing, hyperparameters, class weights
  output/classification_report.txt
  output/plots/                   EDA, training curves, confusion matrices, ROC
  output/logs/                    Per-phase training history

suara/production/                 Voice model
  ravdess_audio_ser.ipynb         Training notebook (EDA, augmentation, features, tuning, export)
  audio_ser.tflite                Model shipped in the app
  sanity_check.py                 Checks the Keras models against the training preprocessing
  output/models/                  Label map, scaler parameters, best hyperparameters
  output/plots/                   EDA, training curves, confusion matrix, error analysis
```

## Face model

- Input `[1, 224, 224, 3]` float32. VGG-16 preprocessing: RGB to BGR, subtract mean `[103.939, 116.779, 123.68]`.
- Hyperparameters chosen with 20-trial Bayesian optimization (KerasTuner): 1 dense layer of 512 units, dropout 0.6, L2 5e-4, SE ratio 16, learning rate 2e-3.
- Progressive fine-tuning in 4 phases that unfreeze deeper VGG-16 blocks. Class weights offset the small `disgust` class.
- Full metrics are in `wajah/production/output/classification_report.txt` and `per_class_metrics.csv`.

## Voice model

- Input `[1, 130, 120]`: 40 MFCC plus delta and delta-delta, padded to 130 frames (about 3 s at 22,050 Hz, hop 512). Features are standardized using `scaler_params.json`.
- Six versions of every clip (the original and five augmentations), split 70/15/15 stratified.
- The TFLite export uses TensorFlow Select ops. On Android it needs the Flex delegate (`tensorflow-lite-select-tf-ops`).

## How the app uses the outputs

Each model's probabilities become a risk score for its modality. For voice the score is `P(sad) + 0.5·P(neutral) − P(happy)`, clipped to `[0, 1]`. The app then fuses the PHQ-9, face and voice scores. See `RiskFusionEngine.kt` and `RiskMapper.kt` in the [app repo](https://github.com/mdprana/mentalys) for the weights and thresholds.

## Reproducing

1. Download the datasets into the folders the notebooks expect:
   - [RAVDESS](https://zenodo.org/records/1188976) speech audio into `suara/production/dataset/audio_speech_actors_01-24/`
   - [FER-2013](https://www.kaggle.com/datasets/msambare/fer2013) into `wajah/production/dataset/fer2013/` and [CK+](https://www.kaggle.com/datasets/shawon10/ckplus) into `wajah/production/dataset/ck+/`
2. Install Python 3.10+ with `tensorflow`, `keras-tuner`, `librosa`, `scikit-learn`, `pandas`, `matplotlib`, `seaborn` and `joblib`.
3. Run the notebooks top to bottom. They write checkpoints, features and TFLite files to `output/`.

Datasets, Keras/H5 checkpoints, cached features and tuning trials are not included because many of them exceed GitHub's 100 MB limit. The notebooks regenerate all of them.

## Limitations

- The voice split is done after augmentation, so augmented copies of the same recording can end up in both train and test. The 97.22% figure is therefore optimistic. A speaker-independent split (holding out whole actors) would give a more realistic estimate.
- RAVDESS is acted speech from 24 North American actors, and FER-2013 is low-resolution web images. Neither matches Indonesian users in natural conditions.
- These models recognize emotions, not depression. Mentalys is a screening aid, not a diagnostic tool.

## Author

Made Pranajaya Dibyacita, Informatics, Universitas Udayana.
