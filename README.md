# Mixing Languages, Mixing Signals: Hinglish Sentiment (CS 725 final phase)

This code covers the final part of the project. It continues from the mid-review
(classical models, best macro-F1 0.6909) and adds the deep models, the transformer,
the comparison, the error analysis and the base paper's WhatsApp app.

| Goal from the proposal / mid-review | Where it is |
|---|---|
| Data pipeline: SentiMix → `id, text, label, source, split` | `src/data.py` |
| Clean → normalize (learned spelling dictionary) | `src/preprocess.py` |
| LR / SVM / NB / Tree baselines (24 runs) | `src/classical.py` (`--full` for all 24) |
| **LSTM / BiLSTM, CNN** | `src/train_deep.py` |
| **Transformer (HingBERT)** | `src/train_transformer.py` |
| Classical vs deep comparison, paired bootstrap | `src/evaluate.py` |
| Error analysis: negation, long, heavily mixed, emoji, sarcasm tag, neutral confusion | `src/evaluate.py` |
| Q3: when is a transformer worth its cost? (params, train time, CPU latency, size) | `src/evaluate.py` → `results/cost.csv`, `figures/f1_vs_cost.png` |
| Base paper (Sinha et al., 2025): VADER labels + LSTM + stopwords + hybrid | `src/vader_study.py` |
| Base paper's Streamlit WhatsApp app, with our models | `app/streamlit_app.py` |
| Test on YouTube comments (out of domain) | `src/predict.py --input ...` |

---

## 1. Fastest way: Google Colab (about 12 minutes in total)

1. Open https://colab.research.google.com → **File → Upload notebook** → `colab_run_all.ipynb` (inside this zip).
2. **Runtime → Change runtime type → T4 GPU.**
3. Run the cells from top to bottom. The first cell asks you to upload this zip.

Or do it by hand in a Colab cell:

```bash
!unzip -q hinglish-sentiment.zip && cd hinglish-sentiment
!pip install -q ftfy vaderSentiment tabulate wordcloud
!bash run_all.sh
```

## 2. On your own laptop (Mac / Linux / Windows)

```bash
cd hinglish-sentiment
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
bash run_all.sh --quick        # everything except HingBERT, ~6 min on CPU
python src/train_transformer.py   # HingBERT: uses the Apple GPU (mps) on M-series Macs, ~10-15 min
python src/evaluate.py
```

Without a GPU, HingBERT on plain CPU is slow (30+ min). Train it on Colab and run the rest anywhere.

---

## 3. The steps, one by one

Run them in this order. Times are for a free Colab T4.

| # | Command | Time | What it does |
|---|---|---|---|
| 1 | `python src/data.py` | 10 s | Downloads the 3 CoNLL files + test labels from `RTT1/SentiMix`, writes `data/sentimix.csv` (14k / 3k / 3k tweets, with per-word language tags and the code-mixing index CMI). |
| 2 | `python src/classical.py` | 1-2 min | TF-IDF + SVM / LR / NB / Tree. C or alpha tuned on validation, test scored once. Saves `saved_models/svm_best.pkl` for the app. `--full` reruns all 24 mid-review runs and the dictionary-size study (~10 min). |
| 3 | `python src/vader_study.py` | 2 min | Checks the base paper's method against human labels (see section 5). |
| 4 | `python src/train_deep.py` | 2-3 min | Paper LSTM, BiLSTM, CNN, char-CNN. Early stopping on validation macro-F1. `--seeds 42 43 44` gives mean ± std. `--text normalized` to compare text versions. |
| 5 | `python src/train_transformer.py` | 3-4 min | Fine-tunes HingBERT (`l3cube-pune/hing-bert`), 3 epochs, max 64 tokens, mixed precision. |
| 6 | `python src/evaluate.py` | 20 s | Compares every model that wrote predictions. Writes `results/SUMMARY.md`. |

Optional extra transformers (about 4 min each), all compared automatically by step 6:

```bash
python src/train_transformer.py --model l3cube-pune/hing-roberta --name hingroberta
python src/train_transformer.py --model bert-base-multilingual-cased --name mbert
python src/train_transformer.py --model xlm-roberta-base --name xlmr --lr 2e-5
```

Quick check that the transformer code runs, without a GPU: `python src/train_transformer.py --max_train 1000 --epochs 1 --no_save`.

---

## 4. What you get (`results/`)

| File | Use it for |
|---|---|
| `SUMMARY.md` | All tables in one place. Copy them straight into the slides. |
| `main_results.csv` | Macro-F1, accuracy and per-class F1 for every model. |
| `slices.csv` | Macro-F1 on hard tweets: with negation, long (> 20 words), heavily mixed, with emoji, sarcasm tag. The number in `[ ]` is how many test tweets are in the group. |
| `bootstrap.csv` | Each model vs the mid-review best (SVM, cleaned, char). `significant = yes` means the 95% interval does not cross 0, so the gap is real and not luck. |
| `cost.csv` | Parameters, training time, CPU time per 1,000 tweets, model size, F1 (Q3). |
| `vader_study.csv` | Base-paper check. |
| `errors_fixed_by_<best>.csv` | Tweets the SVM got wrong and the best model got right. Use these as slide examples. |
| `errors_all_models_wrong.csv` | Tweets no model got right. Read 30-40 of them and group them (sarcasm, wrong gold label, needs world knowledge, ...). |
| `figures/confusion_matrices.png` | SVM vs the top 3 models. |
| `figures/slices.png` | Hard-tweet chart: SVM vs deep vs transformer. |
| `figures/f1_vs_cost.png` | Accuracy vs CPU cost (log scale). This is the Q3 answer slide. |

Every model writes `results/preds/<name>.csv` (id, gold, pred, probabilities), so you can add any new model and `evaluate.py` picks it up.

**Heavily mixed:** we use the code-mixing index (Das & Gambäck 2014; 0 = one language, 50 = evenly mixed) with threshold 40. The mid-review had 624 such test tweets. If you used another rule then, change it with `python src/evaluate.py --cmi 30` until the count in `slices.csv` matches.

---

## 5. Checking the base paper (`vader_study.py`)

The paper labels messages with VADER (an **English** word list), trains an LSTM on those labels,
and reports 97.89% accuracy. SentiMix has human labels, so we can check this:

1. **VADER alone vs human labels.** How good are the labels the paper trains on? The script also prints how many tweets VADER finds no sentiment word in (Hindi words are not in its list).
2. **Train on VADER labels.** Score the model against VADER labels (the paper's set-up) and then against human labels. If the first is very high and the second is low, the high accuracy only means "the LSTM learned to copy VADER". We run this with the paper's LSTM and with an SVM.
3. **Stopword removal.** The paper removes English + Hinglish stopwords. Standard lists also contain *not, no, nahi*. We compare no removal, the paper-style list, and the same list without negations, and also report F1 on tweets with negation.
4. **Hybrid.** SVM char n-grams + the 4 VADER scores as extra features.

This gives a clear "what we learned from the base paper" slide.

---

## 6. The WhatsApp app (base paper's Streamlit app, with our models)

```bash
streamlit run app/streamlit_app.py
```

- Sidebar: pick a model (`svm` is ready after step 2; `bilstm` after step 4; `hingbert` after step 5), then upload a chat export, or click **Use sample chat** (`app/sample_chat.txt`, made-up messages for the demo).
- Shows the same views as the paper: sentiment counts, monthly activity map, daily/monthly timeline, weekday activity, most positive/negative/neutral users, word clouds, most common words, a table of all messages with a CSV download, and a box to test one message.
- Export a real chat: WhatsApp → open chat → ⋮ → More → **Export chat → Without media**. Android and iPhone formats both work. For dates in month/day order, untick "Dates are day/month".

Run the app on your laptop. After Colab, the last notebook cell downloads `results_and_models.zip`; unzip it into this folder. HingBERT (~440 MB) is not in it by default, so the app then has `svm` and `bilstm`.

---

## 7. YouTube comments test (out of domain, optional)

1. Collect 150-200 Hinglish YouTube comments (for example from a cricket match or movie trailer), put them in a CSV with a `text` column, and hand-label a `label` column (negative / neutral / positive). Two people label, and you keep the ones you agree on.
2. Run the best models on it:

```bash
python src/predict.py --model svm      --input youtube.csv --output yt_svm.csv
python src/predict.py --model hingbert --input youtube.csv --output yt_hingbert.csv
```

If the file has a `label` column, the script prints precision / recall / F1. Compare with the SentiMix test score to see how much each model drops on new data.

---

## 8. Project layout

```
hinglish-sentiment/
├── README.md               this file
├── run_all.sh              runs steps 1-6
├── colab_run_all.ipynb     same, as a Colab notebook
├── requirements.txt
├── src/
│   ├── common.py           paths, labels, helpers to save predictions and costs
│   ├── data.py             download + parse SentiMix
│   ├── preprocess.py       clean / normalize / stopwords / negation words
│   ├── classical.py        TF-IDF + LR, SVM, NB, Tree
│   ├── deep_models.py      paper LSTM, BiLSTM, TextCNN, char-CNN + training loop
│   ├── train_deep.py       trains the deep models
│   ├── train_transformer.py  HingBERT (or any Hugging Face model)
│   ├── vader_study.py      base-paper checks
│   ├── evaluate.py         comparison, slices, bootstrap, cost, errors, figures
│   └── predict.py          label new text / CSV (used by the app)
├── app/
│   ├── streamlit_app.py
│   ├── whatsapp_parser.py
│   └── sample_chat.txt
├── data/        (created) raw files + sentimix.csv
├── results/     (created) tables, figures, predictions
└── saved_models/ (created) svm_best.pkl, bilstm.pt, hingbert/
```

## 9. Model details (for the report)

| Model | Input | Design | Training |
|---|---|---|---|
| Paper LSTM | words, cleaned | Embedding 128 → LSTM 100 → Dense 64 ReLU → softmax 3 (as in the paper) | Adam 2e-3, batch 64, ≤ 8 epochs, early stop (patience 2) on val macro-F1 |
| BiLSTM | words, cleaned | Embedding 128 → BiLSTM 128 → max + mean pooling → dropout 0.4 → 3 | same |
| TextCNN | words, cleaned | Kim (2014): filters 3/4/5 × 100, max pooling, dropout 0.5 | same |
| Char-CNN | characters | filters 3/5/7 × 128 over characters (no spelling problem) | same |
| HingBERT | cleaned text | `l3cube-pune/hing-bert` + classification head | AdamW 3e-5, 10% warm-up, 3 epochs, batch 32, max 64 tokens, fp16 |

Vocabulary and the spelling dictionary are learned from the training split only. Hyper-parameters
and the best epoch are picked on validation; the test set is used once.

## 10. If something goes wrong

| Problem | Fix |
|---|---|
| `data/sentimix.csv not found` | Run `python src/data.py` first. |
| Download fails (no internet / Hugging Face down) | Download the 4 files from https://huggingface.co/datasets/RTT1/SentiMix/tree/main into `data/raw/`, then `python src/data.py --local`. |
| `CUDA out of memory` in HingBERT | `--batch_size 16`. |
| HingBERT very slow | You are on CPU. In Colab: Runtime → Change runtime type → T4 GPU. |
| App says no model found | Run `python src/classical.py` (2 min). |
| `ModuleNotFoundError: vaderSentiment / ftfy / tabulate` | `pip install ftfy vaderSentiment tabulate wordcloud` |

## References

- Sinha, Jain, Pandey (2025). *Sentiment Analysis of Code-Mixed WhatsApp Conversations (Hinglish): A Hybrid Approach.* IEEE IATMSI. (base paper)
- Patwa et al. (2020). *SemEval-2020 Task 9: Overview of Sentiment Analysis of Code-Mixed Tweets.*
- Patra, Das, Das (2018). *SAIL Code-Mixed Shared Task @ICON-2017.*
- Nayak & Joshi (2022). *L3Cube-HingCorpus and HingBERT.*
- Kim (2014). *Convolutional Neural Networks for Sentence Classification.*
- Das & Gambäck (2014). *Identifying Languages at the Word Level in Code-Mixed Indian Social Media Text* (code-mixing index).
- Hutto & Gilbert (2014). *VADER.*
