EMAIL SPAM CLASSIFICATION - PHASE 1

1. Copy these files into a NEW folder named cornerstone_final.
2. Copy spamassassin_sample.csv from your original project folder into cornerstone_final.
3. Open cornerstone_final in VS Code.
4. Activate your existing Windows virtual environment if you copied it, or create a new one:
       py -m venv .venv
       .venv\Scripts\activate
       py -m pip install -r requirements.txt
5. Run these commands in this exact order:
       python emailspamai.py
       python train_model.py
       python evaluate_optimize.py

IMPORTANT:
- Do not overwrite or delete your original project folder. Keep it as a backup.
- The new preprocessing splits first and fits TF-IDF only on training data.
- The optimized model is a scikit-learn Pipeline. For future Flask inference, pass cleaned email text directly to model.predict() or model.predict_proba(); do not transform it with vectorizer.pkl first.
- Metrics may differ from the milestone reports because the corrected evaluation avoids data leakage and uses a fold-safe TF-IDF pipeline. Record the new measured values; do not copy old metrics as if they were new results.
