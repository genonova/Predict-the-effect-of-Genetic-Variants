# Predict-the-effect-of-Genetic-Variants
Kaggle Personalized Medicine: Redefining Cancer Treatment

| Folder   | Content                                  |
| -------- | ---------------------------------------- |
| xgb      | data engineering + xgboost model         |
| xgb_res  | xgboost result / **submission** (kaggle) |
| lstm     | LSTM model                               |
| lstm_res | LSTM result ( loss / accuracy ) + saved model +  submission |

```
├── lstm
│   ├── lstm.py
│   └── lstm_test.py
├── lstm_res
│   ├── accuracy.png
│   ├── history_all
│   ├── history_loss_train
│   ├── history_loss_val
│   ├── inter_output_allTest
│   ├── loss.png
│   ├── lstm_model.h5
│   ├── model_output_allTest
│   ├── submission.csv
│   └── submission_raw.csv
├── README.md
├── xgb
│   ├── feature.py
│   ├── kaggle_stage2_submission.py
│   ├── xgb_param.py
│   └── xgb.py
└── xgb_res
    └── submission.csv

```

