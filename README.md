# Predict-the-effect-of-Genetic-Variants
Kaggle Personalized Medicine: Redefining Cancer Treatment

| Folder   | Content                                  |
| -------- | ---------------------------------------- |
| xgb      | data engineering + xgboost model         |
| xgb_res  | xgboost result / **submission** (kaggle) |
| lstm     | LSTM model                               |
| lstm_res | LSTM result ( loss / accuracy ) + saved model +  submission |
| xgb+lstm | add LSTM intermediate outputs as features; run xgboost model again |

```
├── lstm
│   ├── lstm.py
│   ├── lstm_test.py
│   └── README.md
├── lstm_res
│   ├── accuracy.png
│   ├── history_all
│   ├── history_loss_train
│   ├── history_loss_val
│   ├── inter_output_allTest
│   ├── inter_output_allTrain
│   ├── loss.png
│   ├── lstm_model.h5
│   ├── model_output_allTest
│   ├── README.md
│   ├── submission.csv
│   └── submission_raw.csv
├── README.md
├── xgb
│   ├── feature.py
│   ├── kaggle_stage2_submission.py
│   ├── README.md
│   ├── xgb_param.py
│   └── xgb.py
├── xgb+lstm
│   ├── feature2.py
│   ├── loss.png
│   ├── submission_xgb.csv
│   └── validLoss
└── xgb_res
    ├── submission.csv
    ├── validLoss
    └── valid_loss.png

```

