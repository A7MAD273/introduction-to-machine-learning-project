# Regression Report: ADR (Average Daily Rate) Prediction

Course: CSE382, City Hotel (H2.csv). Branch: `regression`.

## 1. Task definition and prediction point

**Goal:** predict the ADR (price per night) of a booking.

**Prediction point:** the moment the booking is confirmed. Only information known at that moment may be used as input.

**Scope:** paid bookings only (ADR > 0). Free or complimentary stays follow a different rule (the price is zero by arrangement, not by market), so they are outside the question "what will this booking cost per night?".

## 2. Data and cleaning

| Stage | Rows |
|---|---|
| Raw H2.csv | 79,330 (31 columns) |
| After team cleaning (duplicates, bookings with no guests or no nights, ADR > 1000) | 53,042 |
| Train / Test (80/20, seed 42) | 42,433 / 10,609 |
| After removing ADR = 0 (this task only) | 41,861 / 10,459 |

**ADR = 0 decision.** The training set had 572 rows with ADR = 0 (no negative values). 356 of them belong to the Complementary market segment, where 93.2% of bookings have ADR = 0, so they are free stays. The rest are spread over other segments at low rates (0.1% to 4.0% of each segment). We removed all of them. The rule is fixed and does not learn anything from the data, so it is applied to both train and test without leakage.

**ADR distribution (train, before the ADR = 0 filter).** Mean 111.78, median 106, skew 0.78 (mild), maximum 510. We did not transform the target, so MAE stays in price units.

## 3. Features

The model uses 20 features: LeadTime, ArrivalDateMonth, ArrivalDateWeekNumber, ArrivalDateDayOfMonth, StaysInWeekendNights, StaysInWeekNights, Adults, Children, Babies, Meal, MarketSegment, IsRepeatedGuest, PreviousCancellations, PreviousBookingsNotCanceled, ReservedRoomType, DepositType, Agent, HasCompany, DaysInWaitingList, CustomerType.

`HasCompany` is a 0/1 flag derived from `Company`, which is 95% empty.

## 4. Leakage Audit (regression)

### 4.1 Columns excluded and why

| Column | Reason |
|---|---|
| ReservationStatus | Equals the final outcome of the booking (direct leakage) |
| ReservationStatusDate | Set when the booking ends |
| AssignedRoomType | Decided at arrival, after the prediction point |
| BookingChanges | Grows after the booking is made |
| RequiredCarParkingSpaces | Requested at arrival |
| TotalOfSpecialRequests | Grows after the booking is made |
| Country | May be set or corrected at arrival |
| ArrivalDateYear | Not leakage, but it never repeats, so it cannot generalize to future years |
| DistributionChannel | Not leakage, but it is almost a copy of MarketSegment |
| IsCanceled | The target of the classification task. It is not a feature here. It is only used to stratify the split so all tasks share the same train/test rows |

### 4.2 Risks we checked

- **Direct target leakage:** no column derived from ADR is used (for example total revenue). ADR itself is only the target.
- **Timing leakage:** every feature is known when the booking is confirmed. `DaysInWaitingList` is kept because the waiting period ends at confirmation, which is our prediction point. This is the most debatable feature and we state it explicitly.
- **Indirect leakage through duplicates:** exact duplicate rows are removed before the split, so the same booking cannot appear in both train and test.
- **Leakage through preprocessing:** median values, scaling and category frequencies are learned inside the sklearn Pipeline, so in every cross-validation fold they are fitted only on that fold's training part.
- **Leakage through tuning:** hyperparameters are chosen with 3-fold cross-validation on the training set only. The test set is not used for any decision.

### 4.3 How the stages are separated

| Stage | Data used |
|---|---|
| Cleaning (fixed rules) | Applied before the split, no statistics learned |
| Preprocessing (median, scaler, one-hot) | Fitted on training data only, inside the Pipeline |
| Baseline and tuning | Cross-validation on the training set |
| Final fit | Training set only |
| Model selection | Cross-validation MAE (training set) |
| Final evaluation | Test set, once (see section 8) |

## 5. Preprocessing

- Numeric columns: missing values filled with the median, then StandardScaler.
- Categorical columns: missing values filled with "None", then one-hot encoding. Categories seen in less than 1% of training rows are grouped into one "infrequent" column.
- The preprocessor and the model are one `Pipeline` object, so the same steps are applied in training, cross-validation and inference.

## 6. Experimental protocol

- Split: 80/20, `random_state=42` (shared by the whole team).
- Cross-validation: 3-fold KFold with shuffle, seed 42, on the training set.
- Metrics: MAE, RMSE, R2, plus fit time.
- Models: Ridge (linear), Random Forest (bagging ensemble), XGBoost (gradient boosting).
- Baseline: library default settings (random_state fixed).
- Tuning: `RandomizedSearchCV`, 20 random combinations per model, optimized for MAE. The search spaces are in `src/regression/tune.py` and the best configurations are saved in `outputs/regression_best_params.json`.

### Search spaces

| Model | Hyperparameters searched |
|---|---|
| Ridge | alpha: log-uniform 0.01 to 100 |
| Random Forest | n_estimators 100-399, max_depth {None, 10, 20, 30}, min_samples_split 2-10, min_samples_leaf 1-5, max_features {1.0, 0.5, 0.3} |
| XGBoost | n_estimators 200-699, max_depth 3-10, learning_rate log-uniform 0.02 to 0.3, subsample 0.6-1.0, colsample_bytree 0.5-1.0, min_child_weight 1-10 |

## 7. Results: baseline vs tuned (3-fold CV on the training set)

| Model | Baseline MAE | Tuned MAE | Tuned RMSE | Tuned R2 | Baseline R2 |
|---|---|---|---|---|---|
| Ridge | 15.955 | 15.873 | 22.395 | 0.681 | 0.679 |
| Random Forest | 10.787 | 10.831 | 17.797 | 0.799 | 0.797 |
| XGBoost | 11.220 | **9.815** | **16.171** | **0.834** | 0.812 |

Baseline RMSE: Ridge 22.474, Random Forest 17.887, XGBoost 17.182.

Training and search times (seconds): Ridge 0.14 baseline fit, 0.14 tuned fit, 10.6 search. Random Forest 1.18 baseline fit, 1.53 tuned fit, 93.5 search. XGBoost 0.54 baseline fit, 1.61 tuned fit, 64.4 search.

**Interpretation.**
- Ridge is the weakest because ADR does not depend linearly on the inputs (season, room type and guest count interact). Tuning its only parameter changes almost nothing.
- Random Forest did not improve after tuning (10.787 to 10.831, a difference of 0.04 which is within fold-to-fold noise). Its default settings were already close to a good configuration and a random search with 20 combinations did not find a clearly better one. We report this result as it is.
- XGBoost benefited most from tuning (MAE 11.220 to 9.815) because its defaults are not ideal for this data.
- An MAE of about 9.8 means the model is wrong by roughly 10 per night on average, against a mean ADR of about 112.

## 8. Final test-set evaluation

TODO: fill from `outputs/regression_final_test.csv` (produced by `evaluate.py`). The test set is evaluated once, after the model was selected.

## 9. Model selection

**Criterion:** lowest tuned cross-validation MAE on the training set. Ties would be broken by RMSE, then R2.

**Selected model:** XGBoost (tuned CV MAE 9.815). The choice is stored in `outputs/regression_selected.json` and the inference code reads it, so the model name is not hard-coded.

## 10. Inference

`python -m src.inference.predict_regression` loads the saved pipeline (preprocessing and model together) with joblib and predicts the ADR of a new booking. It does not retrain. Without `--json` it uses a built-in sample booking, and with `--json path` it reads the 20 booking fields from a file.

## 11. Reproduction commands

    python -m pip install -r requirements.txt
    python -m src.regression.dataset
    python -m src.regression.baseline
    python -m src.regression.tune
    python -m src.regression.final_fit
    python -m src.inference.predict_regression

Trained model files (`artifacts/*.joblib`) are not stored in the repository. `final_fit` regenerates them.

## 12. Limitations

- Random search with 20 combinations per model may miss better configurations, especially for Random Forest.
- `DaysInWaitingList` depends on the interpretation of the prediction point (see section 4.2).
- Cross-validation uses 3 folds to keep the experiments fast.
