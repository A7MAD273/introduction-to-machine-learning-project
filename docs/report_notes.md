# Report Notes

## Data loading (`src/data/load_data.py`)
- `load_raw_data()` reads `data/H2.csv`, strips padding spaces from text values (e.g. `"HB       "` → `"HB"`), turns the text `"NULL"` into a real missing value, and stops with an error if the shape is not (79,330, 31).

## Data quality (`src/data/quality.py`, `docs/data_quality_notes.md`)
- Six tables in `outputs/tables/`: column types, summary statistics, missing values, duplicates, unusual values, cardinality.
- Missing: Company 95.35%, Agent 10.25%, Country 0.03%, Children 0.01%. Missing Agent/Company means "no agent/company", not an error.

## EDA (`src/data/eda_plots.py`)
- One command plots every column on the cleaned data: `python -m src.data.eda_plots` → 49 figures in `outputs/figures/` (distribution per column, plus a correlation figure for numeric columns).
- Missing values are shown as a "Missing" bar in categorical plots.
- Skewed: LeadTime, DaysInWaitingList, PreviousCancellations, PreviousBookingsNotCanceled, BookingChanges.

## Cleaning (`src/preprocessing/clean.py`)
- **Duplicates:** removed 25,902 exact duplicates (32.65%) before the split, so the same row can't be in train and test.
- **Invalid bookings:** removed bookings with 0 guests or 0 nights.
- **ADR outliers:** IQR (9–209) would remove 2,490 rows (4.66%), mostly real summer prices. ADR is the target, and IQR on all data uses test info. Removed only ADR = 5,400 (next highest is 510), using ADR ≤ 1000.
- **Kept on purpose:**
  - ADR = 0: 722 rows after cleaning, 459 of them "Complementary" (real free stays).
  - 3 bookings with babies but no adults, and 219 with 0 adults: few rows, kept. The quality notes suggested removing the 3; the team can add this to `clean_data` if needed.
  - 3 stays longer than 30 nights: plausible.
- **Not done in cleaning (done in each task's pipeline, fitted on train only):** fill the 4 missing Children with 0; fill missing Agent with "None"; group rare Agents into "Other".
- **Different from the quality notes:** Country was suggested to be filled/grouped; it is removed instead (leakage, see below).
- **Rows:** 79,330 → 53,042.
- **Cancel rate:** 41.7% → 30.2% after cleaning; 65.8% of the removed duplicates were canceled bookings.

## Split (`src/data/split.py`)
- 80% train (42,433) / 20% test (10,609), `random_state=42`, done after cleaning.
- Why 80/20: a test set of 10.6k rows gives a stable score, and the train set stays large. Validation comes from CV, so no extra data is held out.
- Stratified on IsCanceled: 30.2% canceled in both train and test.
- Validation: 5-fold cross-validation on the train set (train/test + CV, which the spec allows). It is used for model comparison and tuning (`GridSearchCV(cv=5)`).
- The test set is used only once, for the final evaluation.

## Preprocessing (`src/preprocessing/pipeline.py`)
- `build_preprocessor(X)` returns an unfitted ColumnTransformer. It goes inside each model's Pipeline, so everything is fitted on train only.
- Numbers: fill missing with the median (the 4 missing Children → 0), then StandardScaler.
- Text: fill missing with "None" (missing Agent = no agent), then OneHotEncoder. Categories in < 1% of training rows are grouped into one "infrequent" column. Agent: 215 → 8 kept + 1 group.
- Output columns: regression 57, classification 58, clustering 50 (≤ 530, so ≥ 100 rows per variable).

## Leakage / feature selection (`src/features/selection.py`)
- Code: the three feature lists, `LEAKAGE_COLUMNS`, and `add_features()` (creates HasCompany). Run `python -m src.features.selection` to check the lists.
- **Prediction point:** for regression and classification, the moment the booking is confirmed.
- **Rule:** if a column is not surely known at that moment, or it reveals the answer, remove it.
- **Removed (reg + clf):**
  - ReservationStatus: it equals the label (Check-Out = not canceled 100%; Canceled/No-Show = canceled 100%).
  - ReservationStatusDate: set when the booking ends.
  - AssignedRoomType: decided at check-in.
  - BookingChanges: grows after booking.
  - RequiredCarParkingSpaces: usually requested at arrival.
  - TotalOfSpecialRequests: can grow after booking.
  - Country: may be recorded or corrected at arrival.
  - ArrivalDateYear: does not repeat (2015–2017 only), so it can't generalize to future bookings. With a random split it would look useful only because the same years are in train and test. Month and week number are kept because they repeat every year (season).
  - DistributionChannel: almost the same as MarketSegment (e.g. 99% of Online TA came via TA/TO), so it is redundant.
- **Not leakage:** DaysInWaitingList counts days before confirmation, so it is known at the prediction point.
- **Feature engineering:** Company is 95% empty, so it becomes HasCompany (0/1). Having a company means 12% canceled vs 31%. Rare Agent IDs are grouped into "Other".
- **Text columns:** one-hot encoded in the pipeline.
- **Evidence (cleaned data):** bookings with a parking space requested had a 0% cancel rate, and bookings whose assigned room differed from the reserved one had 5% (vs ~31%). Both are filled at arrival, so they leak the outcome.
- **DepositType "Non Refund" (97% canceled):** kept. These are not normal guests. 60% are Groups and 32% Offline TA/TO, the median lead time is ~198 days (vs 51), 98% are recorded as PRT, and almost none have special requests. This looks like agencies blocking rooms in advance and releasing them. Deposit terms are set at booking, so it is not leakage. Raw data had 12,868 such rows (12,023 duplicates); after cleaning, 844 remain (1.6%).
- **Regression:** 20 features. ADR is the target; IsCanceled is not used (it is a future outcome).
- **Classification:** the same 20 + ADR (the price is set at booking) = 21.

## Full decision table (all 31 columns)
Reg/Clf rule: keep a column if it is known when the booking is confirmed, does not reveal the answer, and is not a copy of another column. Clustering uses the 4 sourced criteria below. Numbers are from the cleaned data; the average ADR is ~112 and the average cancel rate is 30%.

| # | Column | Regression (target ADR) | Classification (target IsCanceled) | Clustering |
|---|---|---|---|---|
| 1 | IsCanceled | ✖ future outcome, unknown at booking | 🎯 target | ✖ outcome (descriptor only) |
| 2 | LeadTime | ✔ known at booking; weak alone (ADR 105–117) | ✔ ≤7 days 11% → >180 days 45% | ✔ passes all 4 |
| 3 | ArrivalDateYear | ✖ doesn't repeat; no future year in training | ✖ same | ✔ passes all 4 (keep only if clusters aren't split by year) |
| 4 | ArrivalDateMonth | ✔ season: Jan 86 → May 128 | ✔ Sep 25% → Apr 34% | ✔ passes all 4 |
| 5 | ArrivalDateWeekNumber | ✔ finer season: week 2 = 84, week 20 = 129 | ✔ 24%–37% by week | ✖ copy of month (r = 0.995) |
| 6 | ArrivalDateDayOfMonth | ✔ known; ~no effect alone (111–113) | ✔ known; ~no effect (30–31%) | ✔ passes all 4 |
| 7 | StaysInWeekendNights | ✔ known; weak alone (111–112) | ✔ 0 = 28% → 3+ = 59% | ✔ passes all 4 |
| 8 | StaysInWeekNights | ✔ known; weak alone (108–113) | ✔ 0 = 24% → 6+ = 57% | ✔ passes all 4 |
| 9 | Adults | ✔ 1 = 94 → 3 = 156 | ✔ 1 = 22%, 2 = 32% | ✔ passes all 4 |
| 10 | Children | ✔ with children 153 vs 107 | ✔ 35% vs 30% | ✔ passes all 4 (ratio 16.7) |
| 11 | Babies | ✔ known; small (116 vs 112) | ✔ with baby 17% | ✖ near-zero variance (ratio 147) |
| 12 | Meal | ✔ HB 138, BB 112, FB 54 | ✔ FB 18%, SC 36% | ✔ passes all 4 |
| 13 | Country | ✖ may be set/corrected at arrival | ✖ same (PRT 38% vs 28%) | ✖ set at arrival |
| 14 | MarketSegment | ✔ Online TA 120, Corporate 84 | ✔ Online TA 36%, Corporate 12% | ✔ passes all 4 |
| 15 | DistributionChannel | ✖ copy of MarketSegment | ✖ same | ✖ copy of MarketSegment |
| 16 | IsRepeatedGuest | ✔ repeat guests 71 vs 113 | ✔ 12% vs 31% | ✖ near-zero variance (ratio 33) |
| 17 | PreviousCancellations | ✔ 80 vs 112 | ✔ 69% vs 29% | ✖ near-zero variance (ratio 54) |
| 18 | PreviousBookingsNotCanceled | ✔ 71 vs 113 | ✔ 7% vs 31% | ✖ near-zero variance (ratio 100) |
| 19 | ReservedRoomType | ✔ room A 100 → G 204 | ✔ 24%–39% by room | ✔ passes all 4 |
| 20 | AssignedRoomType | ✖ decided at arrival | ✖ decided at arrival (changed room = 5%) | ✖ set at arrival |
| 21 | BookingChanges | ✖ grows after booking | ✖ same (17% vs 33%) | ✖ grows after booking |
| 22 | DepositType | ✔ Non Refund 92, Refundable 116 | ✔ Non Refund 97%, none 29% | ✖ near-zero variance (ratio 62) |
| 23 | Agent | ✔ grouped (top + "Other"); agent 1 = 75, agent 14 = 128 | ✔ grouped; agent 28 = 6%, agent 9 = 40% | ✔ grouped, to stay under 530 variables |
| 24 | Company | ✔ as HasCompany: 78 vs 113 | ✔ as HasCompany: 12% vs 31% | ✔ as HasCompany (raw IDs = 203 columns) |
| 25 | DaysInWaitingList | ✔ before confirmation; 87 vs 112 | ✔ 39% vs 30% | ✖ near-zero variance (ratio 987) |
| 26 | CustomerType | ✔ Transient 115, Group 94 | ✔ Transient 33%, Group 9% | ✔ passes all 4 |
| 27 | ADR | 🎯 target | ✔ known at booking; cheapest quarter 24%, most expensive 35% | ✔ passes all 4 |
| 28 | RequiredCarParkingSpaces | ✖ requested at arrival | ✖ same (with parking = 0% canceled) | ✖ set at arrival |
| 29 | TotalOfSpecialRequests | ✖ grows after booking | ✖ same (22% vs 39%) | ✖ grows after booking (descriptor only) |
| 30 | ReservationStatus | ✖ final outcome | ✖ equals the label 100% | ✖ outcome |
| 31 | ReservationStatusDate | ✖ set when the booking ends | ✖ same | ✖ outcome |

Totals: regression 20, classification 21, clustering 15. "Weak alone" columns are kept: they are known at booking and harmless, and the model can use them together with other columns (e.g. nights × season).

## Clustering feature selection
**Analysis point:** booking time, the same as the other tasks. The spec asks to justify each feature "at the stated prediction/analysis point" and to check that no clustering feature "encodes a final outcome, future event" [A].

Four criteria, each from a source:

1. **No outcome or future-event variables** [A]. These are *descriptor variables*, used only to describe the clusters afterwards, not to form them [B, Ch. 5].
2. **No near-zero-variance variables.** Noisy variables add nothing to finding segments [B, Ch. 5]. Rule: most-common/second-most-common frequency ratio > 19 (95/5) and < 10% unique values (caret `nearZeroVar` defaults) [C].
3. **No redundant or correlated variables.** Segmentation data should "contain no correlated items" [B, Ch. 5 checklist].
4. **Enough data per variable:** at least 100 rows per segmentation variable [B, Ch. 5, citing Dolnicar et al. 2016]. With 53,042 rows this allows ≤ 530 variables after one-hot encoding. The variables must also be scaled, because larger values dominate distances [A; B, Ch. 7].

| Column | Result | Criterion / evidence |
|---|---|---|
| IsCanceled, ReservationStatus, ReservationStatusDate | ✖ | 1: outcome |
| AssignedRoomType, RequiredCarParkingSpaces, Country | ✖ | 1: set at arrival |
| BookingChanges, TotalOfSpecialRequests | ✖ | 1: grow after booking |
| Babies | ✖ | 2: ratio 147 |
| IsRepeatedGuest | ✖ | 2: ratio 33 |
| PreviousCancellations | ✖ | 2: ratio 54 |
| PreviousBookingsNotCanceled | ✖ | 2: ratio 100 |
| DepositType | ✖ | 2: ratio 62 |
| DaysInWaitingList | ✖ | 2: ratio 987 |
| DistributionChannel | ✖ | 3: duplicates MarketSegment (99% of Online TA via TA/TO) |
| ArrivalDateWeekNumber | ✖ | 3: correlation 0.995 with month |
| Company → HasCompany, Agent → top agents + "Other" | ✔ | 4: raw IDs would add ~430 one-hot columns |
| LeadTime, ArrivalDateYear, ArrivalDateMonth, ArrivalDateDayOfMonth, StaysInWeekendNights, StaysInWeekNights, Adults, Children (ratio 16.7), Meal, MarketSegment, ReservedRoomType, CustomerType, ADR | ✔ | pass all 4 |

- **Clustering (15):** LeadTime, ArrivalDateYear, ArrivalDateMonth, ArrivalDateDayOfMonth, StaysInWeekendNights, StaysInWeekNights, Adults, Children, Meal, MarketSegment, ReservedRoomType, Agent (grouped), HasCompany, CustomerType, ADR.
- **Profiling after clustering:** IsCanceled, TotalOfSpecialRequests and the other removed columns can be used to *describe* each cluster (descriptor variables), not to form it [B].

### Sources
- [A] CSE382 project specification, "Clustering" and "Leakage Audit" sections.
- [B] Dolnicar, S., Grün, B., & Leisch, F. (2018). *Market Segmentation Analysis*. Springer (open access). https://link.springer.com/book/10.1007/978-981-10-8818-6
- [C] Kuhn, M. caret package, `nearZeroVar` documentation. https://search.r-project.org/CRAN/refmans/caret/html/nearZeroVar.html (method from Kuhn & Johnson, *Applied Predictive Modeling*, 2013).
