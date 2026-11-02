# Leakage Audit — Milestone 1 (Preliminary)

> Status: preliminary audit for Milestone 1. It will be extended in Milestone 2
> with PCA/LDA, hyperparameter tuning and the final test-set evaluation.

All numbers below come from the cleaned dataset produced by
`src/preprocessing/pipeline.py` (`clean_data`), with `random_state = 42`.

---

## 1. Prediction point

**All supervised predictions are made at the moment the booking is created.**

At that moment we know who booked, for when, for how long, through which
channel, which room type was requested, and the deposit type. We do **not**
know anything that happens later: room assignment at arrival, changes made
during the booking's life, or the booking's final outcome.

Every column is therefore judged by one question:
*"Would this value be known, with this exact value, when the booking is created?"*

---

## 2. Columns excluded from all tasks

| Column | Why it leaks | Evidence from the data |
|---|---|---|
| `ReservationStatus` | It *is* the outcome (Canceled / Check-Out / No-Show). Direct copy of `IsCanceled`. | Canceled + No-Show = 16,022 rows, exactly the canceled bookings |
| `ReservationStatusDate` | Date of the final status; only exists after the booking ends or is canceled. | — |
| `AssignedRoomType` | Assigned at check-in, after the prediction point. | Bookings whose assigned room differs from the reserved room are canceled **5.3%** of the time vs **33.3%** otherwise: a room change mostly happens to guests who arrived |
| `BookingChanges` | Counts changes made during the booking's life, so its final value is only known at the end. | Bookings with ≥ 1 change: **16.6%** canceled vs **32.9%** with none |
| `RequiredCarParkingSpaces` | Parking is recorded when the guest arrives. | In the raw data, **0 of 1,926** bookings with parking were canceled (0.0%) vs 42.8% without parking |
| `Company` | Mostly missing ID code; no stable meaning as a number. | — |
| `DaysInWaitingList` | Its final value depends on how long the booking stayed on the list after creation. Excluded to stay conservative. | — |

### How `RequiredCarParkingSpaces` was found

It was first included in **Set B** (Set A + `TotalOfSpecialRequests` +
`RequiredCarParkingSpaces`). Set B looked much better for classification
(F1 0.584 → 0.681), so we tested each extra column on its own with the same
model and split:

| Features | F1 | ROC-AUC |
|---|---|---|
| Set A | 0.589 | 0.848 |
| Set A + `TotalOfSpecialRequests` | 0.671 | 0.882 |
| Set A + `RequiredCarParkingSpaces` | 0.592 | 0.857 |

Most of the gain came from `TotalOfSpecialRequests`. Parking added little
because it appears in only about 3.6% of the cleaned bookings, but it is still removed because it
reveals that the guest arrived. **Decision: Set B is redefined as Set A +
`TotalOfSpecialRequests`.**

---

## 3. Regression task — target `ADR`

**Feature set used: Set A (21 features)** from `src/features/feature_selection.py`.

| Feature group | Columns | Why it is valid at booking time |
|---|---|---|
| Timing | `LeadTime`, `ArrivalDateYear`, `ArrivalDateMonth`, `ArrivalDateWeekNumber`, `ArrivalDateDayOfMonth` | Arrival date and lead time are fixed when the booking is made; price depends on season and week |
| Stay | `StaysInWeekendNights`, `StaysInWeekNights` | Requested length of stay |
| Guests | `Adults`, `Children`, `Babies` | Party size is entered at booking |
| Product | `Meal`, `ReservedRoomType`, `DepositType` | Chosen at booking; directly affect the price |
| Customer & channel | `Country`, `MarketSegment`, `DistributionChannel`, `Agent`, `CustomerType` | Known from the booking source |
| History | `IsRepeatedGuest`, `PreviousCancellations`, `PreviousBookingsNotCanceled` | Past behaviour, known before this booking |

**Specific regression risks**

- `IsCanceled` is **not** used to predict ADR: whether the booking will be
  canceled is a future event relative to pricing.
- `AssignedRoomType` would leak strongly: the real room decides the real price.
  The model uses `ReservedRoomType` only.
- Rows with `ADR <= 0` or `ADR > 1000` are removed **for regression only**
  (723 rows). The single `ADR = 5400` row is a room type A, 1 night, 2 adults,
  BB, March booking, while the next highest ADR in the whole dataset is 510, so
  it is treated as an input error.
- Set B (+ `TotalOfSpecialRequests`) gave no improvement
  (R² 0.857 vs 0.855), so the simpler Set A is kept.

**Baseline result (XGBoost, default parameters, validation set):**
MAE 9.77, RMSE 14.98, R² 0.857.

---

## 4. Classification task — target `IsCanceled`

**Feature set used: Set B (Set A + `TotalOfSpecialRequests`).**

**Specific classification risks**

- `ReservationStatus`, `ReservationStatusDate`: direct copies of the target.
- `AssignedRoomType`, `BookingChanges`, `RequiredCarParkingSpaces`: indirect
  leakage. Each is updated by events that only happen if the guest does not
  cancel (see the evidence in Section 2).
- `ADR` is **not** used: it is the average rate over the stay and can be
  revised after booking. We exclude it to stay conservative.
- `TotalOfSpecialRequests` is **kept with justification**: requests are usually
  entered with the booking. We note that requests can also be added later;
  this is a known limitation.
- `DepositType` is kept even though *Non Refund* bookings are canceled 97.2% of
  the time: the deposit type is chosen at booking, so it is a legitimate,
  strong signal, not leakage.
- `Country` is kept but flagged: Portuguese bookings cancel more
  (37.7% vs 27.8%). If the country were confirmed only at check-in this could
  leak; the difference is moderate, so the column is kept and will be
  re-checked in Milestone 2.

**Baseline results (XGBoost, default parameters, validation set):**

| Features | F1 | ROC-AUC |
|---|---|---|
| Set A | 0.589 | 0.848 |
| Set B (Set A + `TotalOfSpecialRequests`) | 0.671 | 0.882 |

Set B is used for classification.

---

## 5. Clustering task

**Features:** `LeadTime`, `ADR`, `Adults`, `Children`, `TotalNights`
(defined in `src/models/clustering/features.py`).

- `IsCanceled`, `ReservationStatus` and `ReservationStatusDate` are **never**
  used as clustering inputs. If they were, the clusters would simply split
  canceled and non-canceled bookings, an answer we supplied rather than a
  pattern the algorithm discovered.
- `IsCanceled` may be used **only after** clustering, to describe each cluster
  (e.g. cancellation rate per cluster). It never affects the cluster assignment.
- `AssignedRoomType` and `BookingChanges` are excluded for the same reason as
  in the supervised tasks: they describe what happened later.
- `Agent` and `Company` are excluded because they are ID codes; distances
  between ID numbers have no meaning.
- Clusters are evaluated **only** with internal metrics (Silhouette and
  Davies-Bouldin, `src/evaluation/clustering_metrics.py`), never with labels.
- The `ADR = 5400` row is removed (`ADR <= 1000`), the same rule as regression.
- Clustering has no train/test split, so the `StandardScaler` is fitted on all
  rows. No target is involved, so this does not leak a label.

---

## 6. Separation of preprocessing, tuning and evaluation

| Step | How leakage is prevented |
|---|---|
| Duplicates | 25,902 duplicate rows are removed **before** splitting, so the same booking cannot appear in both train and test |
| Split | 80% development / 20% test (`split_data`, `random_state=42`). The development part is split again 75% train / 25% validation. Classification uses `stratify` |
| Test set | Not used in Milestone 1. Reserved for one final evaluation in Milestone 2 |
| Imputation, encoding, scaling | Inside a scikit-learn `Pipeline` (`create_preprocessor`), so they are fitted on the training data only |
| Unknown categories | `OneHotEncoder(handle_unknown="ignore")`: new categories at inference do not break the model and are not learned from validation/test |
| Tuning (Milestone 2) | Cross-validation on the development set only, with the whole `Pipeline` inside the search so each fold refits its own preprocessing |
| PCA / LDA (Milestone 2) | Added as a `Pipeline` step, fitted on training folds only |

### Known limitation

The split is random, not time-based. Bookings from 2015–2017 are mixed, so the
model can learn from bookings that arrive later than some validation bookings.
A time-based split would be stricter; this will be discussed in Milestone 2.