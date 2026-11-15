# Lab Practical Code

Executable code for the lab practicals. Blockchain (BT) practicals are not
included — the BT Lab Manual contains no code.

## ML (code verified against `ML_output1.pdf` printout; `output_combined.pdf` used as cross-check)

Each practical is provided twice: a runnable `.py` and the same code as a
Jupyter notebook (`.ipynb`) for Google Colab.

| Practical | .py / .ipynb | Dataset needed | Verified result vs PDF printout |
|---|---|---|---|
| 1 — Uber Fare Prediction (LR + RF) | `ML/prac_1_uber_fare.py` / `.ipynb` | `uber.csv` | shape (178573, 11) and LR metrics exact; RF metrics match to 4 decimals (library version drift) |
| 2 — Email spam (KNN + SVM) | `ML/prac_2_email_spam.py` / `.ipynb` | `emails.csv` | exact match (KNN 0.8715 / SVM 0.9671) |
| 3 — Churn (ANN, Keras) | `ML/prac_3_churn_ann.py` / `.ipynb` | `Churn_Modelling.csv` | runs end-to-end, accuracy ≈ 0.86 (no seed in the practical, so exact numbers vary run to run) |
| 4 — Diabetes (KNN) | `ML/prac_4_diabetes_knn.py` / `.ipynb` | `diabetes.csv` | exact match (CM [[80 20],[26 28]], acc 0.7013) |
| 5 — K-Means + Hierarchical | `ML/prac_5_sales_clustering.py` / `.ipynb` | `sales_data_sample.csv` | exact match (KMeans 2/0/1 = 996/975/852; Hier. 0/1/2 = 1204/934/685) |

## How the code accesses the CSV files (read before running on another machine)

Every ML file opens its dataset with a plain filename, e.g.
`pd.read_csv("uber.csv")`. That has a few consequences:

- A bare filename is resolved against the **current working directory** —
  wherever you launch Python from — **not** the folder where the `.py`/`.ipynb`
  file itself lives. (If you run from a different folder than the dataset,
  it will not find it.)
- The code **never downloads anything**. The absolute Windows paths from the
  original manuals (`C:\ML\...`) were replaced with these bare filenames so
  the same file works anywhere, but the data file must exist locally in the
  working directory (or you must give `pd.read_csv(...)` a different path).
- If the file is missing you get `FileNotFoundError: [Errno 2] No such file
  or directory: 'uber.csv'`. Nothing is silently fetched from the internet.
- Nothing about the data travels with this folder: the datasets are *not*
  included in this directory.

### On Google Colab

Colab's working directory is `/content`, and its filesystem is **ephemeral**:
uploaded files disappear when the runtime is recycled or restarted, so bring
the data back each session. Options:

- Upload the CSV into `/content` (left sidebar → Files → Upload). For the
  larger files (`uber.csv` 23 MB, `emails.csv` 30 MB) the upload dialog can
  be unreliable; prefer one of the options below.
- Mount Google Drive (`from google.colab import drive; drive.mount('/content/drive')`),
  put the CSV there, and either cd into that folder or change the filename in
  the `pd.read_csv("...")` line to the Drive path.
- Fetch it in a cell with KaggleHub (no Kaggle account needed — it works
  anonymously for public datasets), e.g.:
  `import kagglehub; kagglehub.dataset_download("yasserh/uber-fares-dataset")`
  and point the path at the downloaded folder.
- Or use the Kaggle CLI (`!kaggle datasets download -d <slug>`) — this one
  needs your own `kaggle.json` API credentials from kaggle.com → Account →
  Create New API Token.
- `pd.read_csv(...)` also accepts an `https://` URL of a raw CSV if you have
  one — then you don't need a local file at all.
- No code changes are needed if the CSV simply sits in the same folder you
  run from, with exactly the filename used in the file (case-sensitive).

### Re-downloading the datasets

The code never fetches these itself — this section is how you get the data
back (e.g. after `~/.cache/kagglehub` was wiped) so that `pd.read_csv(...)`
finds a local file. All four Kaggle datasets are public and download
anonymously with KaggleHub; files land in `~/.cache/kagglehub/datasets/...`
(the `dataset_download()` return value prints the exact folder). If
kagglehub is missing, `pip install kagglehub` first (Colab image includes it).

For each dataset, in Python:

```python
import kagglehub

path = kagglehub.dataset_download("yasserh/uber-fares-dataset")
print(path)
# -> /home/YOU/.cache/kagglehub/datasets/yasserh/uber-fares-dataset/versions/1
# the CSV is <path>/uber.csv
```

Then either copy the CSV next to the code (same folder you run from, exact
filename), or `cd` to that folder, or change the filename in the
`pd.read_csv("...")` line to the full path.

- `uber.csv` — Kaggle: `yasserh/uber-fares-dataset` — file: `<path>/uber.csv` — expected shape (200000, 9)
- `emails.csv` — Kaggle: `ashutoshkun/emailscsv` — file: `<path>/emails.csv` — expected shape (5172, 3002)
- `Churn_Modelling.csv` — Kaggle: `shantanudhakadd/bank-customer-churn-prediction` — file: `<path>/Churn_Modelling.csv` — expected shape (10000, 14)
- `sales_data_sample.csv` — Kaggle: `kyanyoga/sample-sales-data` — file: `<path>/sales_data_sample.csv` — expected shape (2823, 25)

CLI alternative (needs your own `kaggle.json` from kaggle.com → Account →
Create New API Token):

```
kaggle datasets download -d yasserh/uber-fares-dataset --unzip
# likewise: ashutoshkun/emailscsv, shantanudhakadd/bank-customer-churn-prediction, kyanyoga/sample-sales-data
```

Or download the ZIP from the dataset page on kaggle.com in a browser and unzip
it (CSV keeps the names above).

`diabetes.csv` is the Pima Indians dataset, not a Kaggle file. To rebuild it
(use `/content/diabetes.csv` naming so the practical finds it):

```python
import urllib.request
u = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
body = urllib.request.urlopen(u).read().decode()
header = "Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age,Outcome\n"
open("diabetes.csv", "w").write(header + body)
```

or use the practical's own `diabetes.csv` as-is, or change the
`pd.read_csv("diabetes.csv")` line to the raw URL (pandas reads it directly,
though it still needs the header prepend for exact compatibility).

Sanity-check any re-downloaded file with:

```python
import pandas as pd
print(pd.read_csv("uber.csv").shape)
```

Note: `sales_data_sample.csv` is not valid UTF-8 — the practical already
reads it with `encoding="ISO-8859-1"`; leave that as-is.

## DAA (from `DAA Lab Manual.pdf`)

| File | Assignment | Verified result |
|---|---|---|
| `DAA/assignment_1_fibonacci.py` | 1 — Fibonacci (non-recursive + recursive) | prints 0 1 1 2 3 5 8 for nterms=7, both methods |
| `DAA/assignment_2_huffman.py` | 2 — Huffman encoding (greedy) | matches the manual's printed output exactly (codes A:00 C:01 E:10 D:110 B:111, 168 → 45 bits) |
| `DAA/assignment_3_fractional_knapsack.py` | 3 — Fractional knapsack (greedy) | prints 240.0 (the manual's printed output "24" is a typo in the manual) |
| `DAA/assignment_4_01_knapsack.py` | 4 — 0/1 knapsack (DP) | prints 220 (matches the manual) |
| `DAA/assignment_5_n_queens.py` | 5 — N-Queens (backtracking) | prints a valid 4-queen solution |

Assignment 2's manual page shows the code as a screenshot clipped on the right
edge and `Calculate_Codes` plus the driver code are not shown anywhere in the
manual — both were reconstructed from the visible code and the manual's
printed output (verified to reproduce it exactly).

## What was fixed

Nothing was added beyond the practicals: same libraries, same models, same
logic, no extra steps and no comments of any kind (all comments, including
the ones present in the original manuals, were removed). Only:
- absolute Windows paths (`C:\ML\...`) replaced with relative paths;
- Practical 3's dataset path, which in the PDF is split across two lines
  (a syntax error as printed), replaced with `Churn_Modelling.csv`;
- Assignment 3: `__init__` and `if __name__ == "__main__":` restored (the PDF
  mangled the double underscores);
- Assignment 2: missing `Calculate_Codes` + driver reconstructed (see above);
- Assignment 1: the manual contains two separate programs (non-recursive,
  then recursive) for the same assignment — both are kept, in manual order.

Library versions used for testing: Python 3.14.10, pandas 3.0.6,
scikit-learn 1.9.1, numpy 2.5.4, matplotlib 3.11.2, seaborn 0.13.2,
tensorflow-cpu 2.22.0-rc0 (ML practical 3).
