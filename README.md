# City Hotel Booking Demand: Regression, Classification, and Clustering

CSE382 – Introduction to Machine Learning, end-to-end project using the City Hotel (H2) dataset.

## Team

| Name | Student ID | GitHub username |
|------|------------|-----------------|
|Ahmed khaled mohamed abdullfatah elraiany     |23p0364            |A7MAD273                 |
|Tervina Samir Bishay Fahem     |23p0363            |Tervinasamir65                 |
|Omar kashef ibrahim said labeb     |23P0298            |                 |
|Mohamed Ashraf salem Handok      |23p0377            |m7md19006-eng                 |
|Mahmoud Adel Kamal Essa      |23p0421            |Mahmoud-Adel793                 |
|Juliana Hany Saad El Deen Waheeb      |23p0267            |julianasaad77-cyber                 |

## Repository

- GitHub: https://github.com/A7MAD273/introduction-to-machine-learning-project
- Submitted commit SHA: <to be added at submission>

## Setup

Requires Python 3.11.

**Windows**

```
py -3.11 -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

**Mac / Linux**

```
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Data

Place `H2.csv` in the `data/` folder.

- Direct download: https://raw.githubusercontent.com/afaq-ahmad/Hotel-booking-demand-Case-Study/master/H2.csv
- Dataset article: https://pmc.ncbi.nlm.nih.gov/articles/PMC6297060/

## Usage

Run all commands from the project root, with the virtual environment active.

**Data preparation (working now)**

```
python -m src.data.load_data          # load H2.csv and check its shape
python -m src.data.quality            # data quality tables -> outputs/tables/
python -m src.data.eda_plots          # EDA figures -> outputs/figures/
python -m src.preprocessing.clean     # remove duplicates and invalid bookings
python -m src.data.split              # 80/20 stratified train/test split
python -m src.features.selection      # check the feature lists for each task
```

**Application (Milestone 2, not implemented yet)**

```
python main.py analyze
python main.py train --task regression --model all
python main.py train --task classification --model all
python main.py cluster --algorithm kmeans
python main.py predict --task regression
python main.py predict --task classification
```

## Project structure

```
├── README.md
├── requirements.txt
├── config.yaml
├── main.py
├── data/            raw dataset
├── docs/            notes and reports
├── src/
│   ├── data/            loading and validation
│   ├── preprocessing/   pipelines, encoding, scaling
│   ├── features/        feature selection and engineering
│   ├── regression/      ADR models
│   ├── classification/  IsCanceled models
│   ├── clustering/      clustering algorithms
│   ├── tuning/          hyperparameter search
│   ├── evaluation/      metrics, tables, plots
│   └── inference/       loading artifacts and predicting
├── artifacts/       saved models and pipelines (not uploaded)
├── outputs/         result tables and figures
└── tests/
```

