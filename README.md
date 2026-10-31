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

- GitHub: <repository link>
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
