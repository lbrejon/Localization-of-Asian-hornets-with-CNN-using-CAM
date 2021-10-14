# Localization-of-Asian-hornets-by-fine-tuned-CNN
Detection and localization of Asian hornets by fine-tuned CNN using PyTorch

## Table of contents 📝
* [My goals](#my-goals)
* [Acquired skills](#acquired-skills)
* [Technologies](#technologies)
* [Project composition](#project-composition)
* [Description](#description)
* [Help](#help)
* [Launch the program](#launch-the-program)
* [Sources](#sources)

Estimated reading time : ⏱️ 5min

## My goals 🎯
- Deepen knowledge in Computer Vision with CNN
- Learn how to fine-tune and train a model

## Acquired skills :zap:
- Optimization of hyper-parameters

## Technologies 🖥️
Programming languages:
```bash
- Python (framework PyTorch)
```

Librairies:
```bash
- pandas
- requests
- geopy
- bs4 (BeautifulSoup)
- flask
- sqlalchemy
```

## Project composition 📂
```bash
.
├── README.md
│
├── data
│   ├── raw
│   │   └── geoId.csv
│   │
│   ├── processed
│   │   └── geoId.csv
│   │
│   ├── jobs.csv
│   │
│   ├── jobs.json
│   │
│   └── jobs_parameters_user_request.json
│
└── notebooks
    ├── scraping_jobs.ipynb
    │
    └── scraping_jobs.py
```

## Description 📋 

## Launch the program ▶️
Create project with a virtual environment (in 'app' folder)
```
$ mkdir myproject
$ cd myproject
$ python3 -m venv flask
```
Activate it (virtual environment's name is flask)
```
$ source flask/bin/activate
```
Install requirements
```
$ pip install -r requirements.txt
```
Set environment variables in terminal (in order to not rerun code after modifications)
```
$ export FLASK_APP=app.py
$ export FLASK_ENV=development
```
Run the app
```
$ flask run
```


## Sources ⚙️
