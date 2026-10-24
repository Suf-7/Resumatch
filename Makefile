PY ?= .venv/bin/python
S = data_scripts

.PHONY: setup data scrape refresh synthetic clean features live ats label eda gold report all

setup:
	python3 -m venv .venv && .venv/bin/pip install -r requirements.txt

data:             ## download Kaggle datasets (HF mirrors if no Kaggle creds) + ESCO
	cd $(S) && ../$(PY) download_kaggle.py && ../$(PY) download_esco.py

scrape:           ## polite Rozee.pk scrape (robots.txt, 2 s delay)
	cd $(S) && ../$(PY) scrape_jobs.py --max-jobs 400

synthetic:        ## needs LLM_API_KEY in .env
	cd $(S) && ../$(PY) generate_synthetic_cvs.py --per-role 20

clean:
	cd $(S) && ../$(PY) clean_resumes.py && ../$(PY) clean_jobs.py

features:
	cd $(S) && ../$(PY) build_features.py --embeddings

live:             ## open, applyable Rozee.pk postings users are matched against
	cd $(S) && ../$(PY) live_pool.py

ats:              ## baseline ATS score sanity check on the test pairs
	cd $(S) && ../$(PY) ats_score.py evaluate

refresh:          ## run every few days: Rozee.pk postings close within about a week
	cd $(S) && ../$(PY) scrape_jobs.py --refresh --max-jobs 150
	$(MAKE) clean features live

label:            ## needs LLM_API_KEY in .env
	cd $(S) && ../$(PY) label_pairs.py --n 2000

eda:
	cd $(S) && ../$(PY) eda.py

gold:
	cd $(S) && ../$(PY) sample_gold_set.py sample --n 200

report:
	$(PY) report/make_numbers.py && $(PY) report/build_pdf.py

all: data scrape clean features live ats eda gold report
