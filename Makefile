PY ?= python
export PYTHONPATH := src

.PHONY: install data pipeline pipeline-spark dashboard test sql docker-up clean

install:            ## install dependencies
	$(PY) -m pip install -r requirements.txt

data:               ## generate the synthetic dataset
	$(PY) -m vta.data_generator

pipeline:           ## full run with the Pandas engine (~10 s)
	$(PY) -m vta.run_pipeline --engine pandas --regenerate

pipeline-spark:     ## full run with the PySpark engine (needs Java)
	$(PY) -m vta.run_pipeline --engine spark

dashboard:          ## Streamlit dashboard on http://localhost:8501
	streamlit run app/streamlit_app.py

sql:                ## print the analytical SQL reports
	$(PY) -m vta.sql_report

test:               ## unit + integration tests
	$(PY) -m pytest -q

docker-up:          ## pipeline (Spark) then dashboard, in containers
	docker compose up --build

clean:
	rm -rf data/raw data/processed data/*.db .pytest_cache spark-warehouse
