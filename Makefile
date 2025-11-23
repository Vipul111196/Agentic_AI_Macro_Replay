.PHONY: install train test dashboard docker-up docker-down clean

install:
	pip install -r requirements.txt

train:
	python train.py

test:
	python test.py

dashboard:
	KMP_DUPLICATE_LIB_OK=TRUE streamlit run app_dashboard.py --server.port=8501

docker-up:
	docker compose up --build -d

docker-down:
	docker compose down

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; true
	rm -f models/*.pkl models/*.faiss embeddings/*.npz cache/*.pkl
	rm -f reports/*.json test_results/*.json
