run:
	clear && ./env/bin/python run_ml_pipeline.py --do_train --test

test:
	clear && ./env/bin/python run_ml_pipeline.py --config configs/test.yaml --do_test

env:
	python3 -m venv env
	./env/bin/pip install --upgrade pip
	./env/bin/pip install -r requirements.txt

clean:
	rm -rf checkpoint logging 

clean-all:
	rm -rf checkpoint env logging 
