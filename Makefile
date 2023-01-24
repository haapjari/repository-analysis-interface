run:
	python3 main.py

update:
	pip3 freeze > requirements.txt

test:
	python3 -m unittest discover -s tests

upgrade:
	pip3 install -r requirements.txt --upgrade

install:
	pip3 install -r requirements.txt

env:
	python3 -m venv venv
	
activate:
	source ./venv/bin/activate

deactivate:
	deactivate