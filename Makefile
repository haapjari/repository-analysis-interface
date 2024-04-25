run:
	python3 src/main.py
.PHONY: run

update:
	pip3 freeze > requirements.txt
.PHONY: update

test:
	python3 -m unittest discover -s tests
.PHONY: test

upgrade:
	pip3 install -r requirements.txt --upgrade
.PHONY: upgrade

install:
	pip3 install -r requirements.txt
.PHONY: install

compile:
	pyinstaller --onefile src/main.py

env:
	python3 -m venv venv
.PHONY: env

activate:
	source ./venv/bin/activate
.PHONY: activate

deactivate:
	deactivate
.PHONY: deactivate