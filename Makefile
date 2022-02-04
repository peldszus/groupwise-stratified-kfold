VIRTUALENV_DIR=./env
PIP=${VIRTUALENV_DIR}/bin/pip
POETRY=${VIRTUALENV_DIR}/bin/poetry

all: build test

.PHONY: virtualenv
virtualenv:
	if [ ! -e ${PIP} ]; then python3.8 -m venv ${VIRTUALENV_DIR}; fi
	${PIP} install pip==22.0.3
	${PIP} install poetry==1.1.12

.PHONY: build
build: virtualenv
	${POETRY} install --no-interaction
	${POETRY} run pre-commit install

.PHONY: test
test:
	${POETRY} run pytest

.PHONY: lint
lint:
	${POETRY} run pre-commit run --all-files

.PHONY: clean
clean:
	rm -rf ${VIRTUALENV_DIR}
