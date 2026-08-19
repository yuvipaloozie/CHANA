.PHONY: install test validate

install:
	python -m pip install -e ".[tensorflow,test]"

test:
	python -m pytest

validate:
	python scripts/validate_manifests.py
	python scripts/validate_source_data.py
