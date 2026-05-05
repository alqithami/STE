.PHONY: install smoke mac final server summarize package test clean

install:
	python -m pip install -r requirements.txt

smoke:
	bash scripts/run_smoke.sh

mac:
	bash scripts/run_synthetic_mac_m4.sh

final:
	bash scripts/run_final_neurips.sh

server:
	bash scripts/run_synthetic_ibm_server.sh

summarize:
	python -m ste_neurips.neurips_suite summarize --out outputs/neurips_final

test:
	python -m pytest -q

package:
	bash scripts/package_reviewer_artifact.sh

clean:
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	find outputs -mindepth 1 -maxdepth 1 -type d ! -name neurips_smoke -exec rm -rf {} +
