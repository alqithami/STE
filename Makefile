.PHONY: install smoke mac final server real-suite summarize package test clean clean-generated

install:
	python -m pip install -r requirements.txt

test:
	python -m pytest -q

smoke:
	bash scripts/run_smoke.sh

mac:
	bash scripts/run_synthetic_mac_m4.sh

final:
	bash scripts/run_final_neurips.sh

server:
	bash scripts/run_synthetic_ibm_server.sh

real-suite:
	python -m ste_neurips.neurips_suite real-suite \
		--manifest configs/real_datasets_manifest_template.yaml \
		--out outputs/real_suite_final

summarize:
	python -m ste_neurips.neurips_suite summarize --out outputs/neurips_final

package:
	bash scripts/package_reviewer_artifact.sh

clean:
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	rm -rf .pytest_cache .mypy_cache .ruff_cache

clean-generated: clean
	rm -rf outputs/neurips_smoke outputs/neurips_final outputs/real_suite_final outputs/reviewer_artifact
	rm -rf outputs/runs outputs/audit outputs/tmp
