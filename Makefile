.PHONY: build
build:
	@make get-models
	@sh build/build.sh

.PHONY: export
export:
	@sh build/export.sh

.PHONY: test
test:
	@make prepare-test
	@sh build/test.sh

.PHONY: build-grand-challenge
build-grand-challenge:
	@make get-models
	@sh build/build.sh grand-challenge

.PHONY: export-grand-challenge
export-grand-challenge:
	@sh build/export.sh grand-challenge

.PHONY: test-grand-challenge
test-grand-challenge:
	@make prepare-test
	@sh build/test.sh grand-challenge

.PHONY: get-models
get-models:
	@sh models/fetch.sh

.PHONY: prepare-test
prepare-test:
	@make get-models
	@sh tests/prepare-test.sh
	@pip install -r tests/requirements.txt
