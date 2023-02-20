.PHONY: build
build:
	@get-models
	@sh build/build.sh

.PHONY: export
export:
	@sh build/export.sh

.PHONY: test
test:
	@sh build/test.sh

.PHONY: get-models
get-models:
	@sh models/fetch.sh
