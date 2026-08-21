# hnsw — verification without CI.
#
# This repository deliberately ships no GitHub Actions workflow. `make check`
# is the gate: it runs exactly what CI would, and `make hooks` wires it into
# git so a commit cannot land while it fails.
#
# On a host where the native toolchain cannot link (a Windows box with no
# MSVC or MinGW, for instance), prefix any of the native targets with
# `docker-`: `make docker-check`, `make docker-bench`. Those run the identical
# commands inside a Linux container and share a named cargo registry volume so
# the second run is fast.

CARGO       ?= cargo
DOCKER_RUST ?= rust:1-bookworm
CARGO_VOLUME?= hnsw-cargo
WASM_TARGET  := wasm32-unknown-unknown
WASM_OUT     := target/$(WASM_TARGET)/release/hnsw.wasm

# Dataset shape for `make bench`. Override on the command line:
#   make bench BENCH_N=50000 BENCH_DIMS=128
BENCH_N       ?= 20000
BENCH_DIMS    ?= 64
BENCH_CLUSTERS?= 24
BENCH_QUERIES ?= 400
BENCH_K       ?= 10
BENCH_EF      ?= 8,16,32,64,128,256

DOCKER_RUN = docker run --rm \
	-v "$(CURDIR)":/app \
	-v $(CARGO_VOLUME):/usr/local/cargo/registry \
	-w /app $(DOCKER_RUST) bash -c

.PHONY: check fmt lint test bench demo hooks clean help \
        docker-check docker-bench docker-shell

help:
	@echo "hnsw — make targets"
	@echo "  check         fmt + clippy (-D warnings) + tests"
	@echo "  bench         build an index and sweep ef_search against an exact scan"
	@echo "  demo          build the wasm and assemble docs/ for GitHub Pages"
	@echo "  hooks         install .githooks/pre-commit"
	@echo "  clean         remove build artefacts"
	@echo "  docker-check  same as check, inside $(DOCKER_RUST)"
	@echo "  docker-bench  same as bench, inside $(DOCKER_RUST)"

## Every gate, in the order that fails fastest.
check: fmt lint test
	@echo "OK: fmt, clippy and tests all pass"

fmt:
	$(CARGO) fmt --all --check

lint:
	$(CARGO) clippy --all-targets -- -D warnings

test:
	$(CARGO) test

## Real recall and throughput numbers. This is what the README table is.
bench:
	$(CARGO) build --release --bin hnswctl
	./target/release/hnswctl bench \
		--n $(BENCH_N) --dims $(BENCH_DIMS) --clusters $(BENCH_CLUSTERS) \
		--queries $(BENCH_QUERIES) --k $(BENCH_K) --ef $(BENCH_EF)

## Compile the index to WebAssembly and assemble the published page.
##
## GitHub Pages serves docs/ straight from the default branch, so there is
## nothing to deploy and no workflow to run. `--lib` is deliberate: hnswctl is
## a native binary and does not build for wasm.
# The published module is built with the source path remapped, so the same
# source produces the same bytes whether it is built on this host or in the
# container. Without that, panic messages embed the absolute build directory
# and the committed artefact looks stale on every other machine.
demo:
	RUSTFLAGS="--remap-path-prefix=$$(pwd)=." $(CARGO) build --lib --target $(WASM_TARGET) --release
	@mkdir -p docs
	cp $(WASM_OUT) docs/hnsw.wasm
	cp web/index.html docs/index.html
	@echo "OK: docs/ rebuilt ($$(wc -c < docs/hnsw.wasm) bytes of index)"

## Install the pre-commit hook. One command, and it stays installed.
hooks:
	git config core.hooksPath .githooks
	@echo "OK: git will run .githooks/pre-commit before every commit"

clean:
	$(CARGO) clean
	rm -f docs/hnsw.wasm

# -- container escape hatches ------------------------------------------------

docker-check:
	@docker volume inspect $(CARGO_VOLUME) >/dev/null 2>&1 || docker volume create $(CARGO_VOLUME)
	$(DOCKER_RUN) "rustup component add rustfmt clippy >/dev/null 2>&1; make check"

docker-bench:
	@docker volume inspect $(CARGO_VOLUME) >/dev/null 2>&1 || docker volume create $(CARGO_VOLUME)
	$(DOCKER_RUN) "make bench BENCH_N=$(BENCH_N) BENCH_DIMS=$(BENCH_DIMS) \
		BENCH_CLUSTERS=$(BENCH_CLUSTERS) BENCH_QUERIES=$(BENCH_QUERIES) \
		BENCH_K=$(BENCH_K) BENCH_EF=$(BENCH_EF)"

docker-shell:
	docker run --rm -it -v "$(CURDIR)":/app \
		-v $(CARGO_VOLUME):/usr/local/cargo/registry -w /app $(DOCKER_RUST) bash
