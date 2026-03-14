.PHONY: install build test bench clean

install:
	uv sync --all-extras
	uv run maturin develop --release

build:
	uv run maturin build --release

test:
	uv run pytest tests/ -v

bench:
	uv run pytest tests/ -v --benchmark-only

clean:
	cargo clean
	rm -rf target/ dist/ *.egg-info fastnn/*.so fastnn/_core*.so
