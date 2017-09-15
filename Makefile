
all: clean install

clean:
	find . -name '*.pyc' -delete

install:
	pip install -e .
