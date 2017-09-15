
all: clean install

clean:
	find . -name '*.pyc' -delete

install:
	sudo pip3 install -e .
