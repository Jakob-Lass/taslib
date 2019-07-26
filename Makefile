test:
	pytest -vv .

coveralls:
	py.test . -vv --cov .;

coverage:
	coverage run -m pytest -vv .

