test:
	pytest -vv .

coveralls:
	py.test . -vv --cov MJOLNIR;

coverage:
	coverage run -m pytest -vv .

