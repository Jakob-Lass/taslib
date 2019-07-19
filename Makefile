test:
	coverage run -m pytest -vv .
	coverage report
	coverage html
