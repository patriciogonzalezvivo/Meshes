all:
	pip install -r requirements.txt
	python setup.py install

clean:
	rm -rf build
	rm Meshes/*.pyc