install:
	pip install -r requirements.txt
	python setup.py install

install3:
	pip3 install -r requirements.txt
	python3 setup.py install

clean:
	rm -rf build
	rm Meshes/*.pyc