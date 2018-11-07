install:
	pip2 install -r requirements.txt
	python2 setup.py install
	pip3 install -r requirements.txt
	python3 setup.py install

clean:
	rm -rf build
	rm Meshes/*.pyc