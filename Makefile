all:
	pip2 install -r requirements.txt
	python2 setup.py install

clean:
	rm -rf build
	rm Meshes/*.pyc