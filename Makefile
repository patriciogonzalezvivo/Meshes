install:
	pip install -r requirements.txt
	python setup.py install

install3:
	pip3 install -r requirements.txt
	python3 setup.py install

install_blender_osx:
	rm /Applications/Blender/blender.app/Contents/Resources/2.80/python/lib/python3.7/site-packages/Meshes*
	#/Applications/Blender/blender.app/Contents/Resources/2.80/python/bin/./python3.7m -m pip install -r requirements.txt --user
	/Applications/Blender/blender.app/Contents/Resources/2.80/python/bin/./python3.7m -m setup.py install 

clean:
	rm -rf dist
	rm -rf build
	rm Meshes/*.pyc