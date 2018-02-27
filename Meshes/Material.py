#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os

# Focusing on Obj and glTF specs
# obj https://en.wikipedia.org/wiki/Wavefront_.obj_file
# glTF https://github.com/AnalyticalGraphicsInc/obj2gltf
class Material(object):
    def __init__(self, name):
        self.name = name
        self.illumination_model = 2     # illum
        self.ambient = [0.2, 0.2, 0.2]  # Ka
        self.ambient_map = ''           # map_Ka
        self.diffuse = [1.0, 1.0, 1.0]  # Kd
        self.diffuse_map = ''           # map_Kd
        self.specular = [0.0, 0.0, 0.0] # Ks
        self.specular_map = ''          # map_Ks
        self.specular_exp = 10.0        # Ns
        self.specular_exp_map = ''      # map_Ns
        self.emissive = [0.0, 0.0, 0.0] # Ke
        self.emissive_map = ''          # map_Ke
        self.opacity = 1.0              # d
        self.opacity_map = ''           # map_d
        self.bump_map = ''              # map_bump
        self.optical_density = 1.0      # Ni
        # TODOs:
        #      
        # self.displacement_map = ''    # disp
        # self.reflection_map = ''      # refl
        
    def toMtl(self, file_name = None):
        lines = '# MTL by Patricio Gonzalez Vivo\n'

        lines += 'newmtl ' + self.name + '\n'
        lines += 'Ke %f %f %f\n' % (self.emissive[0], self.emissive[1], self.emissive[2]) 
        lines += 'Ka %f %f %f\n' % (self.ambient[0], self.ambient[1], self.ambient[2]) 
        lines += 'Kd %f %f %f\n' % (self.diffuse[0], self.diffuse[1], self.diffuse[2]) 
        lines += 'Ks %f %f %f\n' % (self.specular[0], self.specular[1], self.specular[2]) 
        lines += 'Ns %f\n' % (self.specular_exp) 
        lines += 'Ni %f\n' % (self.optical_density) 
        lines += 'd %f\n' % (self.opacity) 
        lines += 'illum ' + str(int(self.illumination_model)) + '\n'

        if len(self.emissive_map) > 0:
            lines += 'map_Ke ' + self.emissive_map + '\n'

        if len(self.ambient_map) > 0:
            lines += 'map_Ka ' + self.ambient_map + '\n'

        if len(self.diffuse_map) > 0:
            lines += 'map_Kd ' + self.diffuse_map + '\n'

        if len(self.specular_map) > 0:
            lines += 'map_Ks ' + self.specular_map + '\n'

        if len(self.specular_exp_map) > 0:
            lines += 'map_Ns ' + self.specular_exp_map + '\n'

        if len(self.opacity_map) > 0:
            lines += 'map_d ' + self.opacity_map + '\n'

        if len(self.bump_map) > 0:
            lines += 'map_bump ' + self.bump_map + '\n'

        if file_name:
            file = open(file_name, 'w')
            file.write( lines )
            file.close()
        else:
            return lines