import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
from matplotlib.backend_bases import key_press_handler
from matplotlib.figure import Figure

from optics.raytracing import *
from optics.raytracing.lenses import *
from optics.raytracing.optical_surfaces import *

import os,sys

sys.path.append(os.path.join(".", "gui_widgets"))
from ZoomableFigure import ZoomableFigure
from TrackingView import TrackingView

class OpticalSurfaceCollection_mod(OpticalSurfaceCollection):
    def addPlane(self,*args,**kwargs):
        obj = Plane(*args,**kwargs)
        self.elements.append(obj)
        return obj

    def addSphere(self,*args,**kwargs):
        obj = Sphere(*args,**kwargs)
        self.elements.append(obj)
        return obj

    def addSymmetricLens(self,*args,**kwargs):
        obj = SymmetricLens(*args,**kwargs)
        self.elements.append(obj)
        return obj

    def addSemiPlanarLens(self,*args,**kwargs):
        obj = SemiPlanarLens(*args,**kwargs)
        self.elements.append(obj)
        return obj


class App(object):
    def __init__(self,window):
        self.window = window
        self.editorFrame = tk.Frame(window)
        self.editorFrame.pack(side=tk.TOP)

        opticalSystem = OpticalSurfaceCollection_mod('optical system',[])
        opticalSystem.__exposedMethods__ = ['addPlane', 'addSphere', 'addSymmetricLens', 'addSemiPlanarLens']

        lightSource = LightSource('light source collection',[])
        lightSource.__exposedMethods__ = ['addRay']


        self.opticalSystemEditor = TrackingView(self.editorFrame,
            rootObjects=[
                opticalSystem
            ])

        self.opticalSystemEditor.trackingDct = {
            OpticalSurfaceCollection_mod:['elements','surfaceRayInteraction'],
            Plane:['origin','normalVec'],
            Sphere:['origin','radius'],
            SphericalCap:['origin','invRadius','r','direction'],
            ConicalSlice:['origin','r1','r2','h','direction'],
            SymmetricLens:['origin','radius','direction','invR','centerThickness'],
            SemiPlanarLens:['origin','radius','direction','invR','centerThickness'],
            OpticalSurface:['surfaceRayInteraction'],
            SurfaceRayInteraction:['kwargs']
            }
        self.opticalSystemEditor.exposedMethodDct = {
            OpticalSurface:['makeReflective','makeRefractive','makeAbsorptive','makeGrating','makeReflGrating'],
            OpticalSurfaceCollection_mod:['makeReflective','makeRefractive','makeAbsorptive','makeGrating','makeReflGrating']
            }
        self.opticalSystemEditor.__onChange__ = self.onChange
        self.opticalSystemEditor.pack(side=tk.LEFT)


        self.sourcesEditor = TrackingView(self.editorFrame,
            rootObjects=[
                lightSource
            ])
        self.sourcesEditor.trackingDct = {
            Ray:['origin','direction','parameters'],
            LightSource:['elements','origin','numRays','offset','angle','direction','beamWidth','parameters']
            }
        self.sourcesEditor.exposedMethodDct = {
            LightSource:['addPointSource','addDirSource']
            }
        self.sourcesEditor.__onChange__ = self.onChange
        self.sourcesEditor.pack(side=tk.LEFT)


        self.zoomableFigure = ZoomableFigure(window)
        self.zoomableFigure.scale = 10.0
        self.zoomableFigure.permanentGeometry = [
            {'type':'grid','size':10.0,'color':(0,0,0,64),'z':1.0,'width':0.5}]
        self.zoomableFigure.pack(side=tk.TOP)

        button = tk.Button(self.editorFrame, text = 'Update', command = self.update)
        button.pack()

    @property
    def lightSource(self):
        return self.sourcesEditor.rootObjects[0]

    @property
    def opticalSystem(self):
        return self.opticalSystemEditor.rootObjects[0]

    def onChange(self):
        pass

    def update(self, *args):
        self.lightSource.rayTrace(self.opticalSystem)
        self.zoomableFigure.geometry = self.opticalSystem.geometry()
        self.zoomableFigure.geometry += self.lightSource.geometry()
        self.zoomableFigure.redraw()

window = tk.Tk()

app = App(window)

window.mainloop()
