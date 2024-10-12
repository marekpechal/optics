import matplotlib.pyplot as plt
import tkinter as tk
from PIL import Image,ImageDraw,ImageTk,ImageFont
import numpy as np
from math import floor,ceil

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
from matplotlib.backend_bases import key_press_handler
from matplotlib.figure import Figure

class ZoomableFigure(tk.Frame):
    def __init__(self,master,**kwargs):
        if not 'facecolor' in kwargs:
            self.facecolor = 'white'
        else:
            self.facecolor = kwargs['facecolor']
            del kwargs['facecolor']

        if not 'size' in kwargs:
            self.size = (640,480)
        else:
            self.size = kwargs['size']
            del kwargs['size']

        if not 'supersampling' in kwargs:
            self.supersampling = 4
        else:
            self.supersampling = kwargs['supersampling']
            del kwargs['supersampling']

        tk.Frame.__init__(self,master,**kwargs)

        self.canvas = tk.Canvas(self,width=self.size[0],height=self.size[1])
        self.canvas.pack(expand=tk.YES,fill=tk.BOTH)

        self.scale = 100.0
        self.origin = np.zeros(2)

        self.clearImage()
        self.permanentGeometry = []
        self.geometry = []
        self.display()

        self.canvas.bind('<MouseWheel>',self.onMouseWheel)

        self.middlePt = None
        self.canvas.bind('<Button-2>',self.onMiddlePress)
        self.canvas.bind('<ButtonRelease-2>',self.onMiddleRelease)
        self.canvas.bind('<Leave>',self.onMiddleRelease)
        self.canvas.bind('<B2-Motion>',self.onMiddleMove)

    def clearImage(self):
        self.image = Image.new('RGB',
            (self.size[0]*self.supersampling,self.size[1]*self.supersampling),
            self.facecolor)
        self.draw = ImageDraw.Draw(self.image,'RGBA')

    def _toDrawCoord(self,pts):
        s = np.array([self.scale,-self.scale])
        pts = np.array(self.size)/2 + s*(pts-self.origin)
        pts = (pts*self.supersampling)
        return list(map(tuple,pts))

    def _fromCanvasCoord(self,pts):
        s = np.array([self.scale,-self.scale])
        pts = np.array(pts)
        pts = pts - np.array(self.size)/2
        pts = pts/s + self.origin
        return pts

    def redraw(self):
        self.clearImage()
        geom = sorted(self.geometry+self.permanentGeometry,
            key = lambda g: g['z'] if 'z' in g else 0.0)
        for g in geom:
            if g['type'] == 'grid':
                pt1,pt2 = self._fromCanvasCoord(
                    [(0,self.size[1]),(self.size[0],0)])
                i1 = int(ceil(pt1[0]/g['size']))
                i2 = int(floor(pt2[0]/g['size']))
                j1 = int(ceil(pt1[1]/g['size']))
                j2 = int(floor(pt2[1]/g['size']))
                for i in range(i1,i2+1):
                    pts = [[g['size']*i,pt1[1]],[g['size']*i,pt2[1]]]
                    self.draw.line(self._toDrawCoord(pts),fill=g['color'],
                        width=int(g['width']*self.supersampling))
                for j in range(j1,j2+1):
                    pts = [[pt1[0],g['size']*j],[pt2[0],g['size']*j]]
                    self.draw.line(self._toDrawCoord(pts),fill=g['color'],
                        width=int(g['width']*self.supersampling))
            if g['type'] == 'axes':
                pt1,pt2 = self._fromCanvasCoord(
                    [(0,self.size[1]),(self.size[0],0)])
                pts = [[0,pt1[1]],[0,pt2[1]]]
                self.draw.line(self._toDrawCoord(pts),fill=g['color'],
                    width=int(g['width']*self.supersampling))
                pts = [[pt1[0],0],[pt2[0],0]]
                self.draw.line(self._toDrawCoord(pts),fill=g['color'],
                    width=int(g['width']*self.supersampling))
            if g['type'] == 'line':
                self.draw.line(self._toDrawCoord(g['points']),fill=g['color'],
                    width=int(g['width']*self.supersampling))
            if g['type'] == 'polygon':
                self.draw.polygon(self._toDrawCoord(g['points']),
                    fill=g['color'])
            if g['type'] == 'rectangle':
                self.draw.polygon(self._toDrawCoord(g['center']+np.array([
                        [-g['size'][0]/2,-g['size'][1]/2],
                        [+g['size'][0]/2,-g['size'][1]/2],
                        [+g['size'][0]/2,+g['size'][1]/2],
                        [-g['size'][0]/2,+g['size'][1]/2]
                    ])),
                    fill=g['color'])
            if g['type'] == 'circle':
                self.draw.polygon(self._toDrawCoord(
                    g['center']+g['radius']*np.array([[np.cos(u),np.sin(u)] 
                        for u in np.linspace(0,2*np.pi,81)[:-1]])),
                    fill=g['color'])
            if g['type'] == 'text':
                pts = np.array([g['position']])
                p0 = self._toDrawCoord(pts)[0]
                fsize = int(round(self.supersampling*self.scale*g['size']))
                fnt = ImageFont.truetype('bahnschrift.ttf',fsize)
                w,h = self.draw.textsize(g['text'],font=fnt)
                self.draw.text((p0[0]-w/2,p0[1]-h/2),g['text'],
                    font=fnt,fill=g['color'])
            if g['type'] == 'dot':
                r = g['radius']/self.scale
                pts = np.array(g['center'])+r*np.array([
                    [np.cos(u),np.sin(u)]
                    for u in np.linspace(0,2*np.pi,81)[:-1]])
                self.draw.polygon(self._toDrawCoord(pts),
                    fill=g['color'])
        self.display()


    def display(self):
        self.canvas._photoim = ImageTk.PhotoImage(
            self.image.resize(self.size,Image.BICUBIC))
        self.canvas.create_image(0,0,
            image=self.canvas._photoim,anchor=tk.NW)

    def clear(self):
       self.geometry = []
       self.redraw()

    def onMouseWheel(self,event):
        pt = self._fromCanvasCoord([event.x,event.y])
        fac = np.exp(0.001*event.delta)
        self.origin = pt - (pt - self.origin)/fac
        self.scale *= fac
        self.redraw()

    def onMiddlePress(self,event):
        self.middlePt=[[event.x,event.y],self.origin]

    def onMiddleRelease(self,event):
        self.middlePt=None

    def onMiddleMove(self,event):
        if self.middlePt is not None:
            dx = event.x-self.middlePt[0][0]
            dy = event.y-self.middlePt[0][1]
            s = np.array([self.scale,-self.scale])
            self.origin = self.middlePt[1] - np.array([dx,dy])/s
            self.redraw()

    def pack(self,*args,**kwargs):
        tk.Frame.pack(self,*args,**kwargs)
        self.redraw()
