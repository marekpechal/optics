import tkinter as tk
import tkinter.ttk as ttk
import tkinter.simpledialog
import tkinter.filedialog
import pickle
import os
import time
import numpy as np

def classMapping(cls,dct):
    res = []
    for key in dct:
        if cls is key or issubclass(cls,key):
            res += dct[key]
    return res

def getTrackedAttributes(obj,trackingDct):
    res = []
    if hasattr(obj,'__tracked__'):
        res += obj.__tracked__
    res += classMapping(obj.__class__,trackingDct)
    return res

def getExposedMethods(obj,exposedMethodDct):
    res = []
    if hasattr(obj,'__exposedMethods__'):
        res += obj.__exposedMethods__
    if hasattr(obj.__class__,'__exposedMethods__'):
        res += obj.__class__.__exposedMethods__
    res += classMapping(obj.__class__,exposedMethodDct)
    return res

def getTreeChildren(tree,idx):
    for cidx in tree.get_children(idx):
        yield from getTreeChildren(tree,cidx)
    yield idx

def getTrackedChildren(obj,trackingDct):
    res = []

    # tracked attributes
    attrList = getTrackedAttributes(obj,trackingDct)
    # if (hasattr(obj,'__tracked__')
    #         or classMapping(obj.__class__,trackingDct)):
    #     if hasattr(obj,'__tracked__'):
    #         attrList = obj.__tracked__
    #     else:
    #         attrList = classMapping(obj.__class__,trackingDct)

    def makeAttrSetter(obj,attr):
        def setter(x):
            #setattr(obj,attr,x)
            obj.__setattr__(attr,x)
            if hasattr(obj,'__onChange__') and obj.__onChange__ is not None:
                obj.__onChange__()
        return setter

    def makeItemSetter(obj,idx):
        def setter(x):
            #setitem(obj,idx,x)
            obj.__setitem__(idx,x)
            if hasattr(obj,'__onChange__') and obj.__onChange__ is not None:
                obj.__onChange__()
        return setter

    if attrList:
        for attr in attrList:
            if hasattr(obj,attr):
                child = getattr(obj,attr)
                res.append({
                    'parent': obj,
                    'name': attr,
                    'obj': child,
                    #'setter': lambda x,obj=obj,attr=attr: setattr(obj,attr,x),
                    'setter': makeAttrSetter(obj,attr),
                    'deleter': lambda obj=obj,attr=attr: delattr(obj,attr),
                    'children': getTrackedChildren(child,trackingDct)
                })

    # list elements
    if isinstance(obj,list) or isinstance(obj,np.ndarray):
        for idx,child in enumerate(obj):
            name = (child.__name__ if hasattr(child,'__name__')
                else '['+str(idx)+']')
            res.append({
                'parent': obj,
                'name': name,
                'obj': child,
                'setter': makeItemSetter(obj,idx),
                #'setter': lambda x,obj=obj,idx=idx: obj.__setitem__(idx,x),
                'deleter': lambda obj=obj,idx=idx: obj.__delitem__(idx),
                'children': getTrackedChildren(child,trackingDct)
            })

    # dictionary elements
    if isinstance(obj,dict):
        for name,child in obj.items():
            res.append({
                'parent': obj,
                'name': name,
                'obj': child,
                'setter': makeItemSetter(obj,name),
                #'setter': lambda x,obj=obj,name=name: obj.__setitem__(name,x),
                'deleter': lambda obj=obj,name=name: obj.__delitem__(name),
                'children': getTrackedChildren(child,trackingDct)
            })

    return res

def getTrackedChildrenGenerator(lst):
    for dct in lst:
        res = dct.copy()
        del res['children']
        yield res
        yield from getTrackedChildrenGenerator(dct['children'])


class TrackingView(tk.Frame):

    def __init__(self,master,*args,**kwargs):
        if not 'rootObjects' in kwargs:
            self.rootObjects = []
        else:
            self.rootObjects = kwargs['rootObjects']
            del kwargs['rootObjects']

        self.allObjects = {}
        self.trackingDct = {}
        self.exposedMethodDct = {}

        tk.Frame.__init__(self,master,*args,**kwargs)

        self.editable = {
            int: tk.simpledialog.askinteger,
            float: tk.simpledialog.askfloat,
            np.float32: tk.simpledialog.askfloat,
            np.float64: tk.simpledialog.askfloat,
            str: tk.simpledialog.askstring
            }

        self.tree = ttk.Treeview(self,
            columns=('1','2'))

        self.tree.column('#0',width=150,minwidth=150)
        self.tree.column('1',width=150,minwidth=150)
        self.tree.column('2',width=150,minwidth=150)

        self.tree.heading('#0',text='Objects',anchor=tk.W)
        self.tree.heading('1',text='Class name',anchor=tk.W)
        self.tree.heading('2',text='Value',anchor=tk.W)

        self.tree.bind('<Double-Button-1>',self.onTreeDoubleClick)
        self.tree.bind('<Button-3>',self.onTreeRightClick)
        self.tree.bind('<Shift-MouseWheel>',self.onMouseWheel)
        self.tree.bind('<<TreeviewSelect>>',self.onSelChange)

        self.tree.pack(side=tk.TOP,fill=tk.X)

        self.buttonFrame = tk.Frame(self)
        self.buttonFrame.pack(side=tk.TOP,fill=tk.X)

        self.saveButton = tk.Button(self.buttonFrame,text='Save',
            command=self.onSave)
        self.saveButton.pack(side=tk.LEFT)

        self.loadButton = tk.Button(self.buttonFrame,text='Load',
            command=self.onLoad)
        self.loadButton.pack(side=tk.LEFT)

        self.update()

    def _parseEntry(self,objDesc):
        obj = objDesc['obj']
        objtype = str(obj.__class__.__name__)
        objval = str(type(obj) in self.editable)
        objval = (
            str(obj) if (type(obj) in self.editable)
            else '')
        return {'text':objDesc['name'],'values':(objtype,objval)}


    def _insertObj(self,parentEntry,structure):
        resdct = {}
        for objDesc in structure:

            entry = self.tree.insert(parentEntry,'end',
                **self._parseEntry(objDesc))

            resdct[entry] = objDesc.copy()
            del resdct[entry]['children']

            resdct.update(self._insertObj(entry,objDesc['children']))

        return resdct

    def update(self):
        openStatus = {}
        for idx in self.allObjects:
            objId = id(self.allObjects[idx]['obj'])
            openStatus[objId] = self.tree.item(idx)['open']
        structure = getTrackedChildren(self.rootObjects,self.trackingDct)
        self.tree.delete(*self.tree.get_children())
        self.allObjects = self._insertObj('',structure)
        for idx in self.allObjects:
            objId = id(self.allObjects[idx]['obj'])
            if objId in openStatus:
                dct = self.tree.item(idx)
                dct['open'] = openStatus[objId]
                self.tree.item(idx,**dct)

    def updateSoft(self):
        structure = getTrackedChildren(self.rootObjects,self.trackingDct)
        ekeys = list(self.allObjects.keys())
        for k,objDesc in zip(ekeys,getTrackedChildrenGenerator(structure)):
            self.tree.item(k,**self._parseEntry(objDesc))
            self.allObjects[k] = objDesc

    def onTreeDoubleClick(self,event):
        idx = self.tree.identify('item',event.x,event.y)

        if idx in self.allObjects:
            objDesc = self.allObjects[idx]
            obj = objDesc['obj']
            if type(obj) in self.editable:
                dialog = self.editable[type(obj)]
                resp = dialog('Input','Input new value:',parent=self,
                    initialvalue=obj)

                # edit object's value
                objDesc['setter'](resp)
                if hasattr(self,'__onChange__'):
                    self.__onChange__()

                self.update()

    def onSelChange(self,*args):
        if hasattr(self,'__onChange__'):
            self.__onChange__()

    def onMouseWheel(self,event):
        idx = self.tree.identify('item',event.x,event.y)
        if idx in self.allObjects:
            objDesc = self.allObjects[idx]
            obj = objDesc['obj']
            if type(obj) in [float,np.float32,np.float64]:
                val = obj+0.1*(event.delta//120)
                # edit object's value
                objDesc['setter'](val)
                if hasattr(self,'__onChange__'):
                    self.__onChange__()

                self.update()

            if type(obj) in [int]:
                val = obj+(event.delta//120)
                # edit object's value
                objDesc['setter'](val)
                if hasattr(self,'__onChange__'):
                    self.__onChange__()

                self.update()

    def onTreeRightClick(self,event):
        idx = self.tree.identify('item',event.x,event.y)
        if idx in self.allObjects:
            objDesc = self.allObjects[idx]
            obj = objDesc['obj']
            name = objDesc['name']

            self.aMenu = tk.Menu(self,tearoff=0)
            mList = getExposedMethods(obj,self.exposedMethodDct)
            if mList:
                for mName in mList:
                    def cmd(mName=mName):
                        getattr(obj,mName)()
                        if hasattr(self,'__onChange__'):
                            self.__onChange__()
                        self.update()
                    self.aMenu.add_command(label=mName,command=cmd)
            self.aMenu.add_separator()
            self.aMenu.add_command(label='delete '+name,
                command=lambda:self.onPopupDeleteClick(idx))
            self.aMenu.post(event.x_root,event.y_root)

    def onPopupDeleteClick(self,idx):
        self.allObjects[idx]['deleter']()
        for cidx in getTreeChildren(self.tree,idx):
            self.tree.delete(cidx)
            del self.allObjects[cidx]
        if hasattr(self,'__onChange__'): self.__onChange__()
        self.updateSoft()

    def onSave(self):
        if hasattr(self,'__onSave__'):
            self.__onSave__()
            return
        f = tkinter.filedialog.asksaveasfile(
            mode='wb',initialdir=os.getcwd(),defaultextension='pickle',
            filetypes=[('pickle files','*.pickle')])
        if f is not None:
            pickle.dump(self.rootObjects,f)
            f.close()

    def onLoad(self):
        if hasattr(self,'__onLoad__'):
            self.__onLoad__()
            return
        f = tkinter.filedialog.askopenfile(
            mode='rb',initialdir=os.getcwd(),
            filetypes=[('pickle files','*.pickle')])
        if f is not None:
            self.rootObjects = pickle.load(f)
            f.close()
            if hasattr(self,'__onChange__'):
                self.__onChange__()
            self.update()
