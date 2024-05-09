import numpy as np
from scipy.sparse import eye
import matplotlib
from matplotlib import pyplot as plt
import pandas as pd
from pythreejs import *

from IPython.display import display, clear_output
from ipywidgets import *
from traitlets import link, dlink

from magpie.magpie.magpie import magpie
materials = pd.read_csv('magpie/magpie/data/material_properties.csv')

class MagpieInterface():
    
    def __init__(self):
        
        self.Q  = None
        self.Om = None
        self.N = None
        self.biharm = None
        self.mode_shapes = None
        self.interface  = None

        Lx = 2.5; Ly = 0.5; Lz = 5e-3
        self.ldim = [Lx, Ly, Lz]  # -- plate dimensions [x, y, z] in metres
        self.E = 9.0e+9           # -- Young's mod [Pa]
        self.rho = 8765           # -- density [kg/m^3]
        self.nu = 0.3             # -- poisson's ratio
        self.Nmodes = 16          # -- number of modes to compute
        self.resolution = 0.01
        self.h = np.sqrt(self.ldim[0] * self.ldim[1]) * self.resolution   # -- grid spacing
        self.BCs = np.ones((4, 2)) * 1e15 # -- elastic constants around the edges
        self.m = 0; 

        self.auto_plot = False
        self.auto_generate = True

        self.chladni_plot = Output();
        self.generate_modes()
        
        self.mode_label = Label()
        self.__set_label_string__()
        self.__make_interface__()
        self.w0, self.w1, self.w2 = [None]*3

    
    def set_mode_number(self,b):
        m = self.m
        if b.icon == 'arrow-left':
            if m > 0:            
                m -= 1
        elif b.icon == 'arrow-right':
            if m < (self.Nmodes-1):
                m += 1
                
        if self.m != m:
            self.m = m
            if self.auto_plot:
                self.__set_label_string__()
                self.plot_mode_shapes()


    def set_BCs(self, K0y, Kx0, KLy, KxL, R0y, Rx0, RLy, RxL):        
        self.BCs = np.array([[Kx0,Rx0],[K0y,R0y],[KxL,RxL],[KLy,RLy]], dtype=np.float64)        
        
        if self.auto_generate:
            self.generate_modes()

    def set_material(self, E, rho):
        self.E = E     # -- Young's mod [Pa]
        self.rho = rho # -- density [kg/m^3]
        self.nu = 0.1  # -- poisson's ratio
        self.generate_modes()

    
    def set_resolution(self, res):
        self.resolution = np.interp(res,[0,1],[0.05,0.005])
        self.h = np.sqrt(self.ldim[0] * self.ldim[1]) * self.resolution   # -- grid spacing
        
        if self.auto_generate:
            self.generate_modes()


    def set_dimensions(self, Lx, Ly, Lz):        
        self.ldim = [Lx, Ly, Lz]
        
        if self.auto_generate:
            self.generate_modes()
    
    def __set_label_string__(self):
        self.mode_label.value = f"Mode #{self.m+1}: {self.Om[self.m]:.2f} Hz"
    
    def generate_modes(self):

        self.Om, self.Q, self.N, self.biharm = magpie(self.rho, self.E, self.nu, self.ldim, self.h, self.BCs, self.Nmodes)

        for m in range(self.Nmodes):
            self.mode_shapes = [np.real(np.reshape(self.Q[:, m], [self.N['x'], self.N['y']])) for m in range(self.Nmodes)]
        
        if self.auto_plot:
            self.plot_mode_shapes()
    
    def plot_mode_shapes(self):
        """
        """
        m = self.m

        with self.chladni_plot:
            clear_output(wait=True)

            fig = plt.figure(figsize=(4,4))
            ax = fig.add_subplot(111)

            Z = abs(self.mode_shapes[m])
            chladni = plt.pcolormesh(Z.T, cmap='copper_r', shading='gouraud')
            ax.set_axis_off()
            cmax = np.max(Z) * 0.35
            chladni.set_clim(0.00, cmax)


            plt.plot()        
            plt.show()
            
    def __make_interface__(self):
        form_item_layout = Layout(
        display='flex',
        flex_flow='row',
        justify_content='center'
        )

        resolution_slider = FloatSlider(value=0.00, min=0.000, max=1.0, step=0.001,
                                        readout=False, readout_format='.2f',
                                        continuous_update=False, description='')

        dimension_sliders = {
            'Lx':BoundedFloatText(value=1.0, min=0.2, max=2.0, description='X [m]'), 
            'Ly':BoundedFloatText(value=0.8, min=0.2, max=2.0, description='Y [m]'), 
            'Lz':BoundedFloatText(value=0.005, min=0.001, max=0.01, description='thickness [m]')
        }

        material_dropdown = Dropdown(options=[(y,x) for x,y in enumerate(materials['material'])],
                                     value=None,
                                     layout=form_item_layout)

        material_coefs = [BoundedFloatText(min=0.0, max=500, description='Young\'s',disabled=True,),
                 BoundedFloatText(min=0.2, max=30.0, description='Density',disabled=True,),
                 BoundedFloatText(value=0.3, min=0.0, max=0.5, description='Poisson\'s',disabled=True,)]


        form_items = [
            Box([Label(value='Dimensions'), *dimension_sliders.values()], layout=form_item_layout),
            Box([Label(value='Material'), material_dropdown],layout=form_item_layout),
            Box([Label(value=''), *material_coefs], layout=form_item_layout),
            Box([Label(value='Accuracy'), resolution_slider], layout=form_item_layout)
        ]

        form = Box(form_items, layout=Layout(
            display='flex',
            flex_flow='column',
            border='solid 2px',
            align_items='stretch',
            width='auto',
        ))

        labels = ['K0y', 'Kx0', 'KLy', 'KxL', 'R0y', 'Rx0', 'RLy', 'RxL']

        bc_sliders =  {label:FloatLogSlider(
            value=1e15, base=10, min=0, max=15,
            description=label,
            continuous_update=True,
            orientation='vertical' if 'x' in label else 'horizontal',
            readout_format='.1e',
            layout=Layout(flex='1 1 1', width='auto')
        ) for label in labels}


        button_labels = ['arrow-left','arrow-right']
        buttons ={label:widgets.Button(icon=label) for label in button_labels}
        [b.on_click(self.set_mode_number) for _,b in buttons.items()]

        interactive_output(self.set_dimensions, dimension_sliders)
        interactive_output(self.set_BCs, bc_sliders)
        interactive(self.set_resolution, res=resolution_slider)
        interactive(self.set_material_labels,m=material_dropdown)

        self.auto_plot = True
        self.plot_mode_shapes()
        
        self.interface = VBox([form,
                  VBox([bc_sliders['KLy'],bc_sliders['RLy']]),
                  HBox([bc_sliders['Kx0'],bc_sliders['Rx0'],self.chladni_plot,bc_sliders['RxL'],bc_sliders['KxL']]),
                  VBox([bc_sliders['R0y'],bc_sliders['K0y']]),
                  HBox([buttons['arrow-left'],self.mode_label,buttons['arrow-right']])])

    def show(self):
        display(self.interface)

    def set_material_labels(self, m):
        E, rho = [materials.loc[m]['youngs'], materials.loc[m]['densities']]

        material_coefs[0].value = E
        material_coefs[1].value = rho

        self.set_material(E*1e9, rho*1e3)


            
magpie_plot = MagpieInterface() 