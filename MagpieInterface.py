import numpy as np
from scipy.sparse import eye
import matplotlib
from matplotlib import pyplot as plt
import pandas as pd
import base64
import hashlib
from typing import Callable


from IPython.display import display, clear_output
from ipywidgets import *
from traitlets import link, dlink
import ipywidgets
from IPython.display import HTML, display
import time

# --------------------------------------------
# Evil Import Hack while we transition to pypi
try:
    from magpie.magpie.magpie import magpie
except ModuleNotFoundError:
    from magpie import magpie
# --------------------------------------------
materials = pd.read_csv('./data/material_properties.csv')


class DownloadButton(Button):
    """Download button with dynamic content

    The content is generated using a callback when the button is clicked.
    """

    def __init__(self, filename: Callable[[], str], contents: Callable[[], str], **kwargs):
        super(DownloadButton, self).__init__(**kwargs)
        self.filename = filename
        self.contents = contents
        self.on_click(self.__on_click)

    def __on_click(self, b):
        contents: bytes = self.contents().encode('utf-8')
        b64 = base64.b64encode(contents)
        payload = b64.decode()
        digest = hashlib.md5(contents).hexdigest()  # bypass browser cache
        id = f'dl_{digest}'
        display(HTML(f"""<html><body><a id="{id}" download="{self.filename()}" href="data:text/csv;base64,{payload}" download></a><script>(function download() {{document.getElementById('{id}').click();}})()</script></body></html>
"""))


class MagpieInterface:

    def __init__(self):

        self.Q = None
        self.Om = None
        self.N = None
        self.biharm = None
        self.mode_shapes = None
        self.interface = None

        Lx = 2.5
        Ly = 0.5
        Lz = 5e-3

        self.ldim = [Lx, Ly, Lz]  # -- plate dimensions [x, y, z] in metres
        self.E = 9.0e+9  # -- Young's mod [Pa]
        self.rho = 8765  # -- density [kg/m^3]
        self.nu = 0.3  # -- poisson's ratio
        self.Nmodes = 16  # -- number of modes to compute
        self.resolution = 0.01
        self.h = np.sqrt(self.ldim[0] * self.ldim[1]) * self.resolution  # -- grid spacing
        self.BCs = np.ones((4, 2)) * 1e15  # -- elastic constants around the edges
        self.m = 0

        self.auto_plot = True
        self.auto_generate = True

        self.chladni_plot = Output()
        self.mode_label = Label()

        self.generate_modes()

        self.__set_label_string__()
        self.__make_interface__()

        self.w0, self.w1, self.w2 = [None] * 3

    def __make_interface__(self):
        """
        Assemble all IPywidgets Interface elements and hook them up to callbacks
        """
        labels = ['K0y', 'Kx0', 'KLy', 'KxL', 'R0y', 'Rx0', 'RLy', 'RxL']

        bc_sliders = {label: FloatLogSlider(
            value=1e15, base=10, min=0, max=15,
            description=f"${label[0]}_{{ {label[1:]} }}$",
            continuous_update=True,
            orientation='vertical' if 'x' in label else 'horizontal',
            readout_format='.1e',
            layout=Layout(flex='1 1 0%', width='auto', height='auto') if 'x' in label else Layout(flex='3 1 0%',
                                                                                                  width='auto'),
        ) for label in labels}

        button_labels = ['arrow-left', 'arrow-right']
        buttons = {label: widgets.Button(icon=label) for label in button_labels}
        [b.on_click(self.set_mode_number) for _, b in buttons.items()]

        padding = Button(description=' ',
                         layout=Layout(flex='5 1 0%', width='auto'),
                         button_style='',
                         disabled=True,
                         )
        padding.style.button_color = 'white'

        self.chladni_plot.layout = Layout(flex='8 1 0%', width='auto', height='auto')

        centre_plot_row = [
            padding,
            bc_sliders['Kx0'], bc_sliders['Rx0'],
            self.chladni_plot,
            bc_sliders['RxL'], bc_sliders['KxL'],
            padding,
        ]

        rows = [
            self.__make_bcy_slider_row__(bc_sliders['KLy']),
            self.__make_bcy_slider_row__(bc_sliders['RLy']),
            centre_plot_row,
            self.__make_bcy_slider_row__(bc_sliders['R0y']),
            self.__make_bcy_slider_row__(bc_sliders['K0y']),
        ]

        box_layout = Layout(display='flex',
                            flex_flow='row',
                            align_items='center',
                            width='100%',
                            margin='0 0')

        form_row_layout = Layout(
            display='flex',
            flex_flow='row',
            flex_wrap='wrap',
            justify_content='flex-start',
            align_items='center',
            align_content='center',
            height='auto',
        )

        dimension_sliders = {
            'Lx': BoundedFloatText(layout=Layout(flex='1 1 5rem'), value=1.0, min=0.2, max=2.0, step=0.1,
                                   description='X [m]', ),
            'Ly': BoundedFloatText(layout=Layout(flex='1 1 5rem'), value=0.8, min=0.2, max=2.0, step=0.1,
                                   description='Y [m]', ),
            'Lz': BoundedFloatText(layout=Layout(flex='1 1 5rem'), value=0.005, min=0.001, max=0.01, step=0.001,
                                   description='Z [m]', )
        }

        padding = Button(description=' ',
                         layout=Layout(flex='1 1 auto', width='auto', height='auto'),
                         button_style='',
                         disabled=True,
                         )
        padding.style.button_color = 'white'

        material_dropdown = Dropdown(options=[(y, x) for x, y in enumerate(materials['material'])],
                                     value=17,
                                     layout=Layout(flex='1 1 10rem'))

        edit_material_check = Checkbox(
            value=False,
            disabled=False,
            indent=False,
            description='Edit',
            layout=Layout(flex='2 1 auto')
        )

        self.material_coefs = [
            BoundedFloatText(layout=Layout(flex='1 1 5rem', width='5em'), min=0.0, max=500, description='',
                             disabled=False),
            BoundedFloatText(layout=Layout(flex='1 1 5rem', width='5em'), min=0.2, max=20000.0, description='',
                             disabled=False),
            BoundedFloatText(layout=Layout(flex='1 1 5rem', width='5em'), value=0.3, min=0.0, max=0.5, description='',
                             disabled=False)
        ]

        resolution_slider = FloatSlider(value=0.01, min=0.000, max=1.0,
                                        step=0.001,
                                        readout=False, readout_format='.2f',
                                        continuous_update=False,
                                        description='')

        items = [
            [Label(value='Dimensions:', layout=Layout(flex='1 1 auto')), *dimension_sliders.values(), padding],
            [Label(value='Material:', layout=Layout(flex='0 1 auto')), material_dropdown, padding],
            [Label(value='Young\'s [GPa]:'), self.material_coefs[0],
             Label(value='Density $\\left[\\frac{kg}{m^{3}}\\right]$:'), self.material_coefs[1],
             Label(value="Poisson's:"), self.material_coefs[2], padding],
            [Label(value='Accuracy:', layout=Layout(flex='0 1 auto', width='auto')), resolution_slider],
        ]

        form = VBox([Box(children=item, layout=form_row_layout) for item in items],
                    layout=Layout(border='solid'))
        boxes = [form]
        boxes += [Box(children=row, layout=box_layout) for row in rows]

        cent_layout = Layout(
            display='flex',
            flex_flow='row',
            justify_content='center',
            align_items='center',
            align_content='center'
        )

        boxes += [Box(children=[
            buttons['arrow-left'],
            self.mode_label,
            buttons['arrow-right']
        ],
            layout=cent_layout)
        ]

        self.download = DownloadButton(filename=lambda: f'{self.get_current_mode_freq_hz():.2f}_Hz_Shape.csv',
                                       contents=lambda: pd.DataFrame(self.get_current_mode_shape()).to_csv(header=True),
                                       description='Download Shape')

        interactive_output(self.set_dimensions, dimension_sliders)
        interactive_output(self.set_BCs, bc_sliders)
        interactive_output(self.set_resolution, {'res': resolution_slider})
        interactive_output(self.set_material_labels, {'m': material_dropdown})
        interactive_output(self.set_youngs_gpa, {'E': self.material_coefs[0]})
        interactive_output(self.set_density, {'rho': self.material_coefs[1]})
        interactive_output(self.set_poisson, {'nu': self.material_coefs[2]})

        self.interface = VBox(boxes)

    def __make_bcy_slider_row__(self, bc_slide):
        padding = Button(description=' ',
                         layout=Layout(flex='1 1 0%', width='auto', height='auto'),
                         button_style='',
                         disabled=True,
                         )
        padding.style.button_color = 'white'

        return [padding, bc_slide, padding]

    def __set_label_string__(self):
        hz = self.Om[self.m] / (2.0 * np.pi)
        self.mode_label.value = f"Mode #{self.m + 1}: {hz:.2f} Hz"

    def set_mode_number(self, b):
        m = self.m
        if b.icon == 'arrow-left':
            if m > 0:
                m -= 1
        elif b.icon == 'arrow-right':
            if m < (self.Nmodes - 1):
                m += 1

        if self.m != m:
            self.m = m
            if self.auto_plot:
                self.__set_label_string__()
                self.plot_mode_shapes()

    def set_BCs(self, K0y, Kx0, KLy, KxL, R0y, Rx0, RLy, RxL):
        self.BCs = np.array([[Kx0, Rx0], [K0y, R0y], [KxL, RxL], [KLy, RLy]], dtype=np.float64)

        if self.auto_generate:
            self.generate_modes()
            self.__set_label_string__()

    def set_dimensions(self, Lx, Ly, Lz):
        self.ldim = [Lx, Ly, Lz]

        if self.auto_generate:
            self.generate_modes()

    def set_material(self, E, rho):
        self.E = E  # -- Young's mod [Pa]
        self.rho = rho  # -- density [kg/m^3]
        self.generate_modes()

    def set_youngs(self, E):
        self.E = E  # -- Young's mod [Pa]
        self.generate_modes()

    def set_youngs_gpa(self, E):
        self.E = E * 1e9  # -- Young's mod [Pa]
        self.generate_modes()

    def set_density(self, rho):
        self.rho = rho  # -- density [kg/m^3]
        self.generate_modes()

    def set_poisson(self, nu):

        self.nu = nu
        self.generate_modes()

    def set_resolution(self, res):
        self.resolution = np.interp(res, [0, 1], [0.05, 0.005])
        self.h = np.sqrt(self.ldim[0] * self.ldim[1]) * self.resolution  # -- grid spacing

        if self.auto_generate:
            self.generate_modes()
        self.__set_label_string__()

    def set_material_labels(self, m):
        E, rho = [materials.loc[m]['youngs'], materials.loc[m]['densities']]

        self.material_coefs[0].value = E
        self.material_coefs[1].value = rho * 1e3
        self.material_coefs[2].value = self.nu

        self.set_material(E * 1e9, rho * 1e3)

    def generate_modes(self):

        self.Om, self.Q, self.N, self.biharm = magpie(self.rho, self.E, self.nu, self.ldim, self.h, self.BCs,
                                                      self.Nmodes)
        self.__set_label_string__()
        for m in range(self.Nmodes):
            self.mode_shapes = [np.real(np.reshape(self.Q[:, m], [self.N['x'], self.N['y']])) for m in
                                range(self.Nmodes)]

        if self.auto_plot:
            self.plot_mode_shapes()

    def plot_mode_shapes(self):
        """
        """
        m = self.m
        ratio = self.ldim[1] / self.ldim[0]
        with self.chladni_plot:
            clear_output(wait=True)

            fig = plt.figure(figsize=(4, 4 * ratio))
            ax = fig.add_subplot(111)

            Z = abs(self.mode_shapes[m])
            chladni = plt.pcolormesh(Z.T, cmap='copper_r', shading='gouraud')
            ax.set_axis_off()
            cmax = np.max(Z) * 0.35
            chladni.set_clim(0.00, cmax)

            plt.plot()
            plt.show()

    def get_current_mode_freq_hz(self):
        """
        return the current displayed modal frequency in Hz
        """
        return self.Om[self.m] / (2 * np.pi)

    def get_current_mode_shape(self):
        """
        Return the current modal shape as a dictionary of x / y coordinates and z values
        """
        x, y = np.mgrid[0:self.N['x'], 0:self.N['y']]
        return {'x': x.flatten().astype('int64'), 'y': y.flatten().astype('int64'), 'z': np.real(self.Q[:, self.m])}

    def show(self):
        display(self.interface)
