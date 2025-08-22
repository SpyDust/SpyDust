#!/usr/bin/env python3
"""
Grain Size Distribution GUI

This GUI allows interactive exploration of grain size distribution parameters
and their effects on SED characteristics.

Parameters:
- d: grain thickness (0.5 * Grain.d to 2 * Grain.d)
- C: power law index (-3.5 to 1.0)
- log_a0: log of characteristic grain size (ln(1e-8) to ln(1e-6))
- sigma_inv: inverse width parameter (0 to 10)
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, 
    QWidget, QSlider, QLabel, QRadioButton, QButtonGroup,
    QGridLayout, QGroupBox, QDoubleSpinBox
)
from PyQt5.QtCore import Qt
import pickle

# Add the SpyDust package to path
sys.path.append('/Users/zzhang/Workspace/SpyDust/')
sys.path.append('/Users/zzhang/Workspace/SpyDust/tests/grain_size_analysis/')

# Import required modules
import SpyDust.Grain as Grain
from SpyDust.util import makelogtab

from grain_size import *

params_list = np.load('/Users/zzhang/Workspace/SpyDust/tests/grain_size_analysis/CNM_parameter_list.npy')
SED_list = np.load('/Users/zzhang/Workspace/SpyDust/tests/grain_size_analysis/CNM_SED_list.npy')
feature_list = np.load('/Users/zzhang/Workspace/SpyDust/tests/grain_size_analysis/CNM_feature_list.npy')

peak_freq_min = np.exp(feature_list[:, 0].min())
peak_freq_max = np.exp(feature_list[:, 0].max())
width_min = feature_list[:, 1].min()
width_max = feature_list[:, 1].max()




freqs = np.load('/Users/zzhang/Workspace/SpyDust/tests/grain_size_analysis/freqs.npy')


class GrainSizeGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Grain Size Distribution Explorer")
        self.setGeometry(100, 100, 1200, 800)
        
        # Parameter ranges
        self.param_ranges = {
            'd': (d_min, d_max), 
            'C': (C_min, C_max),
            'log_a0': (log_a0_min, log_a0_max),
            'sigma_inv': (sigma_inv_min, sigma_inv_max)
        }
        
        # Default parameter values
        self.params = {
            'd': Grain.d,
            'C': -1.0,
            'log_a0': np.log(5e-8),
            'sigma_inv': 5.0
        }
        
        # Which parameter to vary for plots 3 and 4
        self.varied_param = 'd'
        
        # Create a_list for grain size distribution
        self.a_min = a_min
        self.a_max = a_max
        self.a_list = makelogtab(self.a_min, self.a_max, 100)
        
        # Create frequency array
        self.freqs = freqs
        
        # Load pre-computed parameter grid data
        print("Loading parameter grid data...")
        self.params_list = params_list
        self.SED_list = SED_list
        self.feature_list = feature_list
        
        self.init_ui()
        self.update_plots()
     
    def find_closest_params(self, target_params):
        """Find the closest parameter set in the grid using Euclidean distance"""
        # Normalize parameters for distance calculation
        param_ranges = np.array([[d_min, d_max], [C_min, C_max], [log_a0_min, log_a0_max], [sigma_inv_min, sigma_inv_max]])
        normalized_target = (np.array(target_params) - param_ranges[:, 0]) / (param_ranges[:, 1] - param_ranges[:, 0])
        normalized_grid = (self.params_list - param_ranges[:, 0]) / (param_ranges[:, 1] - param_ranges[:, 0])
        
        # Calculate distances
        distances = np.linalg.norm(normalized_grid - normalized_target, axis=1)
        closest_idx = np.argmin(distances)
        return closest_idx
    
    def get_sed_from_grid(self, params):
        """Get SED from parameter grid using closest match"""
        closest_idx = self.find_closest_params(params)
        return self.SED_list[closest_idx]
    
    def get_features_from_grid(self, params):
        """Get features from parameter grid using closest match"""
        closest_idx = self.find_closest_params(params)
        return self.feature_list[closest_idx]
    
    def get_features_array_from_grid(self, params_array):
        """Get features for an array of parameter sets"""
        features = np.zeros((len(params_array), 2))
        for i, params in enumerate(params_array):
            features[i] = self.get_features_from_grid(params)
        return features

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QHBoxLayout(central_widget)
        
        # Left panel for controls
        control_panel = self.create_control_panel()
        main_layout.addWidget(control_panel, 1)
        
        # Right panel for plots
        plot_panel = self.create_plot_panel()
        main_layout.addWidget(plot_panel, 3)
    
    def create_control_panel(self):
        control_widget = QWidget()
        layout = QVBoxLayout(control_widget)
        
        # Parameter sliders
        param_group = QGroupBox("Parameters")
        param_layout = QGridLayout(param_group)
        
        self.sliders = {}
        self.spinboxes = {}
        
        param_labels = {
            'd': 'Disc-Grain thickness ({:.2e}, {:.2e})'.format(d_min, d_max),
            'C': 'Power law index ({:.2e}, {:.2e})'.format(C_min, C_max),
            'log_a0': 'Log characteristic size ({:.2e}, {:.2e})'.format(log_a0_min, log_a0_max),
            'sigma_inv': 'Inverse width ({:.2e}, {:.2e})'.format(sigma_inv_min, sigma_inv_max)
        }
        
        row = 0
        for param, label in param_labels.items():
            # Label
            param_layout.addWidget(QLabel(label), row, 0, 1, 2)
            row += 1
            
            # Slider
            slider = QSlider(Qt.Horizontal)
            slider.setMinimum(0)
            slider.setMaximum(1000)
            slider.setValue(self.param_to_slider_value(param, self.params[param]))
            slider.valueChanged.connect(lambda v, p=param: self.on_slider_change(p, v))
            self.sliders[param] = slider
            param_layout.addWidget(slider, row, 0)
            
            # SpinBox for precise values
            spinbox = QDoubleSpinBox()
            spinbox.setRange(*self.param_ranges[param])
            spinbox.setValue(self.params[param])
            spinbox.setDecimals(6)
            spinbox.valueChanged.connect(lambda v, p=param: self.on_spinbox_change(p, v))
            self.spinboxes[param] = spinbox
            param_layout.addWidget(spinbox, row, 1)
            row += 1
        
        layout.addWidget(param_group)
        
        # Parameter selection for plots 3 and 4
        selection_group = QGroupBox("Parameter to vary (plots 3 & 4)")
        selection_layout = QVBoxLayout(selection_group)
        
        self.param_buttons = QButtonGroup()
        for param, label in param_labels.items():
            radio = QRadioButton(label)
            if param == self.varied_param:
                radio.setChecked(True)
            radio.toggled.connect(lambda checked, p=param: self.on_param_selection(p, checked))
            self.param_buttons.addButton(radio)
            selection_layout.addWidget(radio)
        
        layout.addWidget(selection_group)
        
        return control_widget
    
    def create_plot_panel(self):
        # Create matplotlib figure
        self.figure = Figure(figsize=(12, 10))
        self.canvas = FigureCanvas(self.figure)
        
        return self.canvas
    
    def param_to_slider_value(self, param, value):
        """Convert parameter value to slider value (0-1000)"""
        min_val, max_val = self.param_ranges[param]
        normalized = (value - min_val) / (max_val - min_val)
        return int(normalized * 1000)
    
    def slider_value_to_param(self, param, slider_value):
        """Convert slider value (0-1000) to parameter value"""
        min_val, max_val = self.param_ranges[param]
        normalized = slider_value / 1000.0
        return min_val + normalized * (max_val - min_val)
    
    def on_slider_change(self, param, value):
        param_value = self.slider_value_to_param(param, value)
        self.params[param] = param_value
        self.spinboxes[param].setValue(param_value)
        self.update_plots()
    
    def on_spinbox_change(self, param, value):
        self.params[param] = value
        slider_value = self.param_to_slider_value(param, value)
        self.sliders[param].setValue(slider_value)
        self.update_plots()
    
    def on_param_selection(self, param, checked):
        if checked:
            self.varied_param = param
            self.update_plots()
    
    def update_plots(self):
        self.figure.clear()
        
        # Create subplots
        gs = self.figure.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
        ax1 = self.figure.add_subplot(gs[0, 0])
        ax2 = self.figure.add_subplot(gs[0, 1])
        ax3 = self.figure.add_subplot(gs[1, 0])
        ax4 = self.figure.add_subplot(gs[1, 1])
        
        # Current parameter values
        current_params = [self.params['d'], self.params['C'], self.params['log_a0'], self.params['sigma_inv']]
        
        # Plot 1: Grain size distribution
        distribution = grain_size_dist(self.a_list, self.params['C'], self.params['log_a0'], self.params['sigma_inv'])
        ax1.loglog(self.a_list, distribution)
        dist_max = np.max(distribution)
        ax1.set_ylim(1e-8*dist_max, 10*dist_max)
        ax1.set_xlabel('Grain size a (cm)')
        ax1.set_ylabel('dn/da (normalized)')
        ax1.set_title('Grain Size Distribution')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: SED
        sed = self.get_sed_from_grid(current_params)
        
        ax2.loglog(self.freqs, sed)
        ax2.set_xlabel('Frequency (GHz)')
        ax2.set_ylabel('SED (normalised)')
        ax2.set_ylim(1e-4, 5)
        ax2.set_title('Spectral Energy Distribution')
        ax2.grid(True, alpha=0.3)
        
        # Plots 3 and 4: Features vs varied parameter
        varied_param = self.varied_param
        param_min, param_max = self.param_ranges[varied_param]
        param_values = np.linspace(param_min, param_max, 50)
        
        # Create parameter matrix: repeat current params for each test value
        num_test_points = len(param_values)
        test_params = np.tile(current_params, (num_test_points, 1))
        
        # Define parameter order for consistent indexing
        param_names = ['d', 'C', 'log_a0', 'sigma_inv']
        param_index = param_names.index(varied_param)
        
        # Replace the varied parameter column with test values
        test_params[:, param_index] = param_values

        features = self.get_features_array_from_grid(test_params)
        peak_freqs = np.exp(features[:, 0])
        widths = features[:, 1]
        
        # Plot 3: Peak frequency vs varied parameter
        ax3.plot(param_values, peak_freqs, 'b-', linewidth=2)
        ax3.axvline(self.params[varied_param], color='r', linestyle='--', alpha=0.7, label='Current value')
        ax3.set_xlabel(f'{varied_param}')
        ax3.set_ylabel('Peak Frequency (GHz)')
        ax3.set_yscale('log')
        ax3.set_ylim(1, 2*peak_freq_max)
        ax3.set_title(f'Peak Frequency vs {varied_param}')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        
        # Plot 4: Width vs varied parameter
        ax4.plot(param_values, widths, 'g-', linewidth=2)
        ax4.axvline(self.params[varied_param], color='r', linestyle='--', alpha=0.7, label='Current value')
        ax4.set_xlabel(f'{varied_param}')
        ax4.set_ylabel('Width')
        ax4.set_ylim(width_min-0.1, width_max+0.1)
        ax4.set_title(f'Width vs {varied_param}')
        ax4.grid(True, alpha=0.3)
        ax4.legend()
        
        self.canvas.draw()

def main():
    app = QApplication(sys.argv)
    
    window = GrainSizeGUI()
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()