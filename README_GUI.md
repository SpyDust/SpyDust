# Grain Size Distribution GUI

This GUI application allows interactive exploration of grain size distribution parameters and their effects on spectral energy distribution (SED) characteristics.

## Features

The GUI provides:

1. **Interactive Parameter Control**: Sliders and spin boxes for four key parameters:
   - **d**: Grain thickness (0.5 × Grain.d to 2 × Grain.d)
   - **C**: Power law index (-3.5 to 1.0)
   - **log_a0**: Log of characteristic grain size (ln(1e-8) to ln(1e-6))
   - **sigma_inv**: Inverse width parameter (0 to 10)

2. **Four Real-time Plots**:
   - **Plot 1**: Grain size distribution dn/da vs grain size a
   - **Plot 2**: Spectral Energy Distribution (SED) vs frequency
   - **Plot 3**: Peak frequency vs selected parameter
   - **Plot 4**: SED width vs selected parameter

3. **Parameter Selection**: Radio buttons to choose which parameter to vary in plots 3 and 4

## Installation

1. Install required dependencies:
   ```bash
   pip install -r gui_requirements.txt
   ```

2. Ensure SpyDust is properly installed and accessible in your Python path.

## Usage

1. Run the GUI:
   ```bash
   python grain_size_gui.py
   ```

2. **Adjust Parameters**:
   - Use sliders for quick adjustments
   - Use spin boxes for precise values
   - All plots update in real-time

3. **Select Parameter for Analysis**:
   - Choose which parameter to vary using radio buttons
   - Plots 3 and 4 will show how peak frequency and width depend on the selected parameter
   - The red dashed line shows the current parameter value

## Understanding the Plots

### Plot 1: Grain Size Distribution
- Shows the normalized grain size distribution dn/da
- **C**: Controls the power-law slope
- **log_a0**: Sets the characteristic grain size (peak location)
- **sigma_inv**: Controls the width of the distribution

### Plot 2: Spectral Energy Distribution
- Shows the emission spectrum vs frequency
- Uses either real emulator predictions or demo calculations
- Frequency axis is logarithmic (GHz)

### Plot 3: Peak Frequency vs Parameter
- Shows how the SED peak frequency varies with the selected parameter
- Useful for understanding parameter degeneracies

### Plot 4: Width vs Parameter
- Shows how the SED width varies with the selected parameter
- Helps understand spectral broadening effects

## Technical Details

### Grain Size Distribution Function
The grain size distribution follows:
```
dn/da ∝ exp[C ln(a) - 0.5 × ((ln(a) - ln(a0)) × sigma_inv)²]
```

### Emulators
- The GUI attempts to load pre-trained polynomial emulators
- If emulators are not available, it falls back to demo mode with synthetic data
- Real emulators require the MomentEmu package and pre-computed training data

### Demo Mode
- When real emulators are unavailable, the GUI uses simplified analytical models
- Demo SEDs are log-normal shaped with parameter-dependent peak frequencies and widths
- This allows exploration of the GUI functionality even without full emulator setup

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure SpyDust is in your Python path
2. **Missing Dependencies**: Install requirements using `pip install -r gui_requirements.txt`
3. **Emulator Loading Failures**: The GUI will automatically fall back to demo mode
4. **Display Issues**: Ensure you have a working display (X11 forwarding if using SSH)

### Performance Notes

- Real-time plotting may be slower with actual emulators
- Demo mode provides faster response for parameter exploration
- Consider reducing plot resolution for better performance on slower systems

## Files

- `grain_size_gui.py`: Main GUI application
- `gui_requirements.txt`: Python dependencies
- `README_GUI.md`: This documentation

## Future Enhancements

- Save/load parameter configurations
- Export plots and data
- Additional visualization options
- Parameter fitting capabilities
- Integration with observational data