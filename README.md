# Nuclear Shape Plotter (Cassini Parametrization)

An interactive visualization tool for plotting nuclear shapes using Cassini oval parametrization with volume conservation.

## Features

- Interactive GUI with sliders and buttons for parameter adjustment
- Real-time shape visualization
- Volume conservation calculations
- Neck thickness measurements
- Axis length calculations
- Parameter input via keyboard (supports copy/paste)
- Plot saving functionality

## Requirements

- Python 3.x
- NumPy
- Matplotlib
- SciPy

## Installation

Install the required packages using pip:

```bash
pip install numpy matplotlib scipy
```

## Usage

Run the script using Python:

```bash
python ShapePlotterCassini.py
```

### Controls

- **Sliders:**
  - Z (Proton number): 82-120
  - N (Neutron number): 100-180
  - α1 through α6 (Deformation parameters): -1.0 to 1.0

- **Buttons:**
  - +/- buttons for fine adjustment of each parameter
  - Save Plot: Saves the current plot as PNG
  - Reset: Resets all parameters to default values

- **Keyboard Input:**
  - Format: Z N α1 α2 α3 α4 α5 α6
  - Example: 102 154 0.0 0.5 0.0 0.0 0.0 0.0
  - Supports Ctrl+V for pasting values

### Display Information

- Sphere and shape volumes
- Volume fixing factors
- Maximum lengths along X and Y axes
- Neck thickness measurements (45°-135° and 30°-150°)
- Volume conservation status

### Plot Features

- Red line: X-axis measurement
- Blue line: Y-axis measurement
- Green line: Neck thickness (45°-135°)
- Purple line: Neck thickness (30°-150°)

## Output

The plot shows:
- Nuclear shape outline
- Axis measurements
- Neck thickness measurements
- Volume and dimensional calculations

Saved plots are named using the format:
`cassini_Z_N_α1_α2_α3_α4_α5_α6.png`

## Notes

- The script uses r0 = 1.16 fm as the radius constant
- Volume conservation is automatically applied
- Negative radius warnings are displayed when detected
