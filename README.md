# Electrorotation Game

[![Documentation Status](https://readthedocs.org/projects/electrorotation-game/badge/?version=latest)](https://electrorotation-game.readthedocs.io/en/latest/?badge=latest)
[![DOI](https://zenodo.org/badge/1129930758.svg)](https://doi.org/10.5281/zenodo.18176276)
![Python Version](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue)
![License](https://img.shields.io/badge/license-MPL--2.0-green)
![Status](https://img.shields.io/badge/status-active-success)

> A synthetic video generator for simulating electrorotation of biological cells with configurable trajectories, backgrounds, and cell templates.

<p align="center">
  <img src="docs/source/_static/gif/rot_game.gif" alt="Demo" width="60%">
</p>

## Features

- **Visual Scenario Editor**: Interactive GUI for designing cell simulations
- **Flexible Trajectories**: Support for stationary, linear, parabolic, and cubic Bézier paths
- **Realistic Effects**: Camera noise simulation and edge feathering for seamless compositing
- **Ground Truth Export**: Precise tracking data for each frame (positions, angles, velocities)
- **Batch Rendering**: CLI tools for automated video generation
- **Plugin System**: Extensible exporter architecture for custom data formats

## Requirements

- Python 3.10–3.13
- NumPy, OpenCV, PySide6 (automatically installed)

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/AmatoAndrea/electrorotation-game.git
cd electrorotation-game

# Install with Poetry (recommended)
poetry install

# Or with pip
pip install .
```

### Basic Usage

**Launch the GUI:**
```bash
poetry run rot-game gui
```

**Render a simulation from config:**
```bash
poetry run rot-game render examples/rot_game.yaml
```

**Preview trajectories:**
```bash
poetry run rot-game preview examples/rot_game.yaml --output preview.png
```

## Documentation

Full documentation is available in the [`docs/`](docs/) directory:

- [Installation Guide](docs/source/installation.rst) - Detailed setup instructions
- [Quick Start Tutorial](docs/source/quickstart.rst) - Your first simulation in 5 minutes
- [Usage Guide](docs/source/usage.rst) - Command-line and YAML configuration reference
- [API Reference](docs/source/api.rst) - Programmatic usage

### Building Documentation Locally

```bash
cd docs
make html
# Open build/html/index.html in your browser
```

## Usage Examples

### Creating a Simple Scenario

```yaml
video:
  resolution: [640, 480]
  fps: 30
  duration_s: 3.0
  magnification: 10.0
  pixel_size_um: 6.5
  background_frames_dir: templates/backgrounds/sample_video

cells:
  - id: cell_001
    radius_um: 8.0
    template: templates/cell_lines/CaCo-2/cells/cell_01.png
    mask: 
      path: templates/cell_lines/CaCo-2/masks/cell_01_mask.png
    trajectory:
      type: linear
      start: [100, 200]
      end: [540, 200]

schedule:
  intervals:
    - frequency_khz: 30
      duration_s: 2.0
      angular_velocity_rad_s: 4.0
    - frequency_khz: 40
      duration_s: 1.0
      angular_velocity_rad_s: 0.0
```

Save as `my_scenario.yaml` and render:
```bash
poetry run rot-game render my_scenario.yaml --output results/
```

### Using the Plugin System

Export to custom formats by creating a plugin in `src/rot_game_plugins/`:

```python
from rot_game.plugins import ExporterPlugin, ExportResult

class MyExporter(ExporterPlugin):
    name = "my-format"
    description = "Export to my custom format"
    
    def export(self, frames, ground_truth, config, output_dir, **kwargs):
        # Your export logic here
        return ExportResult(output_path=output_dir / "output.dat")
```

Then use it:
```bash
poetry run rot-game render config.yaml --export-my-format
```

## Project Structure

```
electrorotation-game/
├── src/
│   └── rot_game/           # Main package
│       ├── gui/            # PySide6 GUI application
│       ├── core/           # Rendering engine
│       └── plugins/        # Plugin system
├── templates/              # Asset templates
│   ├── backgrounds/        # Background images
│   └── cell_lines/         # Cell templates and masks
├── examples/               # Example configurations
├── docs/                   # Sphinx documentation
```

## License

This project is licensed under the Mozilla Public License 2.0 (MPL-2.0) - see the [LICENSE](LICENSE) file for details.
