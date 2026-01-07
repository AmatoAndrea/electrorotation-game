Usage
=====

This page explains how to configure and run simulations via the command-line
interface and how to customise trajectories, rotation schedules, and rendering
options.

Running the command-line interface
----------------------------------

The simulator exposes a console script ``rot-game``. To list available
commands and options, run:

.. code-block:: bash

   poetry run rot-game --help

Use the ``render`` command to generate a synthetic video from a YAML configuration
file:

.. code-block:: bash

   poetry run rot-game render <config.yaml> --output <output_dir> [options]

Common options include ``--include-frames`` to save individual frames and
``--feather <pixels>`` to control edge fethering.

Defining trajectories
---------------------

Cell motion is defined by parametric trajectories in the configuration file.
Supported types are:

- ``stationary`` – the cell stays at a fixed position.
- ``linear`` – a straight path defined by ``start`` and ``end`` coordinates.
- ``parabolic`` – a curved path defined by a start point, an end point and one Bézier control point.
- ``cubic`` – a more complex path defined by two control points.

Coordinates are specified in pixels relative to the output resolution.

Specifying rotation schedules
-----------------------------

Rotational motion is controlled by a schedule consisting of a series of time
intervals.

Example schedule:

.. code-block:: yaml

   schedule:
     - duration: 1.0
       omega: 4.0      # rad/s, clockwise
     - duration: 0.5
       omega: 0.0      # pause
     - duration: 1.5
       omega: -4.0     # rad/s, counter-clockwise

All cells follow the same schedule in the current implementation.

Noise and feathering
--------------------

To improve realism, Gaussian noise is applied inside each cell mask to emulate
camera read-noise. Edge feathering softens transitions between cells and
background. These options can be configured via command-line flags or YAML
settings.

Output files
------------

After rendering, the simulator produces:

- ``<scenario>.avi`` – the synthetic video.
- ``<scenario>_ground_truth.csv`` – a CSV file listing frame number, cell ID, centroid coordinates, instantaneous angular velocity and cumulative angle.
- ``<scenario>_metadata.json`` – a JSON metadata file containing simulation settings.
- Optional ``frames/`` directory – individual frames as PNG files when ``--include-frames`` is used.

Config file template
--------------------

A minimal YAML configuration looks like this:

.. code-block:: yaml

   video:
     resolution: [640, 480]
     fps: 30
     background: data/backgrounds/sample.avi
     magnification: 10.0
     pixel_size: 6.5

   cells:
     - id: cell_1
       template: data/cell_lines/line_A/cells/cell_01.png
       mask: data/cell_lines/line_A/masks/cell_01_mask.png
       radius: 10.0
       trajectory:
         type: linear
         start: [100, 200]
         end: [540, 200]

   schedule:
     - duration: 2.0
       omega: 5.0
     - duration: 1.0
       omega: 0.0
     - duration: 2.0
       omega: -5.0

Refer to the repository for more detailed examples.