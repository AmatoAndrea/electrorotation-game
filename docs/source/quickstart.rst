Quickstart
==========

This quick guide shows how to generate a minimal synthetic video using the
command-line interface and how to launch the graphical user interface for
interactive scenario design.

Command-line simulation
-----------------------

Assuming you have prepared a YAML configuration file (see :doc:`usage` for
details), and have your assets set up (see :doc:`installation`), you can render
a video and its ground truth with:

.. code-block:: bash

   poetry run rot-game render <your_config.yaml> --output <output_dir> --include-frames

This command writes a ``<scenario>.avi`` video and a ``<scenario>_ground_truth.csv``
file to ``<output_dir>``.  When ``--include-frames`` is used, each frame is
also saved as a PNG.

Graphical interface
-------------------

To design a scenario interactively, launch the graphical user interface with:

.. code-block:: bash

   poetry run rot-game gui [<your_config.yaml>]

The GUI lets you choose a background, select cell templates, draw trajectories,
assign rotation schedules, preview the sequence and save the configuration for
batch runs.

For a full description of available options and the configuration file structure,
continue to :doc:`usage`.
