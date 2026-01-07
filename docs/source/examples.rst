Examples
========

This section provides a ready‑to‑run example configuration file that demonstrates
the simulator's capabilities.

Demo scenario
-------------

The repository includes a demonstration configuration file at ``examples/rot_game.yaml``
(also used to generate the preview video on the introduction page). This example
showcases multiple cells with different trajectory types (linear, parabolic, and cubic).

Render the demo video with:

.. code-block:: bash

   poetry run rot-game render examples/rot_game.yaml --output outputs/demo --include-frames

This command generates:

- ``demo.avi`` – the synthetic video
- ``demo_ground_truth.csv`` – per-frame position and rotation data for each cell
- Individual frame images (when ``--include-frames`` is used)

Interactive exploration
-----------------------

You can also open the demo configuration in the GUI to explore and modify the scenario
interactively:

.. code-block:: bash

   poetry run rot-game gui examples/rot_game.yaml

The GUI allows you to:

- Visualize trajectories and cell paths
- Adjust rotation schedules and timing
- Preview the simulation before rendering
- Save modifications back to the YAML configuration

.. note::
   The example configuration includes absolute paths to cell templates and background
   frames that may need to be adjusted for your local setup. See :doc:`installation`
   for details on setting up assets.
