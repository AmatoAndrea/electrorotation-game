.. Electrorotation Game documentation master file

Electrorotation Game
==================================

Electrorotation Game is a powerful and lightweight simulator for generating
synthetic microscopy videos of rotating cells with fully controlled
translational and rotational motion. By combining real cell templates with
cell-free background frame sequences, it produces realistic scenes with
precisely known ground truth.

.. raw:: html

   <div class="hero-video">
     <video autoplay loop muted playsinline width="80%" aria-label="Rotating cell simulation preview">
       <source src="_static/video/rot_game.mp4" type="video/mp4">
       <source src="_static/video/rot_game.webm" type="video/webm">
       Your browser does not support the video tag.
     </video>
   </div>

Features
--------

- **Controlled motion** – define stationary, linear or curved trajectories and assign signed angular velocities for each interval.
- **Realistic appearance** – combine cell‑free background videos with real cell templates and optional edge feathering and sensor‑like noise.
- **Deterministic outputs** – produce a video plus concise metadata and per‑frame ground truth for every cell.
- **Flexible workflows** – operate via command‑line for batch simulations or use the interactive GUI for visual scenario design and preview.

The following pages explain how to install the package, run a quick simulation, explore advanced usage options, review example scenarios and refer to the API.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   quickstart
   usage
   examples
   license

.. toctree::
   :maxdepth: 2
   :caption: Reference:

   api
