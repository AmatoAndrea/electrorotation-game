"""
Command-line interface for the next-generation simulator.

Usage:
    cell-simulator render path/to/config.yaml [--output outputs] [--feather 0] [--include-frames]
    cell-simulator validate path/to/config.yaml [--verbose]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from . import AssetManager, Renderer, BackgroundProvider, build_frame_schedule, export_simulation_outputs, load_simulation_config
from .catalog import find_mask_for_template, list_backgrounds, list_cell_templates
from .scaffold import choose_assets, write_stub
from .settings import asset_root
from .plugins import discover_plugins, discover_trajectory_plugins, ExporterPlugin, CLIArgument

Logger = logging.getLogger(__name__)

# Discover trajectory plugins early so custom types are available for config loading
discover_trajectory_plugins()

# Cache for discovered plugins
_discovered_plugins: Optional[dict] = None


def _get_plugins() -> dict:
    """Get cached plugins or discover them."""
    global _discovered_plugins
    if _discovered_plugins is None:
        _discovered_plugins = discover_plugins()
    return _discovered_plugins


def _register_plugin_arguments(parser: argparse.ArgumentParser) -> None:
    """Register CLI arguments for all discovered exporter plugins."""
    plugins = _get_plugins()
    
    for plugin_name, plugin in plugins.items():
        # Add --export-<name> flag for each plugin
        parser.add_argument(
            f"--export-{plugin_name}",
            action="store_true",
            help=plugin.description,
        )
        
        # Add plugin-specific arguments
        for arg in plugin.get_cli_arguments():
            kwargs = {
                "type": arg.type if arg.type != bool else None,
                "default": arg.default,
                "help": arg.help,
            }
            if arg.action:
                kwargs["action"] = arg.action
                kwargs.pop("type", None)
            if arg.required:
                kwargs["required"] = arg.required
            
            parser.add_argument(arg.name, **kwargs)


def _get_requested_plugins(args: argparse.Namespace) -> list:
    """Get list of plugins that were requested via CLI flags."""
    plugins = _get_plugins()
    requested = []
    
    for plugin_name, plugin in plugins.items():
        flag_name = f"export_{plugin_name}".replace("-", "_")
        if getattr(args, flag_name, False):
            requested.append(plugin)
    
    return requested


def _collect_plugin_kwargs(args: argparse.Namespace, plugin: ExporterPlugin) -> dict:
    """Collect plugin-specific arguments from parsed CLI args."""
    kwargs = {}
    
    for arg in plugin.get_cli_arguments():
        # Convert --some-arg to some_arg
        arg_name = arg.name.lstrip("-").replace("-", "_")
        if hasattr(args, arg_name):
            kwargs[arg_name] = getattr(args, arg_name)
    
    return kwargs


def configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(message)s")


def add_shared_config_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "config",
        type=Path,
        help="Path to the YAML configuration file describing the simulation scenario.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cell-simulator",
        description="Simulate rotating cells from YAML configurations.",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # render command
    render_parser = subparsers.add_parser(
        "render",
        help="Render a simulation from configuration and export outputs (.avi, .csv, .json, optional PNG frames).",
    )
    add_shared_config_argument(render_parser)
    bg_override = render_parser.add_mutually_exclusive_group()
    bg_override.add_argument(
        "--background-image",
        type=Path,
        help="Override the background image defined in the config.",
    )
    bg_override.add_argument(
        "--background-frames",
        type=Path,
        help="Use a folder of background frames instead of a static image.",
    )
    render_parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Root directory for exported artefacts (defaults to ROT_GAME_OUTPUTS or project outputs/).",
    )
    render_parser.add_argument(
        "--feather",
        type=int,
        default=0,
        help="Erode mask edges by this many pixels to reduce seams (default: 0).",
    )
    render_parser.add_argument(
        "--codec",
        type=str,
        default="XVID",
        help="FourCC codec for AVI export (default: XVID).",
    )
    render_parser.add_argument(
        "--include-frames",
        action="store_true",
        help="Also export individual PNG frames alongside the video.",
    )
    render_parser.add_argument(
        "--timestamp",
        type=str,
        default=None,
        help="Override timestamp component of the output directory (mainly for testing).",
    )
    
    # Discover and register exporter plugins
    _register_plugin_arguments(render_parser)

    # validate command
    validate_parser = subparsers.add_parser(
        "validate",
        help="Validate configuration and resources; prints a summary without rendering.",
    )
    add_shared_config_argument(validate_parser)
    validate_bg_override = validate_parser.add_mutually_exclusive_group()
    validate_bg_override.add_argument(
        "--background-image",
        type=Path,
        help="Override the background image defined in the config.",
    )
    validate_bg_override.add_argument(
        "--background-frames",
        type=Path,
        help="Use a folder of background frames instead of a static image.",
    )

    # gui command
    gui_parser = subparsers.add_parser(
        "gui",
        help="Launch the GUI authoring tool.",
    )
    gui_parser.add_argument(
        "config",
        type=Path,
        nargs="?",
        help="Optional configuration file to load on startup.",
    )

    # scaffold command
    scaffold_parser = subparsers.add_parser(
        "scaffold",
        help="Create a stub YAML scenario (optionally selecting assets).",
    )
    scaffold_parser.add_argument("path", type=Path, help="Path where the YAML stub will be written.")
    scaffold_parser.add_argument("--scenario", type=str, default="new_scenario", help="Scenario name metadata.")
    scaffold_parser.add_argument("--background", type=Path, help="Relative or absolute path to background image.")
    scaffold_parser.add_argument("--cell", type=Path, help="Relative or absolute path to cell template.")
    scaffold_parser.add_argument("--mask", type=Path, help="Relative or absolute path to mask image.")
    scaffold_parser.add_argument("--cell-line", type=str, help="Name of the cell line to use (defaults to first available).")
    scaffold_parser.add_argument("--fps", type=float, default=30.0, help="Frames per second (default 30).")
    scaffold_parser.add_argument("--duration", type=float, default=12.0, help="Duration in seconds (default 12).")

    # preview command
    preview_parser = subparsers.add_parser(
        "preview",
        help="Render trajectories without exporting artefacts (optional PNG preview).",
    )
    add_shared_config_argument(preview_parser)
    preview_parser.add_argument("--output", type=Path, help="Optional path to save trajectory preview PNG.")
    preview_parser.add_argument("--feather", type=int, default=0, help="Mask feathering pixels (default 0).")

    # mask-edit command
    mask_edit_parser = subparsers.add_parser(
        "mask-edit",
        help="Launch the mask editor dialog for a template/mask pair.",
    )
    mask_edit_parser.add_argument(
        "template",
        type=Path,
        help="Path to the cell template image.",
    )
    mask_edit_parser.add_argument(
        "--mask",
        type=Path,
        help="Path to the binary mask image (defaults to inferred pair for the template).",
    )

    return parser


def summarize_configuration(config_path: Path, config: Optional[SimulationConfig] = None) -> str:
    if config is None:
        config = load_simulation_config(config_path)
    bg_desc = (
        f"frames_dir={config.video.background_frames_dir}"
        if config.video.background_frames_dir is not None
        else f"image={config.video.background}"
    )
    lines = [
        f"Configuration: {config_path}",
        f"  Video: {config.video.resolution[0]}x{config.video.resolution[1]} px | "
        f"{config.video.duration_s:.3f} s @ {config.video.fps:.2f} fps",
        f"  Background: {bg_desc}",
        f"  Magnification: {config.video.magnification}x | Pixel size: {config.video.pixel_size_um} µm",
        f"  Cells ({len(config.cells)}):",
    ]
    for cell in config.cells:
        lines.append(
            f"    - {cell.id}: radius {cell.radius_um} µm | template {cell.template.name} | mask {cell.mask.path.name}"
        )
        lines.append(
            f"      Trajectory: {cell.trajectory.type} start={cell.trajectory.start} end={cell.trajectory.end}"
        )
    lines.append("  Schedule:")
    for idx, interval in enumerate(config.schedule.intervals):
        delay = interval.delay_after_s
        if delay is None:
            delay = config.schedule.default_delay_s
        lines.append(
            f"    {idx+1}. {interval.frequency_khz} kHz for {interval.duration_s}s "
            f"ω={interval.angular_velocity_rad_s} rad/s delay={delay}s"
        )
    return "\n".join(lines)


def render_command(args: argparse.Namespace) -> int:
    config_path: Path = args.config
    if not config_path.exists():
        Logger.error("Configuration file not found: %s", config_path)
        return 2

    try:
        config = load_simulation_config(config_path)
        if getattr(args, "background_image", None):
            config.video.background = Path(args.background_image).resolve()
            config.video.background_frames_dir = None
        elif getattr(args, "background_frames", None):
            config.video.background_frames_dir = Path(args.background_frames).resolve()
            config.video.background = None

        asset_manager = AssetManager(config)
        background = asset_manager.load_background()
        cell_templates = asset_manager.load_cell_templates()
        schedule = build_frame_schedule(config)

        required_frames = schedule.total_frames
        if isinstance(background, BackgroundProvider):
            source_frames = background.source_frame_count
            if source_frames < required_frames:
                if source_frames <= 2:
                    Logger.warning(
                        "Background frames (%d) are fewer than required frames (%d); the same %d frame(s) will repeat.",
                        source_frames,
                        required_frames,
                        source_frames,
                    )
                else:
                    Logger.warning(
                        "Background frames (%d) are fewer than required frames (%d); ping-pong looping will be used.",
                        source_frames,
                        required_frames,
                    )
        if getattr(background, "resized", False):
            Logger.warning("Background frames were auto-resized to match the target resolution.")
        renderer = Renderer(config, background, cell_templates, schedule)
        
        # Check if any export plugins are requested
        plugins = _get_requested_plugins(args)
        capture_masks = len(plugins) > 0
        
        render_result = renderer.render(feather_pixels=args.feather, capture_masks=capture_masks)
        
        # Unpack result (may or may not include masks)
        if capture_masks:
            frames, ground_truth, rotated_masks = render_result  # type: ignore
        else:
            frames, ground_truth = render_result  # type: ignore
            rotated_masks = None

        # Export standard simulation outputs
        output_dir = export_simulation_outputs(
            frames,
            ground_truth,
            config,
            output_root=args.output,
            include_frames=args.include_frames,
            codec=args.codec,
            feather_pixels=args.feather,
        )

        Logger.info("Render complete. Artefacts written to: %s", output_dir)
        
        # Run export plugins
        from .exporters import determine_scenario_name
        scenario_name = determine_scenario_name(config, fallback="scenario")
        video_path = output_dir / f"{scenario_name}.avi"
        
        for plugin in plugins:
            try:
                # Collect plugin-specific arguments
                plugin_kwargs = _collect_plugin_kwargs(args, plugin)
                plugin_kwargs["video_path"] = video_path if video_path.exists() else None
                
                result = plugin.export(
                    frames=frames,
                    ground_truth=ground_truth,
                    config=config,
                    output_dir=output_dir,
                    cell_templates=cell_templates,
                    rotated_masks=rotated_masks,
                    **plugin_kwargs,
                )
                Logger.info("%s", result.message or f"Plugin '{plugin.name}' export complete: {result.output_path}")
            except Exception as plugin_exc:
                Logger.error("Plugin '%s' failed: %s", plugin.name, plugin_exc)
                if Logger.isEnabledFor(logging.DEBUG):
                    Logger.exception("Plugin stack trace")
        
        return 0
    except Exception as exc:  # pylint: disable=broad-except
        Logger.error("Render failed: %s", exc)
        if Logger.isEnabledFor(logging.DEBUG):
            Logger.exception("Stack trace")
        return 1


def validate_command(args: argparse.Namespace) -> int:
    config_path: Path = args.config
    if not config_path.exists():
        Logger.error("Configuration file not found: %s", config_path)
        return 2

    try:
        config = load_simulation_config(config_path)
        if getattr(args, "background_image", None):
            config.video.background = Path(args.background_image).resolve()
            config.video.background_frames_dir = None
        elif getattr(args, "background_frames", None):
            config.video.background_frames_dir = Path(args.background_frames).resolve()
            config.video.background = None
        summary = summarize_configuration(config_path, config=config)
        print(summary)
        asset_manager = AssetManager(config)
        background = asset_manager.load_background()
        asset_manager.load_cell_templates()
        schedule = build_frame_schedule(config)

        required_frames = schedule.total_frames
        if isinstance(background, FrameSequenceBackgroundProvider):
            source_frames = background.source_frame_count
            if source_frames < required_frames:
                Logger.warning(
                    "Background frames (%d) are fewer than required frames (%d); ping-pong looping will be used.",
                    source_frames,
                    required_frames,
                )
        if getattr(background, "resized", False):
            Logger.warning("Background frames were auto-resized to match the target resolution.")

        Logger.info("Validation succeeded.")
        return 0
    except Exception as exc:  # pylint: disable=broad-except
        Logger.error("Validation failed: %s", exc)
        if Logger.isEnabledFor(logging.DEBUG):
            Logger.exception("Stack trace")
        return 1


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    configure_logging(args.verbose)

    if args.command == "render":
        return render_command(args)
    if args.command == "validate":
        return validate_command(args)
    if args.command == "gui":
        run_gui_with_args(args)
        return 0
    if args.command == "scaffold":
        return scaffold_command(args)
    if args.command == "preview":
        return preview_command(args)
    if args.command == "mask-edit":
        return mask_edit_command(args)

    parser.print_help()
    return 1


def run_gui_with_args(args: argparse.Namespace) -> None:
    config_path = None
    if args.config:
        config_path = Path(args.config)
        if not config_path.exists():
            Logger.error("Configuration file not found: %s", config_path)
            return
    from .gui import run as run_gui  # Local import to avoid Qt initialization unless needed

    run_gui(config_path)


def scaffold_command(args: argparse.Namespace) -> int:
    try:
        background = _coerce_asset_path(args.background) if args.background else None
        cell = _coerce_asset_path(args.cell) if args.cell else None
        mask = _coerce_asset_path(args.mask) if args.mask else None
        choices = choose_assets(background, cell, mask, cell_line=args.cell_line)
        write_stub(
            target_path=args.path,
            scenario_name=args.scenario,
            choices=choices,
            fps=args.fps,
            duration_s=args.duration,
        )
    except Exception as exc:  # noqa: BLE001
        Logger.error("Scaffold failed: %s", exc)
        return 1

    Logger.info("Scenario stub written to %s", args.path)
    return 0


def preview_command(args: argparse.Namespace) -> int:
    config_path: Path = args.config
    if not config_path.exists():
        Logger.error("Configuration file not found: %s", config_path)
        return 2

    try:
        config = load_simulation_config(config_path)
        assets = AssetManager(config)
        background = assets.load_background()
        cell_templates = assets.load_cell_templates()
        schedule = build_frame_schedule(config)
        renderer = Renderer(config, background, cell_templates, schedule)
        render_result = renderer.render(feather_pixels=args.feather, capture_masks=False)
        frames, ground_truth = render_result  # type: ignore
    except Exception as exc:  # noqa: BLE001
        Logger.error("Preview failed: %s", exc)
        return 1

    Logger.info("Preview succeeded: %d frames, %d cells", frames.shape[0], len(ground_truth.cells))

    if args.output:
        try:
            _write_preview_image(args.output, background.data, ground_truth)
            Logger.info("Preview image saved to %s", args.output)
        except Exception as exc:  # noqa: BLE001
            Logger.error("Failed to write preview image: %s", exc)
            return 1

    return 0


def mask_edit_command(args: argparse.Namespace) -> int:
    template_path: Path = args.template
    if not template_path.exists():
        Logger.error("Template image not found: %s", template_path)
        return 2

    if args.mask:
        mask_path = Path(args.mask)
    else:
        mask_path = find_mask_for_template(template_path)
        if mask_path is None:
            Logger.error(
                "No mask found matching %s. Specify --mask explicitly to launch the editor.",
                template_path,
            )
            return 2

    if not mask_path.exists():
        Logger.error("Mask file not found: %s", mask_path)
        return 2

    try:
        from PySide6.QtWidgets import QApplication
        from .gui.widgets import MaskEditorDialog
    except Exception as exc:  # noqa: BLE001
        Logger.error("Mask editor UI is unavailable: %s", exc)
        return 1

    app = QApplication.instance() or QApplication(sys.argv)
    dialog = MaskEditorDialog(template_path, mask_path)
    dialog.exec()
    return 0


from .core.renderer import GroundTruth


def _write_preview_image(output_path: Path, background: np.ndarray, ground_truth: GroundTruth) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(background, cmap="gray", origin="upper")
    for cell in ground_truth.cells:
        positions = cell.positions
        ax.plot(positions[:, 0], positions[:, 1], label=cell.cell_id)
        ax.scatter(positions[0, 0], positions[0, 1], marker="o", c="green")
        ax.scatter(positions[-1, 0], positions[-1, 1], marker="s", c="red")
    ax.legend(loc="upper right")
    ax.set_title("Trajectory Preview")
    ax.set_xlabel("X (px)")
    ax.set_ylabel("Y (px)")
    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _coerce_asset_path(path: Optional[Path]) -> Optional[Path]:
    if path is None:
        return None
    path = Path(path)
    if path.is_absolute():
        return path
    candidate = asset_root() / path
    return candidate.resolve()


if __name__ == "__main__":
    sys.exit(main())
