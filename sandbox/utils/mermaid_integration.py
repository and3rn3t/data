"""
Mermaid diagram integration utilities for Data Science Sandbox.

This module provides functions to generate and integrate Mermaid diagrams
into the documentation and dashboard components.
"""

import json
import subprocess
from pathlib import Path
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class MermaidRenderer:
    """Utility class for rendering Mermaid diagrams."""

    def __init__(self, diagrams_dir: Path = None, images_dir: Path = None):
        """Initialize the Mermaid renderer.

        Args:
            diagrams_dir: Directory containing .mmd files
            images_dir: Directory for output PNG images
        """
        self.project_root = Path(__file__).parent.parent.parent
        self.diagrams_dir = diagrams_dir or self.project_root / "docs" / "diagrams"
        self.images_dir = images_dir or self.project_root / "docs" / "images"

        # Ensure directories exist
        self.diagrams_dir.mkdir(parents=True, exist_ok=True)
        self.images_dir.mkdir(parents=True, exist_ok=True)

    def get_available_diagrams(self) -> List[str]:
        """Get list of available Mermaid diagram files.

        Returns:
            List of diagram names (without .mmd extension)
        """
        return [f.stem for f in self.diagrams_dir.glob("*.mmd")]

    def render_diagram(
        self, diagram_name: str, output_format: str = "png"
    ) -> Optional[Path]:
        """Render a specific Mermaid diagram.

        Args:
            diagram_name: Name of diagram (without extension)
            output_format: Output format (png, svg, pdf)

        Returns:
            Path to generated image file, None if failed
        """
        input_path = self.diagrams_dir / f"{diagram_name}.mmd"
        output_path = self.images_dir / f"{diagram_name}.{output_format}"

        if not input_path.exists():
            logger.error(f"Diagram file not found: {input_path}")
            return None

        try:
            # Use mermaid-cli to render diagram
            cmd = [
                "npx",
                "@mermaid-js/mermaid-cli",
                "-i",
                str(input_path),
                "-o",
                str(output_path),
                "-b",
                "white",
                "--scale",
                "2",
            ]

            result = subprocess.run(
                cmd, capture_output=True, text=True, cwd=self.project_root
            )

            if result.returncode == 0:
                logger.info(f"Successfully rendered {diagram_name}.{output_format}")
                return output_path
            else:
                logger.error(f"Failed to render diagram: {result.stderr}")
                return None

        except Exception as e:
            logger.error(f"Error rendering diagram {diagram_name}: {e}")
            return None

    def render_all_diagrams(self) -> Dict[str, Optional[Path]]:
        """Render all available Mermaid diagrams.

        Returns:
            Dictionary mapping diagram names to output paths
        """
        results = {}
        diagrams = self.get_available_diagrams()

        logger.info(f"Rendering {len(diagrams)} Mermaid diagrams...")

        for diagram in diagrams:
            results[diagram] = self.render_diagram(diagram)

        return results

    def create_markdown_image_reference(
        self, diagram_name: str, alt_text: str = None
    ) -> str:
        """Create markdown reference for rendered diagram.

        Args:
            diagram_name: Name of diagram
            alt_text: Alternative text for accessibility

        Returns:
            Markdown image reference string
        """
        alt_text = alt_text or f"{diagram_name.replace('-', ' ').title()} Diagram"
        image_path = f"images/{diagram_name}.png"
        return f"![{alt_text}]({image_path})"

    def get_mermaid_code_block(self, diagram_name: str) -> str:
        """Get Mermaid code block for direct embedding in markdown.

        Args:
            diagram_name: Name of diagram

        Returns:
            Mermaid code block string
        """
        input_path = self.diagrams_dir / f"{diagram_name}.mmd"

        if not input_path.exists():
            return f"<!-- Diagram {diagram_name} not found -->"

        try:
            content = input_path.read_text(encoding="utf-8")
            return f"```mermaid\n{content}\n```"
        except Exception as e:
            logger.error(f"Error reading diagram {diagram_name}: {e}")
            return f"<!-- Error loading diagram {diagram_name} -->"


def generate_architecture_diagram_data() -> Dict:
    """Generate dynamic data for architecture diagram.

    Returns:
        Dictionary with component information
    """
    return {
        "frontend": {
            "streamlit": {"status": "active", "port": 8501},
            "jupyter": {"status": "active", "port": 8888},
            "cli": {"status": "available"},
        },
        "application": {
            "game_engine": {"challenges": 25, "levels": 7},
            "progress_tracking": {"users": 1, "completions": 0},
            "dashboard": {"views": ["progress", "analytics", "challenges"]},
        },
        "integration": {
            "duckdb": {"version": "0.9+", "performance": "high"},
            "polars": {"version": "0.20+", "performance": "ultra-fast"},
            "mlflow": {"experiments": 0, "runs": 0},
        },
    }


# Example usage functions
def update_architecture_docs():
    """Update architecture documentation with latest diagrams."""
    renderer = MermaidRenderer()

    # Render all diagrams
    results = renderer.render_all_diagrams()

    # Generate markdown references
    arch_ref = renderer.create_markdown_image_reference(
        "architecture", "System Architecture Overview"
    )
    flow_ref = renderer.create_markdown_image_reference(
        "game-flow", "Game Flow Diagram"
    )
    pipeline_ref = renderer.create_markdown_image_reference(
        "data-pipeline", "Data Processing Pipeline"
    )

    logger.info("Architecture documentation updated with Mermaid diagrams")
    return {"architecture": arch_ref, "flow": flow_ref, "pipeline": pipeline_ref}


if __name__ == "__main__":
    # Example: Render all diagrams
    renderer = MermaidRenderer()
    results = renderer.render_all_diagrams()

    print("Rendered diagrams:")
    for name, path in results.items():
        status = "✅" if path else "❌"
        print(f"  {status} {name}")
