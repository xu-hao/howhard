# Project image provenance

## Spiral mark

[spiral.svg](spiral.svg) is howhard's project mark, published in the repository
README and documentation homepage. It is an abstract geometric spiral with uniform
stroke width, evenly spaced turns and rounded ends. The source specifies exactly
two colors: solid black (`#000000`) for the line and solid white (`#ffffff`) for
the background. It contains no shading, gradients, texture or embedded raster image.

The SVG was authored directly by Codex on 2026-11-14, following the user's selected
spiral concept and approval to publish the proposed geometric SVG replacement.
It replaces the earlier generated PNG. No third-party artwork was used.

Geometry starts with the Archimedean spiral `r(t) = 8 + 180t / (9π)` for
`0 ≤ t ≤ 9π`, rotated so its outer end points toward the lower right. Seventy-two cubic
Bézier segments approximate the curve using its analytic positions and derivatives.
The path is uniformly scaled to fit a 360-unit bounding extent and centered in a
512 by 512 view box. Its stroke is 17 units wide. Browser rendering may antialias
edges; the vector artwork itself uses only the two colors above.

The SVG's SHA-256 checksum is
`25f34101c7c9b64542c9724db8ebf02314aa5631e0edb94e5ab0d5dcae71501d`.

[Issue #89](https://github.com/xu-hao/howhard/issues/89) tracks replacing the soft
raster image with the precise black-and-white abstract spiral.

[Issue #91](https://github.com/xu-hao/howhard/issues/91) tracks increasing the
spiral from 2.5 to 4.5 turns and thickening its uniform stroke from 14 to 20 units,
while preserving its two colors and geometric styling.

[Issue #93](https://github.com/xu-hao/howhard/issues/93) tracks reducing the final
line weight from 20 to 17 units while retaining the 4.5 turns.

## Decision flow figure

[decision-flow.svg](decision-flow.svg) is the overview figure of the
[technical report](../technical-report.md). It shows the three stages of a decision
(capability, forecast and selection), the host running the chosen cell, the project
record that feeds the first two stages and receives each outcome, and the later defect
attribution that amends the record. It consists of rounded rectangles, straight arrows
and text set in a sans-serif font (`Helvetica, Arial, sans-serif`). The source specifies
exactly two colors: solid black (`#000000`) for strokes, arrowheads and text, and solid
white (`#ffffff`) for the background, so the figure reads the same in the light and dark
themes of the documentation. It contains no shading, gradients, texture or embedded
raster image.

The figure was authored directly as SVG for the technical report on 2026-11-27. No
third-party artwork was used. Its view box is 800 by 640 units; the boxes have 2-unit
strokes, except the project record, which has a 3-unit stroke. Browser rendering may
antialias edges and substitute an available sans-serif font.

The SVG's SHA-256 checksum is
`4de0c2a6846fd72ab47eaa0b32f305c4acd7159b5439b65c9ae7e38e8ffe3acf`.

[Issue #157](https://github.com/xu-hao/howhard/issues/157) tracks restructuring the
technical report around an introduction with this figure.
