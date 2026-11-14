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
