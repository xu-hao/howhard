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

Geometry starts with the Archimedean spiral `r(t) = 8 + 180t / (5π)` for
`0 ≤ t ≤ 5π`, rotated so its outer end points toward the lower right. Forty cubic
Bézier segments approximate the curve using its analytic positions and derivatives.
The path is uniformly scaled to fit a 360-unit bounding extent and centered in a
512 by 512 view box. Its stroke is 14 units wide. Browser rendering may antialias
edges; the vector artwork itself uses only the two colors above.

The SVG's SHA-256 checksum is
`b4070d47f435b81c965aefe6818b7e2a4f96a14bb0379b80c4717ce34924a37f`.

[Issue #89](https://github.com/xu-hao/howhard/issues/89) tracks replacing the soft
raster image with the precise black-and-white abstract spiral.
