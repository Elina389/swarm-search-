"""
Damage detection + severity prioritization for the unknown-terrain
(disaster) mode.

WHAT THIS IS, HONESTLY
----------------------
Real visual damage assessment classifies each structure from drone/satellite
imagery into a severity level -- the standard public benchmark for this is
the xBD dataset / xView2 challenge, with four classes:
    0 no-damage, 1 minor, 2 major, 3 destroyed.
A CNN fine-tuned on xBD would take an image patch and output that class.

This project has no camera feed and no trained damage model wired in, so the
per-cell severity here is a SYNTHETIC ground-truth stand-in, not a real
classifier output. It is generated from a plausible spatial pattern (damage
clusters near a disaster epicenter / fire origin, tapering with distance) so
the prioritization pipeline downstream is exercised realistically.

The important part is the SEAM: `DamageModel.severity_of(cell)` is the single
point a real classifier would replace. Everything downstream of it -- change
detection, priority density, priority-biased routing, map popups -- consumes
a severity number and does not care whether that number came from this
synthetic model or a real CNN. Swapping in xView2 weights later means
changing only this one method.

TWO WAYS TO PRODUCE THE SEVERITY SIGNAL (both supported by this design)
----------------------------------------------------------------------
1. Change detection against an OSM prior (no training data needed): OSM says
   a structure belongs in a cell; if live sensing no longer finds an intact
   structure there, that mismatch is itself the damage flag. See
   change_detection_flag() below -- it turns the existing OSM obstacle layer
   into an "expectation" to sense against.
2. Learned severity classification from imagery (xBD/xView2 CNN): a graded
   0-3 severity rather than a binary flag. That is what severity_of() stands
   in for.
"""
import numpy as np

# xView2/xBD-style severity scale.
SEVERITY_LABELS = {0: "intact", 1: "minor", 2: "major", 3: "destroyed"}

SEVERITY_DESC = {
    1: "Minor damage -- facade cracks and broken windows; structure appears "
       "intact and likely safe to approach.",
    2: "Major damage -- partial collapse; structure compromised, entry unsafe, "
       "possible trapped occupants.",
    3: "Destroyed -- full structural collapse; rubble field, high casualty "
       "risk, prioritize for rescue.",
}


def change_detection_flag(expected_occupied, sensed_occupied):
    """Approach #1: binary damage flag from an OSM prior, no model needed.

    A structure OSM says should exist but that live sensing no longer finds
    intact is flagged -- you don't need to know *why* it changed to know
    something is wrong. Returns 1.0 (flag) or 0.0 (no flag). Kept here as the
    documented, training-free alternative to the graded severity_of() signal.
    """
    return 1.0 if (expected_occupied and not sensed_occupied) else 0.0


class DamageModel:
    """Holds ground-truth damage severity for every obstacle cell in a
    disaster-mode environment. In reality this is what a drone's onboard
    damage classifier would infer per structure; here it is generated
    synthetically (see module docstring) from proximity to disaster
    epicenters, so that damage forms realistic clusters."""

    def __init__(self, grid_size, obstacles, seed=0, n_epicenters=2):
        self.grid_size = grid_size
        self.rng = np.random.default_rng(seed)
        self.severity = {}      # (r, c) -> 0..3, for every obstacle cell
        self.epicenters = []

        obstacles = list(obstacles)
        if not obstacles:
            return

        # Epicenters: a few obstacle cells that took the worst of it.
        k = min(n_epicenters, len(obstacles))
        idx = self.rng.choice(len(obstacles), size=k, replace=False)
        self.epicenters = [obstacles[i] for i in idx]

        radius = max(3.0, grid_size / 5.0)
        for (r, c) in obstacles:
            d = min(np.hypot(r - er, c - ec) for (er, ec) in self.epicenters)
            if d <= radius * 0.40:
                sev = 3
            elif d <= radius * 0.70:
                sev = 2
            elif d <= radius:
                sev = 1
            else:
                sev = 0
            self.severity[(r, c)] = int(sev)

    def severity_of(self, cell):
        """THE CLASSIFIER SEAM. Returns 0-3 for an obstacle cell (0 for cells
        with no recorded damage). A real xView2/xBD CNN would replace exactly
        this method, taking an image patch of `cell` and returning its damage
        class -- nothing downstream would need to change."""
        return self.severity.get(tuple(cell), 0)

    def describe(self, cell):
        """Human-readable explanation of what is wrong at a cell, for the
        map popup. Empty string if the cell has no damage."""
        sev = self.severity_of(cell)
        if sev == 0:
            return ""
        return SEVERITY_DESC[sev]


def _gaussian_kernel_1d(sigma):
    radius = max(1, int(round(3 * sigma)))
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    k = np.exp(-(x ** 2) / (2.0 * sigma ** 2))
    return k / k.sum()


def priority_density(severity_grid, sigma=2.0):
    """Turn a per-cell severity grid into a smooth priority heatmap.

    A tight cluster of high-severity cells stacks into a sharp priority
    peak, while scattered minor damage barely registers -- so ranking areas
    by this density prioritizes concentrated destruction (where rescue
    effort matters most) over isolated cosmetic damage. Implemented as a
    separable Gaussian blur using only numpy (no scipy dependency): blur the
    rows, then blur the columns, with the same 1-D kernel.

    severity_grid : 2-D float array (cell severity, optionally weighted by
        detection confidence). Unknown/unsensed cells should be 0.
    """
    grid = np.asarray(severity_grid, dtype=np.float64)
    k = _gaussian_kernel_1d(sigma)
    out = np.apply_along_axis(lambda row: np.convolve(row, k, mode="same"), 1, grid)
    out = np.apply_along_axis(lambda col: np.convolve(col, k, mode="same"), 0, out)
    return out


def top_priority_zones(priority_grid, k=3, min_score=1e-6):
    """Return up to k (row, col, score) peaks of the priority grid, greedily
    picking the highest-scoring cells while suppressing ones too close to an
    already-picked peak (so the list is distinct zones, not k cells of one
    blob). Used to populate a 'priority zones' list in the UI."""
    grid = np.asarray(priority_grid, dtype=np.float64)
    picked = []
    suppression = max(2, grid.shape[0] // 6)
    flat = sorted(
        ((grid[r, c], r, c) for r in range(grid.shape[0]) for c in range(grid.shape[1])),
        reverse=True,
    )
    for score, r, c in flat:
        if score < min_score or len(picked) >= k:
            break
        if all(abs(r - pr) > suppression or abs(c - pc) > suppression
               for (pr, pc, _) in picked):
            picked.append((r, c, float(score)))
    return picked
