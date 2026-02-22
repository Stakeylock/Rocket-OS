"""
Autonomous Landing and Hazard Avoidance Technology (ALHAT) subsystem.

Implements terrain-relative navigation for autonomous precision landing.
The pipeline mirrors NASA's ALHAT architecture:

    Lidar Scan -> Digital Elevation Model -> Hazard Detection -> Site Selection

The HazardDetector uses convolution-based slope analysis and threshold
classification inspired by CNN feature extraction.  The LandingSiteSelector
scores candidate sites using a weighted multi-objective function (fuel cost,
safety margin, slope penalty, distance).

References:
    - Epp, Autolander, & Robinson, "Autonomous Landing and Hazard Avoidance
      Technology (ALHAT)", IEEE Aerospace Conference, 2008.
    - Johnson et al., "Lidar-based Hazard Avoidance for Safe Landing on Mars",
      Journal of Guidance, Control, and Dynamics, 2002.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import SimConfig


# ---------------------------------------------------------------------------
# Enums & data classes
# ---------------------------------------------------------------------------

class HazardType(Enum):
    """Classification of terrain hazards for landing site evaluation."""
    ROCK = auto()          # Boulder or rock field -- mechanical damage risk
    CRATER = auto()        # Depression -- instability on rim, dust at floor
    SLOPE = auto()         # Excessive slope -- tip-over risk
    SOFT_GROUND = auto()   # Low bearing strength -- sinkage risk
    CLEAR = auto()         # Safe for landing


@dataclass
class TerrainCell:
    """Single cell in a digital elevation model grid.

    Attributes:
        x:            East coordinate in landing-frame (m).
        y:            North coordinate in landing-frame (m).
        elevation:    Height above datum (m).
        slope_angle:  Local surface slope (rad).
        hazard_type:  Classified hazard at this cell.
        safety_score: Aggregate safety metric in [0, 1] (1 = safest).
    """
    x: float
    y: float
    elevation: float
    slope_angle: float = 0.0
    hazard_type: HazardType = HazardType.CLEAR
    safety_score: float = 1.0


@dataclass
class LandingSite:
    """Candidate landing site with scoring metadata.

    Attributes:
        cx:              Centre x-coordinate (m).
        cy:              Centre y-coordinate (m).
        mean_elevation:  Average elevation over the pad footprint (m).
        mean_slope:      Average slope angle over the pad (rad).
        safety_score:    Aggregate safety [0, 1].
        fuel_cost:       Estimated delta-v to reach this site (m/s).
        total_score:     Weighted composite objective (higher = better).
    """
    cx: float
    cy: float
    mean_elevation: float
    mean_slope: float
    safety_score: float
    fuel_cost: float
    total_score: float = 0.0


# ---------------------------------------------------------------------------
# Digital Elevation Model
# ---------------------------------------------------------------------------

class DigitalElevationModel:
    """Procedurally-generated terrain elevation map.

    Generates a synthetic DEM using layered Perlin-like noise (octave sum
    of sine waves with random phases) to produce realistic-looking terrain
    with craters and boulders.

    Args:
        grid_size:   Number of cells per side (square grid).
        cell_size:   Physical size of each cell (m).
        base_elev:   Mean terrain elevation (m).
        roughness:   RMS height variation (m).
        seed:        RNG seed for reproducibility.
    """

    def __init__(
        self,
        grid_size: int = 128,
        cell_size: float = 1.0,
        base_elev: float = 0.0,
        roughness: float = 2.0,
        seed: int = 42,
    ) -> None:
        self.grid_size = grid_size
        self.cell_size = cell_size
        self.base_elev = base_elev
        self.roughness = roughness
        self._rng = np.random.default_rng(seed)

        # Grid coordinates
        coords = np.arange(grid_size) * cell_size
        self._x_coords = coords
        self._y_coords = coords
        self._xx, self._yy = np.meshgrid(coords, coords, indexing="ij")

        # Generate elevation data and derived products
        self._elevation: np.ndarray = self._generate_terrain()
        self._slope: np.ndarray = self._compute_slopes()
        self.grid: np.ndarray = self._build_cell_grid()

    # -- terrain generation --------------------------------------------------

    def _generate_terrain(self) -> np.ndarray:
        """Build a synthetic elevation map using octave noise + features.

        The terrain is composed of:
        1. Multi-octave sinusoidal noise (large-scale undulation).
        2. Gaussian craters (random depressions).
        3. Gaussian boulders (random bumps).

        Returns:
            (grid_size, grid_size) elevation array in metres.
        """
        n = self.grid_size
        elev = np.full((n, n), self.base_elev, dtype=np.float64)

        # --- multi-octave noise ---
        n_octaves = 5
        for octave in range(n_octaves):
            freq = 2.0 ** octave / (n * self.cell_size)
            amplitude = self.roughness / (2.0 ** octave)
            phase_x = self._rng.uniform(0, 2 * np.pi)
            phase_y = self._rng.uniform(0, 2 * np.pi)
            elev += amplitude * np.sin(2 * np.pi * freq * self._xx + phase_x)
            elev += amplitude * np.cos(2 * np.pi * freq * self._yy + phase_y)

        # --- craters (Gaussian depressions) ---
        n_craters = max(1, int(self._rng.poisson(lam=5)))
        for _ in range(n_craters):
            cx = self._rng.uniform(0, n * self.cell_size)
            cy = self._rng.uniform(0, n * self.cell_size)
            radius = self._rng.uniform(3.0, 15.0)
            depth = self._rng.uniform(0.5, 3.0)
            r2 = (self._xx - cx) ** 2 + (self._yy - cy) ** 2
            elev -= depth * np.exp(-r2 / (2 * radius ** 2))

        # --- rocks / boulders (Gaussian bumps) ---
        n_rocks = max(1, int(self._rng.poisson(lam=12)))
        for _ in range(n_rocks):
            rx = self._rng.uniform(0, n * self.cell_size)
            ry = self._rng.uniform(0, n * self.cell_size)
            radius = self._rng.uniform(0.3, 2.0)
            height = self._rng.uniform(0.2, 1.5)
            r2 = (self._xx - rx) ** 2 + (self._yy - ry) ** 2
            elev += height * np.exp(-r2 / (2 * radius ** 2))

        return elev

    def _compute_slopes(self) -> np.ndarray:
        """Compute slope angle at each cell using central differences.

        Returns:
            (grid_size, grid_size) slope array in radians.
        """
        dy, dx = np.gradient(self._elevation, self.cell_size)
        slope = np.arctan(np.sqrt(dx ** 2 + dy ** 2))
        return slope

    def _build_cell_grid(self) -> np.ndarray:
        """Construct object array of TerrainCell instances.

        Returns:
            (grid_size, grid_size) object array of TerrainCell.
        """
        n = self.grid_size
        grid = np.empty((n, n), dtype=object)
        for i in range(n):
            for j in range(n):
                grid[i, j] = TerrainCell(
                    x=float(self._xx[i, j]),
                    y=float(self._yy[i, j]),
                    elevation=float(self._elevation[i, j]),
                    slope_angle=float(self._slope[i, j]),
                )
        return grid

    # -- public query API ----------------------------------------------------

    def get_elevation(self, x: float, y: float) -> float:
        """Return interpolated elevation at an arbitrary (x, y) position.

        Uses bilinear interpolation between the four nearest grid nodes.

        Args:
            x: East coordinate (m).
            y: North coordinate (m).

        Returns:
            Interpolated elevation (m).
        """
        ix = x / self.cell_size
        iy = y / self.cell_size

        i0 = int(np.clip(np.floor(ix), 0, self.grid_size - 2))
        j0 = int(np.clip(np.floor(iy), 0, self.grid_size - 2))
        i1 = i0 + 1
        j1 = j0 + 1

        fx = ix - i0
        fy = iy - j0

        # Bilinear interpolation
        e00 = self._elevation[i0, j0]
        e10 = self._elevation[i1, j0]
        e01 = self._elevation[i0, j1]
        e11 = self._elevation[i1, j1]

        return float(
            e00 * (1 - fx) * (1 - fy)
            + e10 * fx * (1 - fy)
            + e01 * (1 - fx) * fy
            + e11 * fx * fy
        )

    def get_slope(self, x: float, y: float) -> float:
        """Return interpolated slope angle at an arbitrary (x, y) position.

        Args:
            x: East coordinate (m).
            y: North coordinate (m).

        Returns:
            Interpolated slope angle (rad).
        """
        ix = x / self.cell_size
        iy = y / self.cell_size

        i0 = int(np.clip(np.floor(ix), 0, self.grid_size - 2))
        j0 = int(np.clip(np.floor(iy), 0, self.grid_size - 2))
        i1 = i0 + 1
        j1 = j0 + 1

        fx = ix - i0
        fy = iy - j0

        s00 = self._slope[i0, j0]
        s10 = self._slope[i1, j0]
        s01 = self._slope[i0, j1]
        s11 = self._slope[i1, j1]

        return float(
            s00 * (1 - fx) * (1 - fy)
            + s10 * fx * (1 - fy)
            + s01 * (1 - fx) * fy
            + s11 * fx * fy
        )

    def get_patch(
        self, cx: float, cy: float, half_size: int = 8,
    ) -> np.ndarray:
        """Extract a square elevation sub-grid centred on (cx, cy).

        Args:
            cx:        Centre x-coordinate (m).
            cy:        Centre y-coordinate (m).
            half_size: Half-width of the patch in cells.

        Returns:
            (2*half_size, 2*half_size) elevation array.
        """
        ic = int(np.clip(cx / self.cell_size, half_size, self.grid_size - half_size - 1))
        jc = int(np.clip(cy / self.cell_size, half_size, self.grid_size - half_size - 1))
        return self._elevation[
            ic - half_size : ic + half_size,
            jc - half_size : jc + half_size,
        ].copy()


# ---------------------------------------------------------------------------
# Hazard Detector (CNN-inspired convolution pipeline)
# ---------------------------------------------------------------------------

class HazardDetector:
    """Detect and classify terrain hazards using convolution-based analysis.

    Mimics the feature extraction stage of a Convolutional Neural Network
    using hand-crafted kernels for slope detection, Laplacian-based crater
    detection, and local-maxima rock identification -- all implemented with
    numpy convolutions.

    Args:
        slope_threshold:  Slope above which terrain is classified SLOPE (rad).
        rock_threshold:   Local height anomaly for ROCK classification (m).
        crater_threshold: Laplacian magnitude for CRATER detection.
        roughness_window: Kernel half-width for roughness estimation (cells).
        seed:             RNG seed for soft-ground probability model.
    """

    # Sobel kernels for gradient estimation (3x3)
    _SOBEL_X: np.ndarray = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1],
    ], dtype=np.float64) / 8.0

    _SOBEL_Y: np.ndarray = np.array([
        [-1, -2, -1],
        [ 0,  0,  0],
        [ 1,  2,  1],
    ], dtype=np.float64) / 8.0

    # Laplacian kernel -- highlights craters and ridges
    _LAPLACIAN: np.ndarray = np.array([
        [0,  1, 0],
        [1, -4, 1],
        [0,  1, 0],
    ], dtype=np.float64)

    def __init__(
        self,
        slope_threshold: float = np.radians(10.0),
        rock_threshold: float = 0.4,
        crater_threshold: float = 0.8,
        roughness_window: int = 2,
        seed: int = 42,
    ) -> None:
        self.slope_threshold = slope_threshold
        self.rock_threshold = rock_threshold
        self.crater_threshold = crater_threshold
        self.roughness_window = roughness_window
        self._rng = np.random.default_rng(seed)

    # -- convolution helper --------------------------------------------------

    @staticmethod
    def _convolve2d(data: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        """Simple 2-D convolution using numpy (no scipy dependency).

        Performs zero-padded 'same'-size convolution via sliding window
        summation.  Suitable for small kernels (3x3 to 7x7).

        Args:
            data:   Input 2-D array.
            kernel: Convolution kernel (must be odd-sized square).

        Returns:
            Convolved array of same shape as *data*.
        """
        kh, kw = kernel.shape
        ph, pw = kh // 2, kw // 2
        padded = np.pad(data, ((ph, ph), (pw, pw)), mode="edge")
        out = np.zeros_like(data)
        for di in range(kh):
            for dj in range(kw):
                out += kernel[di, dj] * padded[di:di + data.shape[0],
                                                dj:dj + data.shape[1]]
        return out

    # -- feature extraction --------------------------------------------------

    def _compute_slope_map(self, elevation: np.ndarray, cell_size: float) -> np.ndarray:
        """Compute per-cell slope from elevation using Sobel convolutions.

        Args:
            elevation: 2-D elevation grid (m).
            cell_size: Physical size of one cell (m).

        Returns:
            (H, W) slope magnitude array (rad).
        """
        gx = self._convolve2d(elevation, self._SOBEL_X) / cell_size
        gy = self._convolve2d(elevation, self._SOBEL_Y) / cell_size
        return np.arctan(np.sqrt(gx ** 2 + gy ** 2))

    def _compute_roughness_map(self, elevation: np.ndarray) -> np.ndarray:
        """Local roughness as standard deviation in a sliding window.

        Args:
            elevation: 2-D elevation grid (m).

        Returns:
            (H, W) roughness array (m).
        """
        w = self.roughness_window
        h, ww = elevation.shape
        roughness = np.zeros_like(elevation)

        for i in range(h):
            for j in range(ww):
                i0 = max(0, i - w)
                i1 = min(h, i + w + 1)
                j0 = max(0, j - w)
                j1 = min(ww, j + w + 1)
                patch = elevation[i0:i1, j0:j1]
                roughness[i, j] = np.std(patch)

        return roughness

    def _compute_laplacian_map(self, elevation: np.ndarray) -> np.ndarray:
        """Laplacian feature map -- peaks at crater rims and boulder tops.

        Args:
            elevation: 2-D elevation grid (m).

        Returns:
            (H, W) absolute Laplacian response.
        """
        return np.abs(self._convolve2d(elevation, self._LAPLACIAN))

    # -- classification ------------------------------------------------------

    def detect_hazards(
        self,
        dem_patch: np.ndarray,
        cell_size: float = 1.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Run the full hazard detection pipeline on an elevation patch.

        Pipeline stages:
        1. Slope map (Sobel convolution).
        2. Roughness map (sliding-window std-dev).
        3. Laplacian response (crater / boulder edges).
        4. Threshold-based classification per cell.
        5. Safety score computation.

        Args:
            dem_patch: 2-D elevation array (m).
            cell_size: Physical size of one cell (m).

        Returns:
            hazard_map:  (H, W) integer array of HazardType.value.
            safety_map:  (H, W) float array of safety scores [0, 1].
        """
        slope_map = self._compute_slope_map(dem_patch, cell_size)
        roughness_map = self._compute_roughness_map(dem_patch)
        laplacian_map = self._compute_laplacian_map(dem_patch)

        h, w = dem_patch.shape
        hazard_map = np.full((h, w), HazardType.CLEAR.value, dtype=np.int32)
        safety_map = np.ones((h, w), dtype=np.float64)

        # --- Slope hazard ---
        slope_mask = slope_map > self.slope_threshold
        hazard_map[slope_mask] = HazardType.SLOPE.value
        # Safety degrades linearly from threshold to 2x threshold
        slope_penalty = np.clip(
            (slope_map - self.slope_threshold) / self.slope_threshold, 0.0, 1.0
        )
        safety_map -= 0.4 * slope_penalty

        # --- Rock hazard (high roughness + positive Laplacian) ---
        rock_mask = (
            (roughness_map > self.rock_threshold)
            & (laplacian_map > self.crater_threshold * 0.5)
            & (~slope_mask)
        )
        hazard_map[rock_mask] = HazardType.ROCK.value
        safety_map[rock_mask] -= 0.5

        # --- Crater hazard (strong negative Laplacian centre) ---
        # Craters have negative Laplacian at centre (concave up), detected
        # by strong laplacian magnitude in low-elevation local regions
        local_mean = self._convolve2d(
            dem_patch,
            np.ones((5, 5), dtype=np.float64) / 25.0,
        )
        depression = local_mean - dem_patch
        crater_mask = (
            (depression > 0.3)
            & (laplacian_map > self.crater_threshold)
            & (~slope_mask)
            & (~rock_mask)
        )
        hazard_map[crater_mask] = HazardType.CRATER.value
        safety_map[crater_mask] -= 0.6

        # --- Soft ground (probabilistic -- simulates regolith analysis) ---
        soft_prob = self._rng.random((h, w))
        soft_mask = (
            (soft_prob < 0.05)
            & (hazard_map == HazardType.CLEAR.value)
        )
        hazard_map[soft_mask] = HazardType.SOFT_GROUND.value
        safety_map[soft_mask] -= 0.3

        # Clamp safety scores
        safety_map = np.clip(safety_map, 0.0, 1.0)

        return hazard_map, safety_map


# ---------------------------------------------------------------------------
# Landing Site Selector
# ---------------------------------------------------------------------------

class LandingSiteSelector:
    """Score and rank candidate landing sites from a hazard-annotated DEM.

    Scoring function (higher is better):
        score = w_safety * mean_safety
              - w_slope  * mean_slope / slope_max
              - w_fuel   * fuel_cost  / fuel_budget
              - w_dist   * distance   / max_range

    Args:
        pad_radius:   Landing pad footprint radius (m).
        w_safety:     Weight for aggregated safety score.
        w_slope:      Weight for slope penalty.
        w_fuel:       Weight for fuel-cost penalty.
        w_dist:       Weight for distance penalty.
        min_safety:   Hard threshold -- reject sites below this safety mean.
        max_slope:    Hard threshold -- reject sites above this slope mean (rad).
    """

    def __init__(
        self,
        pad_radius: float = 5.0,
        w_safety: float = 0.40,
        w_slope: float = 0.20,
        w_fuel: float = 0.25,
        w_dist: float = 0.15,
        min_safety: float = 0.6,
        max_slope: float = np.radians(8.0),
    ) -> None:
        self.pad_radius = pad_radius
        self.w_safety = w_safety
        self.w_slope = w_slope
        self.w_fuel = w_fuel
        self.w_dist = w_dist
        self.min_safety = min_safety
        self.max_slope = max_slope

    def _estimate_fuel_cost(
        self,
        current_position: np.ndarray,
        current_velocity: np.ndarray,
        site_x: float,
        site_y: float,
        site_elev: float,
    ) -> float:
        """Estimate delta-v to divert to a candidate site.

        Uses a simplified energy-based approximation:
            dv ~ sqrt(2 * g * |dh|) + |lateral_divert| / t_go

        where t_go is estimated from current altitude and descent rate.

        Args:
            current_position: Vehicle [x, y, z] in landing frame (m).
            current_velocity: Vehicle [vx, vy, vz] (m/s).
            site_x:           Site x-coordinate (m).
            site_y:           Site y-coordinate (m).
            site_elev:        Site surface elevation (m).

        Returns:
            Estimated delta-v cost (m/s).
        """
        g = 9.81  # local gravity approximation

        # Time to ground (avoid division by zero)
        altitude = current_position[2] - site_elev
        descent_rate = max(-current_velocity[2], 1.0)
        t_go = max(altitude / descent_rate, 1.0)

        # Lateral divert cost
        dx = site_x - current_position[0]
        dy = site_y - current_position[1]
        lateral_dist = np.sqrt(dx ** 2 + dy ** 2)
        lateral_dv = lateral_dist / t_go

        # Vertical braking margin (energy-based)
        dh = abs(altitude)
        vertical_dv = np.sqrt(2.0 * g * dh) if dh > 0 else 0.0

        return float(lateral_dv + 0.1 * vertical_dv)

    def _gather_candidates(
        self,
        dem: DigitalElevationModel,
        safety_map: np.ndarray,
        slope_map: np.ndarray,
    ) -> List[LandingSite]:
        """Scan the DEM grid for candidate landing pads.

        A candidate is the centre of a circular pad region.  We stride
        across the grid with spacing = pad_radius to avoid redundant
        evaluations.

        Args:
            dem:        The digital elevation model.
            safety_map: (H, W) per-cell safety scores.
            slope_map:  (H, W) per-cell slope angles (rad).

        Returns:
            List of LandingSite candidates (un-scored).
        """
        n = dem.grid_size
        cs = dem.cell_size
        pad_cells = max(1, int(self.pad_radius / cs))
        stride = max(1, pad_cells)

        candidates: List[LandingSite] = []

        for i in range(pad_cells, n - pad_cells, stride):
            for j in range(pad_cells, n - pad_cells, stride):
                # Circular mask
                ii, jj = np.ogrid[
                    i - pad_cells : i + pad_cells + 1,
                    j - pad_cells : j + pad_cells + 1,
                ]
                mask = ((ii - i) ** 2 + (jj - j) ** 2) <= pad_cells ** 2

                patch_safety = safety_map[
                    i - pad_cells : i + pad_cells + 1,
                    j - pad_cells : j + pad_cells + 1,
                ]
                patch_slope = slope_map[
                    i - pad_cells : i + pad_cells + 1,
                    j - pad_cells : j + pad_cells + 1,
                ]

                # Ensure mask and patches are compatible shapes
                rows = min(mask.shape[0], patch_safety.shape[0])
                cols = min(mask.shape[1], patch_safety.shape[1])
                mask = mask[:rows, :cols]
                patch_safety = patch_safety[:rows, :cols]
                patch_slope = patch_slope[:rows, :cols]

                if mask.sum() == 0:
                    continue

                mean_safety = float(np.mean(patch_safety[mask]))
                mean_slope = float(np.mean(patch_slope[mask]))

                # Hard-reject unsafe candidates
                if mean_safety < self.min_safety:
                    continue
                if mean_slope > self.max_slope:
                    continue

                cx = float(i * cs)
                cy = float(j * cs)
                mean_elev = float(dem.get_elevation(cx, cy))

                candidates.append(LandingSite(
                    cx=cx,
                    cy=cy,
                    mean_elevation=mean_elev,
                    mean_slope=mean_slope,
                    safety_score=mean_safety,
                    fuel_cost=0.0,
                ))

        return candidates

    def select_site(
        self,
        current_state: Dict[str, np.ndarray],
        dem: DigitalElevationModel,
        safety_map: np.ndarray,
        fuel_remaining: float,
    ) -> Optional[LandingSite]:
        """Select the best landing site from the full DEM.

        Args:
            current_state:  Dict with keys "position" (3,) and "velocity" (3,).
            dem:            Digital elevation model.
            safety_map:     (H, W) per-cell safety scores from HazardDetector.
            fuel_remaining: Remaining propellant mass (kg).

        Returns:
            Best LandingSite or None if no acceptable site found.
        """
        position = current_state["position"]
        velocity = current_state["velocity"]

        # Compute slope map for candidate filtering
        dy_grad, dx_grad = np.gradient(dem._elevation, dem.cell_size)
        slope_map = np.arctan(np.sqrt(dx_grad ** 2 + dy_grad ** 2))

        candidates = self._gather_candidates(dem, safety_map, slope_map)

        if not candidates:
            return None

        # Fuel budget: Isp * g0 * ln(m0/m_dry) -- simplified available dv
        g0 = 9.81
        isp = 282.0
        dry_mass = 22_200.0
        fuel_budget = isp * g0 * np.log(
            max((dry_mass + fuel_remaining) / dry_mass, 1.01)
        )

        # Determine max range for normalisation
        max_range = 0.0
        for site in candidates:
            dx = site.cx - position[0]
            dy = site.cy - position[1]
            dist = np.sqrt(dx ** 2 + dy ** 2)
            if dist > max_range:
                max_range = dist
        max_range = max(max_range, 1.0)

        slope_max = max(self.max_slope, 1e-6)

        # Score each candidate
        best_site: Optional[LandingSite] = None
        best_score = -np.inf

        for site in candidates:
            site.fuel_cost = self._estimate_fuel_cost(
                position, velocity, site.cx, site.cy, site.mean_elevation,
            )

            # Reject if fuel cost exceeds budget
            if site.fuel_cost > fuel_budget:
                continue

            dx = site.cx - position[0]
            dy = site.cy - position[1]
            distance = np.sqrt(dx ** 2 + dy ** 2)

            score = (
                self.w_safety * site.safety_score
                - self.w_slope * site.mean_slope / slope_max
                - self.w_fuel * site.fuel_cost / max(fuel_budget, 1.0)
                - self.w_dist * distance / max_range
            )

            site.total_score = float(score)

            if score > best_score:
                best_score = score
                best_site = site

        return best_site


# ---------------------------------------------------------------------------
# ALHAT System (top-level orchestrator)
# ---------------------------------------------------------------------------

class ALHATSystem:
    """Autonomous Landing and Hazard Avoidance Technology orchestrator.

    Implements the full ALHAT pipeline:
        1. Lidar scan simulation -> raw elevation data.
        2. DEM construction from scan.
        3. Hazard detection via convolution feature extraction.
        4. Landing site selection and scoring.

    This class owns the DEM, HazardDetector, and LandingSiteSelector and
    exposes a single ``run_pipeline()`` call that executes the complete
    chain.

    Args:
        grid_size:       DEM resolution (cells per side).
        cell_size:       Physical cell size (m).
        terrain_roughness: RMS terrain height variation (m).
        pad_radius:      Landing pad footprint radius (m).
        slope_threshold: Maximum allowable slope (rad) for CLEAR classification.
        seed:            RNG seed.
    """

    def __init__(
        self,
        grid_size: int = 128,
        cell_size: float = 1.0,
        terrain_roughness: float = 2.0,
        pad_radius: float = 5.0,
        slope_threshold: float = np.radians(10.0),
        seed: int = 42,
    ) -> None:
        self.grid_size = grid_size
        self.cell_size = cell_size
        self.seed = seed

        # Sub-components
        self.dem: Optional[DigitalElevationModel] = None
        self.detector = HazardDetector(
            slope_threshold=slope_threshold,
            seed=seed,
        )
        self.selector = LandingSiteSelector(
            pad_radius=pad_radius,
        )

        # Cached pipeline products
        self._hazard_map: Optional[np.ndarray] = None
        self._safety_map: Optional[np.ndarray] = None
        self._terrain_roughness = terrain_roughness

    # -- lidar scan simulation -----------------------------------------------

    def simulate_lidar_scan(self, seed: Optional[int] = None) -> DigitalElevationModel:
        """Simulate a flash-lidar terrain scan and build a DEM.

        In a real system this would process raw point-cloud data from the
        lidar.  Here we procedurally generate a synthetic terrain.

        Args:
            seed: Optional override seed for terrain generation.

        Returns:
            The generated DigitalElevationModel.
        """
        self.dem = DigitalElevationModel(
            grid_size=self.grid_size,
            cell_size=self.cell_size,
            roughness=self._terrain_roughness,
            seed=seed if seed is not None else self.seed,
        )
        return self.dem

    # -- full pipeline -------------------------------------------------------

    def run_pipeline(
        self,
        current_state: Dict[str, np.ndarray],
        fuel_remaining: float,
        lidar_seed: Optional[int] = None,
    ) -> Tuple[Optional[LandingSite], np.ndarray, np.ndarray]:
        """Execute the complete ALHAT pipeline.

        Steps:
            1. Simulate lidar scan (generate DEM).
            2. Detect hazards across the full DEM.
            3. Select the best landing site.

        Args:
            current_state:  Dict with "position" and "velocity" arrays.
            fuel_remaining: Remaining propellant mass (kg).
            lidar_seed:     Optional terrain seed override.

        Returns:
            Tuple of (best_site_or_None, hazard_map, safety_map).
        """
        # Step 1: Lidar scan -> DEM
        self.simulate_lidar_scan(seed=lidar_seed)

        # Step 2: Hazard detection on full DEM elevation grid
        self._hazard_map, self._safety_map = self.detector.detect_hazards(
            self.dem._elevation,
            cell_size=self.cell_size,
        )

        # Step 3: Site selection
        best_site = self.selector.select_site(
            current_state=current_state,
            dem=self.dem,
            safety_map=self._safety_map,
            fuel_remaining=fuel_remaining,
        )

        return best_site, self._hazard_map, self._safety_map

    # -- query cached results ------------------------------------------------

    @property
    def hazard_map(self) -> Optional[np.ndarray]:
        """Return the most recently computed hazard map (or None)."""
        return self._hazard_map

    @property
    def safety_map(self) -> Optional[np.ndarray]:
        """Return the most recently computed safety map (or None)."""
        return self._safety_map
