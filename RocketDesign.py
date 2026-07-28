from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st




st.set_page_config(
    page_title="Rocket Fin Optimizer",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
)




F15_8_MOTOR = {
    "name": "Loaded Estes F15-8 motor",
    "diameter_mm": 29.0,
    "length_mm": 114.0,
    "loaded_mass_g": 104.6,
    "propellant_mass_g": 60.0,
    "total_impulse_ns": 49.6,
    "average_thrust_n": 14.4,
    "max_thrust_n": 25.3,
    "burn_time_s": 3.45,
    "delay_s": 8.0,
    "listed_max_lift_weight_g": 425.0,
    "preferred_beginner_loaded_mass_g": 375.0,
}

MATERIALS: Dict[str, Dict[str, float]] = {
    "Balsa, edge grain": {
        "shear_modulus_pa": 170e6,
        "density_kg_m3": 160.0,
    },
    "Basswood / lite ply": {
        "shear_modulus_pa": 400e6,
        "density_kg_m3": 500.0,
    },
    "Birch plywood": {
        "shear_modulus_pa": 600e6,
        "density_kg_m3": 680.0,
    },
    "G10 fiberglass": {
        "shear_modulus_pa": 5e9,
        "density_kg_m3": 1850.0,
    },
    "Carbon fiber plate": {
        "shear_modulus_pa": 8e9,
        "density_kg_m3": 1600.0,
    },
}

DEFAULT_MASS_TABLE = pd.DataFrame(
    [
        {"Component": "Ogive nose cone", "Mass_g": 55.000, "X_from_nose_mm": 80.000},
        {"Component": "54 mm thick-wall body tube", "Mass_g": 85.000, "X_from_nose_mm": 350.000},
        {"Component": "Recovery system", "Mass_g": 45.000, "X_from_nose_mm": 240.000},
        {"Component": "Shock cord / mount", "Mass_g": 12.000, "X_from_nose_mm": 260.000},
        {"Component": "Rail buttons / launch guide", "Mass_g": 6.000, "X_from_nose_mm": 420.000},
        {"Component": "29 mm motor mount tube", "Mass_g": 18.000, "X_from_nose_mm": 630.000},
        {"Component": "Centering rings", "Mass_g": 18.000, "X_from_nose_mm": 620.000},
        {"Component": "Estes 29 mm motor retainer", "Mass_g": 8.000, "X_from_nose_mm": 685.000},
        {"Component": "Loaded Estes F15-8 motor", "Mass_g": 104.600, "X_from_nose_mm": 643.000},
        {"Component": "Paint / epoxy allowance", "Mass_g": 28.000, "X_from_nose_mm": 390.000},
    ]
)




@dataclass
class RocketConfig:
    body_diameter_mm: float
    body_length_mm: float
    nose_length_mm: float
    nose_type: str
    fin_count: int
    fin_leading_edge_x_mm: float
    max_speed_mps: float
    altitude_m: float
    target_margin_calibers: float
    min_margin_calibers: float
    max_margin_calibers: float
    min_flutter_safety_factor: float


@dataclass
class FinDesign:
    root_chord_mm: float
    tip_chord_mm: float
    span_mm: float
    sweep_mm: float
    thickness_mm: float
    material: str


@dataclass
class EvaluationResult:
    bare_mass_g: float
    fin_mass_g: float
    loaded_mass_g: float
    bare_cg_mm: float
    loaded_cg_mm: float
    cp_mm: float
    static_margin_calibers: float
    flutter_velocity_mps: float
    flutter_safety_factor: float
    fin_area_mm2: float
    fin_centroid_x_mm: float
    score: float
    warnings: List[str]



def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
        if math.isfinite(number):
            return number
        return default
    except Exception:
        return default


def mm_to_m(value_mm: float) -> float:
    return value_mm / 1000.0


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def fmt(value: float, decimals: int = 3) -> str:
    if not math.isfinite(value):
        return "Invalid"
    return f"{value:.{decimals}f}"


def show_plotly(fig: go.Figure) -> None:
    """
    Streamlit changed some width settings across versions.
    This wrapper keeps the app working on more installations.
    """
    try:
        st.plotly_chart(fig, width="stretch")
    except TypeError:
        st.plotly_chart(fig, use_container_width=True)


def show_dataframe(df: pd.DataFrame, height: int = 420) -> None:
    try:
        st.dataframe(df, width="stretch", height=height)
    except TypeError:
        st.dataframe(df, use_container_width=True, height=height)


def edit_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    try:
        return st.data_editor(
            df,
            num_rows="dynamic",
            width="stretch",
            column_config={
                "Component": st.column_config.TextColumn(
                    "Component",
                    required=True,
                    help="Name of the component.",
                ),
                "Mass_g": st.column_config.NumberColumn(
                    "Mass (g)",
                    min_value=0.0,
                    step=0.001,
                    format="%.3f",
                    help="Mass of the component in grams. Decimals are allowed, for example 104.600 g.",
                ),
                "X_from_nose_mm": st.column_config.NumberColumn(
                    "X from nose (mm)",
                    min_value=0.0,
                    step=0.001,
                    format="%.3f",
                    help="Location of the component center of mass from the nose tip.",
                ),
            },
        )
    except TypeError:
        return st.data_editor(
            df,
            num_rows="dynamic",
            use_container_width=True,
            column_config={
                "Component": st.column_config.TextColumn(
                    "Component",
                    required=True,
                    help="Name of the component.",
                ),
                "Mass_g": st.column_config.NumberColumn(
                    "Mass (g)",
                    min_value=0.0,
                    step=0.001,
                    format="%.3f",
                    help="Mass of the component in grams. Decimals are allowed, for example 104.600 g.",
                ),
                "X_from_nose_mm": st.column_config.NumberColumn(
                    "X from nose (mm)",
                    min_value=0.0,
                    step=0.001,
                    format="%.3f",
                    help="Location of the component center of mass from the nose tip.",
                ),
            },
        )




def clean_mass_table(mass_table: pd.DataFrame) -> pd.DataFrame:
    required_columns = ["Component", "Mass_g", "X_from_nose_mm"]

    table = mass_table.copy()

    for column in required_columns:
        if column not in table.columns:
            table[column] = "" if column == "Component" else 0.0

    table = table[required_columns].copy()
    table["Component"] = table["Component"].astype(str)
    table["Mass_g"] = pd.to_numeric(table["Mass_g"], errors="coerce").fillna(0.0).astype(float)
    table["X_from_nose_mm"] = pd.to_numeric(table["X_from_nose_mm"], errors="coerce").fillna(0.0).astype(float)

    table["Mass_g"] = table["Mass_g"].clip(lower=0.0)
    table["X_from_nose_mm"] = table["X_from_nose_mm"].clip(lower=0.0)

    return table


def total_mass_g(mass_table: pd.DataFrame) -> float:
    table = clean_mass_table(mass_table)
    return float(table["Mass_g"].sum())


def center_of_gravity_mm(mass_table: pd.DataFrame) -> float:
    table = clean_mass_table(mass_table)
    mass_sum = float(table["Mass_g"].sum())

    if mass_sum <= 0:
        return float("nan")

    moment_sum = float((table["Mass_g"] * table["X_from_nose_mm"]).sum())
    return moment_sum / mass_sum


def nose_cp_and_cna(config: RocketConfig) -> Tuple[float, float]:
    """
    Simplified nose contribution.

    Returns:
        CNa, x_cp_mm

    Approximate CP locations:
    - Conical: 2/3 nose length
    - Ogive: 0.466 nose length
    - Elliptical: 0.5 nose length
    """
    cna = 2.0

    if config.nose_type == "Conical":
        x_cp = (2.0 / 3.0) * config.nose_length_mm
    elif config.nose_type == "Ogive":
        x_cp = 0.466 * config.nose_length_mm
    elif config.nose_type == "Elliptical":
        x_cp = 0.5 * config.nose_length_mm
    else:
        x_cp = (2.0 / 3.0) * config.nose_length_mm

    return cna, x_cp


def fin_area_mm2(fin: FinDesign) -> float:
    return 0.5 * (fin.root_chord_mm + fin.tip_chord_mm) * fin.span_mm


def polygon_centroid_x(points: List[Tuple[float, float]]) -> float:
    """
    Returns the x-coordinate of the centroid of a polygon.
    Used to estimate fin center of mass location.
    """
    area2 = 0.0
    cx_times_6a = 0.0

    n = len(points)

    for i in range(n):
        x0, y0 = points[i]
        x1, y1 = points[(i + 1) % n]
        cross = x0 * y1 - x1 * y0
        area2 += cross
        cx_times_6a += (x0 + x1) * cross

    if abs(area2) < 1e-9:
        return float("nan")

    return cx_times_6a / (3.0 * area2)


def fin_centroid_x_mm(config: RocketConfig, fin: FinDesign) -> float:
    """
    Estimate x-location of fin mass center from the nose tip.
    """
    points = [
        (0.0, 0.0),
        (fin.root_chord_mm, 0.0),
        (fin.sweep_mm + fin.tip_chord_mm, fin.span_mm),
        (fin.sweep_mm, fin.span_mm),
    ]

    local_cx = polygon_centroid_x(points)

    if not math.isfinite(local_cx):
        return float("nan")

    return config.fin_leading_edge_x_mm + local_cx


def fin_cp_and_cna(config: RocketConfig, fin: FinDesign) -> Tuple[float, float]:
    """
    Simplified Barrowman-style estimate for trapezoidal fins.

    Assumptions:
    - Three or four identical fins
    - Fins are evenly spaced
    - Subsonic/small-angle estimate
    - No detailed body lift correction
    """
    D = config.body_diameter_mm
    N = config.fin_count
    s = fin.span_mm
    cr = fin.root_chord_mm
    ct = fin.tip_chord_mm
    sweep = fin.sweep_mm

    values = [D, N, s, cr, ct]

    if any((not math.isfinite(v)) or v <= 0 for v in values):
        return float("nan"), float("nan")

    mid_chord_sweep = sweep + 0.5 * (ct - cr)
    mid_chord_length = math.sqrt(mid_chord_sweep**2 + s**2)

    cna = (
        (1.0 + D / (2.0 * s + D))
        * (4.0 * N * (s / D) ** 2)
        / (1.0 + math.sqrt(1.0 + (2.0 * mid_chord_length / (cr + ct)) ** 2))
    )

    x_local = (
        sweep * (cr + 2.0 * ct) / (3.0 * (cr + ct))
        + (cr**2 + cr * ct + ct**2) / (3.0 * (cr + ct))
    )

    x_cp = config.fin_leading_edge_x_mm + x_local

    return cna, x_cp


def total_cp_mm(config: RocketConfig, fin: FinDesign) -> float:
    nose_cna, nose_x = nose_cp_and_cna(config)
    fin_cna, fin_x = fin_cp_and_cna(config, fin)

    if not all(math.isfinite(v) for v in [nose_cna, nose_x, fin_cna, fin_x]):
        return float("nan")

    denominator = nose_cna + fin_cna

    if denominator <= 0:
        return float("nan")

    return (nose_cna * nose_x + fin_cna * fin_x) / denominator


def air_properties_simple(altitude_m: float) -> Tuple[float, float]:
    """
    Simple ISA approximation up to 11 km.

    Returns:
        speed_of_sound_mps, static_pressure_pa
    """
    T0 = 288.15
    P0 = 101325.0
    lapse = 0.0065
    gamma = 1.4
    R = 287.05
    g = 9.80665

    altitude = clamp(altitude_m, 0.0, 11000.0)
    T = T0 - lapse * altitude

    if T <= 0:
        return float("nan"), float("nan")

    pressure = P0 * (T / T0) ** (g / (R * lapse))
    speed_of_sound = math.sqrt(gamma * R * T)

    return speed_of_sound, pressure


def flutter_velocity_mps(config: RocketConfig, fin: FinDesign) -> float:
    """
    Thin-panel flutter screening estimate.

    Formula form:
        Vf = a * sqrt( G * (t/c)^3 / (1.337 * AR^3 * P * ((lambda + 1)/2)) )

    Where:
        a = speed of sound
        G = shear modulus
        t/c = thickness-to-chord ratio
        AR = span^2 / area
        P = static pressure
        lambda = tip chord / root chord

    Use this as a screening estimate, not a final structural guarantee.
    """
    if fin.material not in MATERIALS:
        return float("nan")

    material = MATERIALS[fin.material]
    shear_modulus = material["shear_modulus_pa"]

    speed_of_sound, pressure = air_properties_simple(config.altitude_m)

    area_m2 = fin_area_mm2(fin) / 1_000_000.0
    span_m = mm_to_m(fin.span_mm)
    root_m = mm_to_m(fin.root_chord_mm)
    tip_m = mm_to_m(fin.tip_chord_mm)
    thickness_m = mm_to_m(fin.thickness_mm)

    values = [
        speed_of_sound,
        pressure,
        area_m2,
        span_m,
        root_m,
        tip_m,
        thickness_m,
    ]

    if any((not math.isfinite(v)) or v <= 0 for v in values):
        return float("nan")

    avg_chord_m = area_m2 / span_m
    taper_ratio = tip_m / root_m
    aspect_ratio = span_m**2 / area_m2
    thickness_ratio = thickness_m / avg_chord_m

    numerator = shear_modulus * thickness_ratio**3
    denominator = 1.337 * aspect_ratio**3 * pressure * ((taper_ratio + 1.0) / 2.0)

    if numerator <= 0 or denominator <= 0:
        return float("nan")

    return speed_of_sound * math.sqrt(numerator / denominator)


def estimate_fin_mass_g(config: RocketConfig, fin: FinDesign) -> float:
    if fin.material not in MATERIALS:
        return float("nan")

    density = MATERIALS[fin.material]["density_kg_m3"]
    single_fin_area_m2 = fin_area_mm2(fin) / 1_000_000.0
    thickness_m = mm_to_m(fin.thickness_mm)

    if single_fin_area_m2 <= 0 or thickness_m <= 0:
        return float("nan")

    single_fin_volume_m3 = single_fin_area_m2 * thickness_m
    all_fins_mass_kg = single_fin_volume_m3 * density * config.fin_count

    return all_fins_mass_kg * 1000.0


def loaded_cg_with_fins_mm(
    bare_mass_g: float,
    bare_cg_mm: float,
    fin_mass_g_value: float,
    fin_centroid_x_value: float,
) -> float:
    loaded_mass = bare_mass_g + fin_mass_g_value

    if loaded_mass <= 0:
        return float("nan")

    if not all(math.isfinite(v) for v in [bare_mass_g, bare_cg_mm, fin_mass_g_value, fin_centroid_x_value]):
        return float("nan")

    return (bare_mass_g * bare_cg_mm + fin_mass_g_value * fin_centroid_x_value) / loaded_mass


def score_design(
    config: RocketConfig,
    fin: FinDesign,
    loaded_mass_g_value: float,
    static_margin: float,
    flutter_sf: float,
) -> float:
    """
    Lower score is better.

    Penalizes:
    - being far from target margin
    - large fin area
    - excess mass
    - weak flutter safety
    - over-stability
    """
    if not all(math.isfinite(v) for v in [loaded_mass_g_value, static_margin, flutter_sf]):
        return float("nan")

    body_reference_area = max(config.body_diameter_mm * config.body_length_mm, 1.0)
    area_ratio = fin_area_mm2(fin) / body_reference_area

    margin_error = abs(static_margin - config.target_margin_calibers)
    over_stable_penalty = max(0.0, static_margin - config.max_margin_calibers) * 40.0
    low_flutter_penalty = max(0.0, config.min_flutter_safety_factor - flutter_sf) * 120.0

    mass_target = F15_8_MOTOR["preferred_beginner_loaded_mass_g"]
    mass_penalty = max(0.0, loaded_mass_g_value - mass_target) / 10.0

    return (
        margin_error * 30.0
        + area_ratio * 15.0
        + over_stable_penalty
        + low_flutter_penalty
        + mass_penalty
    )


def evaluate_design(config: RocketConfig, mass_table: pd.DataFrame, fin: FinDesign) -> EvaluationResult:
    table = clean_mass_table(mass_table)

    bare_mass = total_mass_g(table)
    bare_cg = center_of_gravity_mm(table)
    fin_mass = estimate_fin_mass_g(config, fin)
    fin_cx = fin_centroid_x_mm(config, fin)
    loaded_mass = bare_mass + fin_mass if math.isfinite(fin_mass) else float("nan")
    loaded_cg = loaded_cg_with_fins_mm(bare_mass, bare_cg, fin_mass, fin_cx)

    cp = total_cp_mm(config, fin)

    if math.isfinite(cp) and math.isfinite(loaded_cg) and config.body_diameter_mm > 0:
        static_margin = (cp - loaded_cg) / config.body_diameter_mm
    else:
        static_margin = float("nan")

    flutter_v = flutter_velocity_mps(config, fin)

    if math.isfinite(flutter_v) and config.max_speed_mps > 0:
        flutter_sf = flutter_v / config.max_speed_mps
    else:
        flutter_sf = float("nan")

    area = fin_area_mm2(fin)
    score = score_design(config, fin, loaded_mass, static_margin, flutter_sf)

    warnings: List[str] = []

    if bare_mass <= 0:
        warnings.append("Mass table total is zero. CG cannot be trusted.")

    if not math.isfinite(loaded_cg):
        warnings.append("Loaded CG calculation failed. Check mass table and fin geometry.")

    if not math.isfinite(cp):
        warnings.append("CP calculation failed. Check fin geometry.")

    if math.isfinite(static_margin):
        if static_margin < config.min_margin_calibers:
            warnings.append("Static margin is below your minimum. This design may be unstable.")
        elif static_margin > config.max_margin_calibers:
            warnings.append("Static margin is above your maximum. This may weathercock badly and add drag.")
    else:
        warnings.append("Static margin is invalid.")

    if math.isfinite(flutter_sf):
        if flutter_sf < config.min_flutter_safety_factor:
            warnings.append("Flutter safety factor is below your minimum.")
    else:
        warnings.append("Flutter calculation failed. Check material, thickness and geometry.")

    if fin.tip_chord_mm > fin.root_chord_mm:
        warnings.append("Tip chord is larger than root chord. This app assumes normal swept trapezoid fins.")

    if config.fin_leading_edge_x_mm + fin.root_chord_mm > config.body_length_mm:
        warnings.append("Fin root chord extends beyond the body length.")

    if fin.span_mm > 1.5 * config.body_diameter_mm:
        warnings.append("Fin span is large relative to body diameter. This may add drag and clearance issues.")

    if fin.material == "Balsa, edge grain" and config.max_speed_mps >= 120:
        warnings.append("Balsa is probably the wrong fin material for a faster F15-8 design.")

    if math.isfinite(loaded_mass):
        if loaded_mass > F15_8_MOTOR["listed_max_lift_weight_g"]:
            warnings.append("Loaded mass is above the listed F15-8 max lift weight of 425 g.")
        elif loaded_mass > F15_8_MOTOR["preferred_beginner_loaded_mass_g"]:
            warnings.append("Loaded mass is under 425 g but above the preferred 375 g beginner target.")

    return EvaluationResult(
        bare_mass_g=bare_mass,
        fin_mass_g=fin_mass,
        loaded_mass_g=loaded_mass,
        bare_cg_mm=bare_cg,
        loaded_cg_mm=loaded_cg,
        cp_mm=cp,
        static_margin_calibers=static_margin,
        flutter_velocity_mps=flutter_v,
        flutter_safety_factor=flutter_sf,
        fin_area_mm2=area,
        fin_centroid_x_mm=fin_cx,
        score=score,
        warnings=warnings,
    )



def generate_range(start: float, end: float, steps: int) -> np.ndarray:
    steps = max(1, int(steps))

    if steps == 1:
        return np.array([start], dtype=float)

    return np.linspace(start, end, steps)


def optimize_fins(
    config: RocketConfig,
    mass_table: pd.DataFrame,
    enabled_materials: List[str],
    root_min_d: float,
    root_max_d: float,
    root_steps: int,
    tip_min_d: float,
    tip_max_d: float,
    tip_steps: int,
    span_min_d: float,
    span_max_d: float,
    span_steps: int,
    sweep_min_d: float,
    sweep_max_d: float,
    sweep_steps: int,
    thickness_min_mm: float,
    thickness_max_mm: float,
    thickness_steps: int,
    max_results: int,
) -> pd.DataFrame:
    D = config.body_diameter_mm

    root_values = generate_range(root_min_d * D, root_max_d * D, root_steps)
    tip_values = generate_range(tip_min_d * D, tip_max_d * D, tip_steps)
    span_values = generate_range(span_min_d * D, span_max_d * D, span_steps)
    sweep_values = generate_range(sweep_min_d * D, sweep_max_d * D, sweep_steps)
    thickness_values = generate_range(thickness_min_mm, thickness_max_mm, thickness_steps)

    rows: List[Dict[str, Any]] = []

    for material in enabled_materials:
        if material not in MATERIALS:
            continue

        for root in root_values:
            for tip in tip_values:
                if tip > root:
                    continue

                for span in span_values:
                    for sweep in sweep_values:
                        for thickness in thickness_values:
                            fin = FinDesign(
                                root_chord_mm=float(root),
                                tip_chord_mm=float(tip),
                                span_mm=float(span),
                                sweep_mm=float(sweep),
                                thickness_mm=float(thickness),
                                material=material,
                            )

                            ev = evaluate_design(config, mass_table, fin)

                            if not math.isfinite(ev.score):
                                continue

                            if not math.isfinite(ev.static_margin_calibers):
                                continue

                            if not math.isfinite(ev.flutter_safety_factor):
                                continue

                            if ev.static_margin_calibers < config.min_margin_calibers:
                                continue

                            if ev.static_margin_calibers > config.max_margin_calibers:
                                continue

                            if ev.flutter_safety_factor < config.min_flutter_safety_factor:
                                continue

                            if config.fin_leading_edge_x_mm + fin.root_chord_mm > config.body_length_mm:
                                continue

                            rows.append(
                                {
                                    "Score": ev.score,
                                    "Material": material,
                                    "Root_chord_mm": fin.root_chord_mm,
                                    "Tip_chord_mm": fin.tip_chord_mm,
                                    "Span_mm": fin.span_mm,
                                    "Sweep_mm": fin.sweep_mm,
                                    "Thickness_mm": fin.thickness_mm,
                                    "Loaded_mass_g": ev.loaded_mass_g,
                                    "Bare_mass_g": ev.bare_mass_g,
                                    "Fin_mass_g": ev.fin_mass_g,
                                    "Loaded_CG_mm": ev.loaded_cg_mm,
                                    "CP_mm": ev.cp_mm,
                                    "Static_margin_calibers": ev.static_margin_calibers,
                                    "Flutter_velocity_mps": ev.flutter_velocity_mps,
                                    "Flutter_safety_factor": ev.flutter_safety_factor,
                                    "Fin_area_mm2": ev.fin_area_mm2,
                                }
                            )

    if not rows:
        return pd.DataFrame()

    result = pd.DataFrame(rows)
    result = result.sort_values("Score", ascending=True).head(max_results)
    return result.reset_index(drop=True)


def make_fin_plot(fin: FinDesign) -> go.Figure:
    x = [
        0.0,
        fin.root_chord_mm,
        fin.sweep_mm + fin.tip_chord_mm,
        fin.sweep_mm,
        0.0,
    ]
    y = [
        0.0,
        0.0,
        fin.span_mm,
        fin.span_mm,
        0.0,
    ]

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            fill="toself",
            mode="lines+markers",
            name="Fin outline",
            hovertemplate="x=%{x:.2f} mm<br>y=%{y:.2f} mm<extra></extra>",
        )
    )

    fig.add_annotation(
        x=fin.root_chord_mm / 2.0,
        y=-0.12 * max(fin.span_mm, 1.0),
        text=f"Root: {fin.root_chord_mm:.3f} mm",
        showarrow=False,
    )

    fig.add_annotation(
        x=fin.sweep_mm + fin.tip_chord_mm / 2.0,
        y=fin.span_mm + 0.12 * max(fin.span_mm, 1.0),
        text=f"Tip: {fin.tip_chord_mm:.3f} mm",
        showarrow=False,
    )

    fig.add_annotation(
        x=max(x) + 0.12 * max(max(x), 1.0),
        y=fin.span_mm / 2.0,
        text=f"Span: {fin.span_mm:.3f} mm",
        showarrow=False,
    )

    fig.update_layout(
        title="Fin geometry",
        xaxis_title="Chord direction (mm)",
        yaxis_title="Span direction (mm)",
        height=430,
        margin=dict(l=20, r=20, t=55, b=40),
        showlegend=False,
    )

    fig.update_yaxes(scaleanchor="x", scaleratio=1)

    return fig


def make_rocket_plot(config: RocketConfig, fin: FinDesign, ev: EvaluationResult) -> go.Figure:
    fig = go.Figure()

    L = config.body_length_mm
    nose = config.nose_length_mm

    body_top = 0.18
    body_bottom = -0.18

    # Nose cone, visual only
    nose_x = [
        0.0,
        nose * 0.25,
        nose * 0.75,
        nose,
        nose * 0.75,
        nose * 0.25,
        0.0,
    ]
    nose_y = [
        0.0,
        body_top * 1.15,
        body_top,
        body_top,
        body_bottom,
        body_bottom * 1.15,
        0.0,
    ]

    fig.add_trace(
        go.Scatter(
            x=nose_x,
            y=nose_y,
            fill="toself",
            mode="lines",
            line=dict(width=2),
            name="Ogive nose visual",
            hoverinfo="skip",
        )
    )

    # Body tube
    fig.add_trace(
        go.Scatter(
            x=[nose, L, L, nose, nose],
            y=[body_top, body_top, body_bottom, body_bottom, body_top],
            fill="toself",
            mode="lines",
            line=dict(width=2),
            name="Body tube",
            hoverinfo="skip",
        )
    )

    # Fin, scaled visually relative to diameter
    span_visual = (fin.span_mm / max(config.body_diameter_mm, 1.0)) * 0.26
    fin_x0 = config.fin_leading_edge_x_mm

    fin_x = [
        fin_x0,
        fin_x0 + fin.root_chord_mm,
        fin_x0 + fin.sweep_mm + fin.tip_chord_mm,
        fin_x0 + fin.sweep_mm,
        fin_x0,
    ]

    fin_y_bottom = [
        body_bottom,
        body_bottom,
        body_bottom - span_visual,
        body_bottom - span_visual,
        body_bottom,
    ]

    fin_y_top = [
        body_top,
        body_top,
        body_top + span_visual,
        body_top + span_visual,
        body_top,
    ]

    fig.add_trace(
        go.Scatter(
            x=fin_x,
            y=fin_y_bottom,
            fill="toself",
            mode="lines",
            line=dict(width=2),
            name="Visible fin",
            hoverinfo="skip",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=fin_x,
            y=fin_y_top,
            fill="toself",
            mode="lines",
            line=dict(width=1),
            opacity=0.35,
            name="Opposite fin",
            hoverinfo="skip",
        )
    )

    # Centerline
    fig.add_trace(
        go.Scatter(
            x=[0, L],
            y=[0, 0],
            mode="lines",
            line=dict(width=1, dash="dash"),
            name="Centerline",
            hoverinfo="skip",
        )
    )

    if math.isfinite(ev.loaded_cg_mm):
        fig.add_vline(
            x=ev.loaded_cg_mm,
            line_width=3,
            line_dash="solid",
            annotation_text=f"Loaded CG {ev.loaded_cg_mm:.1f} mm",
            annotation_position="top",
        )

    if math.isfinite(ev.cp_mm):
        fig.add_vline(
            x=ev.cp_mm,
            line_width=3,
            line_dash="dot",
            annotation_text=f"CP {ev.cp_mm:.1f} mm",
            annotation_position="bottom",
        )

    fig.update_layout(
        title="Rocket side view with loaded CG and CP",
        xaxis_title="Distance from nose tip (mm)",
        yaxis=dict(
            title="Scaled visual height",
            showticklabels=False,
            zeroline=False,
        ),
        height=430,
        margin=dict(l=20, r=20, t=55, b=40),
        showlegend=True,
    )

    fig.update_xaxes(range=[-0.04 * L, 1.08 * L])
    fig.update_yaxes(scaleanchor="x", scaleratio=1)

    return fig



if "mass_table" not in st.session_state:
    st.session_state.mass_table = DEFAULT_MASS_TABLE.copy()

if "optimizer_results" not in st.session_state:
    st.session_state.optimizer_results = pd.DataFrame()



st.markdown(
    """
    <style>
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
    }
    .warning-box {
        padding: 1rem;
        border-radius: 1rem;
        background: #fffbeb;
        border: 1px solid #fde68a;
        color: #78350f;
    }
    .good-box {
        padding: 1rem;
        border-radius: 1rem;
        background: #ecfdf5;
        border: 1px solid #a7f3d0;
        color: #064e3b;
    }
    .bad-box {
        padding: 1rem;
        border-radius: 1rem;
        background: #fef2f2;
        border: 1px solid #fecaca;
        color: #7f1d1d;
    }
    </style>
    """,
    unsafe_allow_html=True,
)



st.title("Rocket Fin Optimizer")
st.caption(
    "Professional-style beginner-friendly design tool for a legal single-stage F15-8 model rocket with recovery."
)

st.markdown(
    """
    ### What this app does in plain English

    A stable rocket needs the **center of gravity** ahead of the **center of pressure**.

    - **CG:** where the rocket balances based on mass.
    - **CP:** where the airflow effectively pushes on the rocket.
    - **Static margin:** how far CP is behind CG, measured in body diameters.
    - **Flutter:** dangerous fin vibration at speed.

    Your job is to enter real dimensions and real measured masses. The app checks whether the design is stable, whether the fins are stiff enough and whether the loaded mass is reasonable for an Estes F15-8.
    """
)

with st.expander("Important safety and accuracy limits", expanded=True):
    st.markdown(
        """
        This app is a **design estimator**, not a launch approval.

        Use it for:
        - comparing airframe sizes
        - choosing fin geometry
        - checking CG/CP relationship
        - screening flutter risk
        - keeping the rocket under the F15-8 mass limit

        You still need:
        - OpenRocket or RockSim validation
        - real measured masses after building
        - legal launch location
        - safe recovery system
        - proper launch equipment
        - adult/club supervision if required

        The app does not check:
        - apogee timing
        - parachute deployment velocity
        - wind drift
        - launch rod/rail exit velocity
        - real motor thrust curve behavior
        """
    )




with st.sidebar:
    st.header("Rocket setup")
    st.caption("These are the main dimensions. Distances are in millimetres.")

    with st.expander("F15-8 motor reference", expanded=False):
        st.write(
            {
                "Motor": F15_8_MOTOR["name"],
                "Diameter_mm": F15_8_MOTOR["diameter_mm"],
                "Length_mm": F15_8_MOTOR["length_mm"],
                "Loaded_mass_g": F15_8_MOTOR["loaded_mass_g"],
                "Total_impulse_Ns": F15_8_MOTOR["total_impulse_ns"],
                "Average_thrust_N": F15_8_MOTOR["average_thrust_n"],
                "Burn_time_s": F15_8_MOTOR["burn_time_s"],
                "Delay_s": F15_8_MOTOR["delay_s"],
                "Listed_max_lift_weight_g": F15_8_MOTOR["listed_max_lift_weight_g"],
            }
        )

    if st.button("Reset mass table to F15-8 professional preset"):
        st.session_state.mass_table = DEFAULT_MASS_TABLE.copy()
        st.session_state.optimizer_results = pd.DataFrame()
        st.rerun()

    body_diameter_mm = st.number_input(
        "Body diameter (mm)",
        min_value=5.000,
        value=57.400,
        step=0.001,
        format="%.3f",
        help="For the 54 mm thick-wall tube discussed earlier, OD is about 57.400 mm.",
    )

    body_length_mm = st.number_input(
        "Body length (mm)",
        min_value=50.000,
        value=700.000,
        step=0.001,
        format="%.3f",
        help="Total rocket length from nose tip to tail.",
    )

    nose_length_mm = st.number_input(
        "Nose length (mm)",
        min_value=10.000,
        value=160.000,
        step=0.001,
        format="%.3f",
        help="Length of the ogive nose cone, not including the body tube.",
    )

    nose_type = st.selectbox(
        "Nose type",
        ["Ogive", "Conical", "Elliptical"],
        index=0,
        help="Use Ogive for this design.",
    )

    fin_count = st.selectbox(
        "Fin count",
        [3, 4],
        index=0,
        help="Three fins usually gives less drag than four.",
    )

    fin_leading_edge_x_mm = st.number_input(
        "Fin leading edge X from nose (mm)",
        min_value=0.000,
        value=585.000,
        step=0.001,
        format="%.3f",
        help="Where the front of the fin root begins, measured from the nose tip.",
    )

    st.divider()
    st.header("Flight assumptions")

    max_speed_mps = st.number_input(
        "Expected max speed (m/s)",
        min_value=10.000,
        value=140.000,
        step=0.001,
        format="%.3f",
        help="Used for flutter screening. Use OpenRocket later for a better number.",
    )

    altitude_m = st.number_input(
        "Flutter check altitude (m)",
        min_value=0.000,
        value=300.000,
        step=0.001,
        format="%.3f",
        help="Air pressure changes with altitude. 300 m is a reasonable default for screening.",
    )

    st.divider()
    st.header("Design targets")

    target_margin_calibers = st.slider(
        "Target static margin",
        min_value=0.500,
        max_value=3.500,
        value=1.500,
        step=0.001,
        help="The optimizer tries to get close to this.",
    )

    min_margin_calibers = st.slider(
        "Minimum accepted margin",
        min_value=0.300,
        max_value=2.500,
        value=1.200,
        step=0.001,
        help="Designs below this are rejected.",
    )

    max_margin_calibers = st.slider(
        "Maximum accepted margin",
        min_value=1.000,
        max_value=5.000,
        value=2.200,
        step=0.001,
        help="Designs above this are rejected to avoid oversized fins and weathercocking.",
    )

    min_flutter_safety_factor = st.slider(
        "Minimum flutter safety factor",
        min_value=1.000,
        max_value=4.000,
        value=1.700,
        step=0.001,
        help="1.700 means estimated flutter speed should be at least 1.7 times expected max speed.",
    )

    config = RocketConfig(
        body_diameter_mm=body_diameter_mm,
        body_length_mm=body_length_mm,
        nose_length_mm=nose_length_mm,
        nose_type=nose_type,
        fin_count=int(fin_count),
        fin_leading_edge_x_mm=fin_leading_edge_x_mm,
        max_speed_mps=max_speed_mps,
        altitude_m=altitude_m,
        target_margin_calibers=target_margin_calibers,
        min_margin_calibers=min_margin_calibers,
        max_margin_calibers=max_margin_calibers,
        min_flutter_safety_factor=min_flutter_safety_factor,
    )




left_col, right_col = st.columns([0.95, 1.35], gap="large")



with left_col:
    st.subheader("1. Mass distribution")
    st.caption("Every X location is measured from the nose tip. Use loaded motor mass for launch CG.")

    st.info(
        "This table accepts decimals. Use measured masses when you have them. "
        "The app calculates bare CG from this table, then adds estimated fin mass to calculate loaded CG."
    )

    with st.expander("How to fill this table", expanded=False):
        st.markdown(
            """
            **Component:** part name, like nose cone, motor mount or recovery system.

            **Mass_g:** mass in grams.

            **X_from_nose_mm:** location of that part's center of mass from the nose tip.

            Example: if the rocket is 700 mm long and the F15-8 motor is mounted flush at the rear:

            `motor center = 700 - 114/2 = 643 mm`
            """
        )

    st.session_state.mass_table = edit_dataframe(st.session_state.mass_table)
    cleaned_mass_table = clean_mass_table(st.session_state.mass_table)

    st.divider()
    st.subheader("2. Manual fin design")
    st.caption("Start here, then run the optimizer.")

    st.info(
        "For this F15-8 build, start with 2 mm G10 fiberglass fins. "
        "The optimizer can search G10 and birch plywood, but G10 is the more professional default."
    )

    with st.expander("Fin vocabulary", expanded=False):
        st.markdown(
            """
            - **Root chord:** length of the fin attached to the body.
            - **Tip chord:** length of the outer edge.
            - **Span:** how far the fin sticks out from the body.
            - **Sweep:** how far the outer/front part shifts backward.
            - **Thickness:** material thickness.
            - **Material:** affects stiffness and mass.
            """
        )

    fin_left, fin_right = st.columns(2)

    with fin_left:
        root_chord_mm = st.number_input(
            "Root chord (mm)",
            min_value=1.000,
            value=80.000,
            step=0.001,
            format="%.3f",
        )

        span_mm = st.number_input(
            "Span (mm)",
            min_value=1.000,
            value=45.000,
            step=0.001,
            format="%.3f",
        )

        thickness_mm = st.number_input(
            "Thickness (mm)",
            min_value=0.500,
            value=2.000,
            step=0.001,
            format="%.3f",
        )

    with fin_right:
        tip_chord_mm = st.number_input(
            "Tip chord (mm)",
            min_value=1.000,
            value=30.000,
            step=0.001,
            format="%.3f",
        )

        sweep_mm = st.number_input(
            "Sweep (mm)",
            min_value=0.000,
            value=25.000,
            step=0.001,
            format="%.3f",
        )

        material = st.selectbox(
            "Material",
            list(MATERIALS.keys()),
            index=list(MATERIALS.keys()).index("G10 fiberglass"),
        )

    manual_fin = FinDesign(
        root_chord_mm=root_chord_mm,
        tip_chord_mm=tip_chord_mm,
        span_mm=span_mm,
        sweep_mm=sweep_mm,
        thickness_mm=thickness_mm,
        material=material,
    )

    manual_eval = evaluate_design(config, cleaned_mass_table, manual_fin)
    show_plotly(make_fin_plot(manual_fin))



with right_col:
    st.subheader("3. Current design evaluation")
    st.caption("This checks the manual fin design using the current rocket and mass inputs.")

    with st.expander("How to read this section", expanded=False):
        st.markdown(
            """
            **Loaded mass:** mass table plus estimated fin mass. This is what matters for F15-8 limits.

            **Loaded CG:** CG after adding the fin mass.

            **CP:** estimated aerodynamic center.

            **Static margin:** `(CP - loaded CG) / body diameter`.

            **Flutter SF:** estimated flutter velocity divided by expected max speed.
            """
        )

    metric_1, metric_2, metric_3, metric_4 = st.columns(4)
    metric_5, metric_6, metric_7, metric_8 = st.columns(4)

    with metric_1:
        st.metric("Loaded mass", f"{fmt(manual_eval.loaded_mass_g, 3)} g")
        st.caption("Mass table + fins")

    with metric_2:
        st.metric("Bare mass table", f"{fmt(manual_eval.bare_mass_g, 3)} g")

    with metric_3:
        st.metric("Loaded CG", f"{fmt(manual_eval.loaded_cg_mm, 3)} mm")

    with metric_4:
        st.metric("Static margin", f"{fmt(manual_eval.static_margin_calibers, 3)} cal")

    with metric_5:
        st.metric("CP", f"{fmt(manual_eval.cp_mm, 3)} mm")

    with metric_6:
        st.metric("Flutter velocity", f"{fmt(manual_eval.flutter_velocity_mps, 3)} m/s")

    with metric_7:
        st.metric("Flutter SF", f"{fmt(manual_eval.flutter_safety_factor, 3)}x")

    with metric_8:
        st.metric("Fin mass", f"{fmt(manual_eval.fin_mass_g, 3)} g")

    if math.isfinite(manual_eval.loaded_mass_g):
        if manual_eval.loaded_mass_g <= F15_8_MOTOR["preferred_beginner_loaded_mass_g"]:
            st.markdown(
                f"""
                <div class='good-box'>
                <b>F15-8 mass check:</b> {manual_eval.loaded_mass_g:.3f} g is within the preferred target of
                {F15_8_MOTOR["preferred_beginner_loaded_mass_g"]:.0f} g or less.
                </div>
                """,
                unsafe_allow_html=True,
            )
        elif manual_eval.loaded_mass_g <= F15_8_MOTOR["listed_max_lift_weight_g"]:
            st.markdown(
                f"""
                <div class='warning-box'>
                <b>F15-8 mass check:</b> {manual_eval.loaded_mass_g:.3f} g is under the listed max lift weight of
                {F15_8_MOTOR["listed_max_lift_weight_g"]:.0f} g, but it is getting heavy.
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
                <div class='bad-box'>
                <b>F15-8 mass check:</b> {manual_eval.loaded_mass_g:.3f} g is above the listed max lift weight of
                {F15_8_MOTOR["listed_max_lift_weight_g"]:.0f} g. This is too heavy for this motor target.
                </div>
                """,
                unsafe_allow_html=True,
            )

    if manual_eval.warnings:
        st.markdown(
            "<div class='warning-box'><b>Warnings</b><br>"
            + "<br>".join(f"• {warning}" for warning in manual_eval.warnings)
            + "</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            "<div class='good-box'><b>Current manual design passes the selected filters.</b></div>",
            unsafe_allow_html=True,
        )

    show_plotly(make_rocket_plot(config, manual_fin, manual_eval))



st.divider()
st.subheader("4. Fin optimizer")
st.caption("This searches many fin shapes and ranks the designs that pass your stability, mass and flutter filters.")

st.info(
    "For a professional F15-8 build, choose the smallest G10 fin design that still gives stable flight and a strong flutter margin. "
    "Do not blindly choose giant fins because they add drag and can make the rocket weathercock."
)

with st.expander("How the optimizer ranges work", expanded=False):
    st.markdown(
        """
        The optimizer uses **body diameters** for root, tip, span and sweep ranges.

        Example:

        If body diameter is 57.400 mm and root max is 1.900 body diameters:

        `57.400 x 1.900 = 109.060 mm`

        More steps gives a finer search but makes the optimizer slower.
        """
    )

opt_col_1, opt_col_2, opt_col_3 = st.columns(3, gap="large")

with opt_col_1:
    st.markdown("**Chord ranges**")

    root_min_d = st.number_input("Root min (body diameters)", min_value=0.100, value=1.000, step=0.001, format="%.3f")
    root_max_d = st.number_input("Root max (body diameters)", min_value=0.100, value=1.900, step=0.001, format="%.3f")
    root_steps = st.slider("Root steps", min_value=2, max_value=30, value=10, step=1)

    tip_min_d = st.number_input("Tip min (body diameters)", min_value=0.100, value=0.250, step=0.001, format="%.3f")
    tip_max_d = st.number_input("Tip max (body diameters)", min_value=0.100, value=0.750, step=0.001, format="%.3f")
    tip_steps = st.slider("Tip steps", min_value=2, max_value=30, value=8, step=1)

with opt_col_2:
    st.markdown("**Span and sweep ranges**")

    span_min_d = st.number_input("Span min (body diameters)", min_value=0.100, value=0.600, step=0.001, format="%.3f")
    span_max_d = st.number_input("Span max (body diameters)", min_value=0.100, value=1.200, step=0.001, format="%.3f")
    span_steps = st.slider("Span steps", min_value=2, max_value=30, value=10, step=1)

    sweep_min_d = st.number_input("Sweep min (body diameters)", min_value=0.000, value=0.100, step=0.001, format="%.3f")
    sweep_max_d = st.number_input("Sweep max (body diameters)", min_value=0.000, value=0.800, step=0.001, format="%.3f")
    sweep_steps = st.slider("Sweep steps", min_value=2, max_value=30, value=8, step=1)

with opt_col_3:
    st.markdown("**Thickness and materials**")

    thickness_min_mm = st.number_input("Thickness min (mm)", min_value=0.500, value=2.000, step=0.001, format="%.3f")
    thickness_max_mm = st.number_input("Thickness max (mm)", min_value=0.500, value=3.000, step=0.001, format="%.3f")
    thickness_steps = st.slider("Thickness steps", min_value=2, max_value=30, value=5, step=1)

    enabled_materials = st.multiselect(
        "Materials to search",
        list(MATERIALS.keys()),
        default=["G10 fiberglass", "Birch plywood"],
    )

    max_results = st.slider("Max results shown", min_value=10, max_value=500, value=100, step=10)

candidate_count = (
    root_steps
    * tip_steps
    * span_steps
    * sweep_steps
    * thickness_steps
    * max(1, len(enabled_materials))
)

run_left, run_right = st.columns([0.25, 0.75])

with run_left:
    run_optimizer = st.button("Optimize fins", type="primary")

with run_right:
    st.info(f"Approximate candidate designs before filtering: {candidate_count:,}")

if run_optimizer:
    if not enabled_materials:
        st.error("Choose at least one material.")
    elif root_max_d < root_min_d:
        st.error("Root max must be greater than root min.")
    elif tip_max_d < tip_min_d:
        st.error("Tip max must be greater than tip min.")
    elif span_max_d < span_min_d:
        st.error("Span max must be greater than span min.")
    elif sweep_max_d < sweep_min_d:
        st.error("Sweep max must be greater than sweep min.")
    elif thickness_max_mm < thickness_min_mm:
        st.error("Thickness max must be greater than thickness min.")
    else:
        with st.spinner("Optimizing fin geometry..."):
            st.session_state.optimizer_results = optimize_fins(
                config=config,
                mass_table=cleaned_mass_table,
                enabled_materials=enabled_materials,
                root_min_d=root_min_d,
                root_max_d=root_max_d,
                root_steps=root_steps,
                tip_min_d=tip_min_d,
                tip_max_d=tip_max_d,
                tip_steps=tip_steps,
                span_min_d=span_min_d,
                span_max_d=span_max_d,
                span_steps=span_steps,
                sweep_min_d=sweep_min_d,
                sweep_max_d=sweep_max_d,
                sweep_steps=sweep_steps,
                thickness_min_mm=thickness_min_mm,
                thickness_max_mm=thickness_max_mm,
                thickness_steps=thickness_steps,
                max_results=max_results,
            )

results = st.session_state.optimizer_results

if results.empty:
    st.warning(
        "No optimizer results yet, or no designs passed your filters. "
        "If this happens after optimizing, loosen the margin or flutter filters first."
    )
else:
    st.success(f"Found {len(results)} acceptable designs.")

    display_results = results.copy()
    numeric_cols = display_results.select_dtypes(include=[np.number]).columns
    display_results[numeric_cols] = display_results[numeric_cols].round(3)

    show_dataframe(display_results, height=430)

    st.subheader("5. Inspect optimized design")

    selected_rank = st.slider(
        "Design rank",
        min_value=1,
        max_value=len(results),
        value=1,
        step=1,
    )

    selected_row = results.iloc[selected_rank - 1]

    selected_fin = FinDesign(
        root_chord_mm=float(selected_row["Root_chord_mm"]),
        tip_chord_mm=float(selected_row["Tip_chord_mm"]),
        span_mm=float(selected_row["Span_mm"]),
        sweep_mm=float(selected_row["Sweep_mm"]),
        thickness_mm=float(selected_row["Thickness_mm"]),
        material=str(selected_row["Material"]),
    )

    selected_eval = evaluate_design(config, cleaned_mass_table, selected_fin)

    inspect_left, inspect_right = st.columns(2, gap="large")

    with inspect_left:
        st.markdown("**Selected fin geometry**")
        st.write(
            {
                "Rank": selected_rank,
                "Material": selected_fin.material,
                "Root_chord_mm": round(selected_fin.root_chord_mm, 3),
                "Tip_chord_mm": round(selected_fin.tip_chord_mm, 3),
                "Span_mm": round(selected_fin.span_mm, 3),
                "Sweep_mm": round(selected_fin.sweep_mm, 3),
                "Thickness_mm": round(selected_fin.thickness_mm, 3),
            }
        )

        show_plotly(make_fin_plot(selected_fin))

    with inspect_right:
        st.markdown("**Selected design performance**")
        st.write(
            {
                "Score": round(selected_eval.score, 3),
                "Loaded_mass_g": round(selected_eval.loaded_mass_g, 3),
                "Bare_mass_g": round(selected_eval.bare_mass_g, 3),
                "Fin_mass_g": round(selected_eval.fin_mass_g, 3),
                "Loaded_CG_mm": round(selected_eval.loaded_cg_mm, 3),
                "CP_mm": round(selected_eval.cp_mm, 3),
                "Static_margin_calibers": round(selected_eval.static_margin_calibers, 3),
                "Flutter_velocity_mps": round(selected_eval.flutter_velocity_mps, 3),
                "Flutter_safety_factor": round(selected_eval.flutter_safety_factor, 3),
                "Fin_area_mm2": round(selected_eval.fin_area_mm2, 3),
            }
        )

        if selected_eval.loaded_mass_g <= F15_8_MOTOR["preferred_beginner_loaded_mass_g"]:
            st.success("Selected design is inside the preferred F15-8 mass target.")
        elif selected_eval.loaded_mass_g <= F15_8_MOTOR["listed_max_lift_weight_g"]:
            st.warning("Selected design is under the F15-8 max lift weight, but it is getting heavy.")
        else:
            st.error("Selected design is too heavy for the F15-8 target.")

        show_plotly(make_rocket_plot(config, selected_fin, selected_eval))

    export_payload = {
        "rocket_config": asdict(config),
        "f15_8_motor_reference": F15_8_MOTOR,
        "mass_table": cleaned_mass_table.to_dict(orient="records"),
        "selected_fin": asdict(selected_fin),
        "selected_evaluation": asdict(selected_eval),
        "top_results": results.head(25).to_dict(orient="records"),
    }

    export_json = json.dumps(export_payload, indent=2)

    dl_left, dl_right = st.columns(2)

    with dl_left:
        st.download_button(
            "Download selected design JSON",
            data=export_json,
            file_name="f15_8_selected_rocket_design.json",
            mime="application/json",
        )

    with dl_right:
        st.download_button(
            "Download optimizer results CSV",
            data=results.to_csv(index=False),
            file_name="f15_8_fin_optimizer_results.csv",
            mime="text/csv",
        )
