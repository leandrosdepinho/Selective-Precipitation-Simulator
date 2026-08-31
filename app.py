import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import bisect


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Selective Precipitation Simulator",
    page_icon="🧪",
    layout="wide"
)


# ============================================================
# PRECIPITANT DATABASE
# ============================================================

PRECIPITANT_DATABASE = {

    # ========================================================
    # OXALATE
    # ========================================================

    "Oxalate": {

        "pkas": [1.25, 4.27],

        "charge_anion": 2,

        "Metals": {

            "Ag+": {"ksp": 3.5e-11, "x": 2, "y": 1},
            "Ba2+": {"ksp": 1.6e-6, "x": 1, "y": 1},
            "Bi3+": {"ksp": 1.0e-24, "x": 2, "y": 3},
            "Ca2+": {"ksp": 2.3e-9, "x": 1, "y": 1},
            "Cd2+": {"ksp": 1.4e-8, "x": 1, "y": 1},
            "Ce3+": {"ksp": 3.0e-26, "x": 2, "y": 3},
            "Ce4+": {"ksp": 1.0e-29, "x": 1, "y": 2},
            "Co2+": {"ksp": 6.3e-8, "x": 1, "y": 1},
            "Cu2+": {"ksp": 4.4e-10, "x": 1, "y": 1},
            "Dy3+": {"ksp": 2.0e-26, "x": 2, "y": 3},
            "Er3+": {"ksp": 2.2e-26, "x": 2, "y": 3},
            "Eu3+": {"ksp": 1.8e-26, "x": 2, "y": 3},
            "Fe2+": {"ksp": 3.2e-7, "x": 1, "y": 1},
            "Ga3+": {"ksp": 1.0e-21, "x": 2, "y": 3},
            "Gd3+": {"ksp": 2.1e-26, "x": 2, "y": 3},
            "Hg2_2+": {"ksp": 1.8e-13, "x": 1, "y": 1},
            "Hg2+": {"ksp": 1.5e-7, "x": 1, "y": 1},
            "Ho3+": {"ksp": 2.4e-26, "x": 2, "y": 3},
            "In3+": {"ksp": 5.8e-24, "x": 2, "y": 3},
            "La3+": {"ksp": 2.0e-26, "x": 2, "y": 3},
            "Lu3+": {"ksp": 3.2e-26, "x": 2, "y": 3},
            "Mg2+": {"ksp": 8.6e-5, "x": 1, "y": 1},
            "Mn2+": {"ksp": 1.1e-15, "x": 1, "y": 1},
            "Nd3+": {"ksp": 3.2e-26, "x": 2, "y": 3},
            "Ni2+": {"ksp": 4.0e-10, "x": 1, "y": 1},
            "Pb2+": {"ksp": 8.5e-9, "x": 1, "y": 1},
            "Pr3+": {"ksp": 2.8e-26, "x": 2, "y": 3},
            "Sc3+": {"ksp": 1.0e-25, "x": 2, "y": 3},
            "Sm3+": {"ksp": 2.5e-26, "x": 2, "y": 3},
            "Sn2+": {"ksp": 1.0e-9, "x": 1, "y": 1},
            "Sn4+": {"ksp": 1.0e-30, "x": 1, "y": 2},
            "Sr2+": {"ksp": 5.6e-8, "x": 1, "y": 1},
            "Tb3+": {"ksp": 2.2e-26, "x": 2, "y": 3},
            "Th4+": {"ksp": 5.0e-25, "x": 1, "y": 2},
            "Tl+": {"ksp": 2.0e-4, "x": 2, "y": 1},
            "Tl3+": {"ksp": 1.0e-28, "x": 2, "y": 3},
            "Tm3+": {"ksp": 2.6e-26, "x": 2, "y": 3},
            "UO2_2+": {"ksp": 1.3e-8, "x": 1, "y": 1},
            "U4+": {"ksp": 1.0e-22, "x": 1, "y": 2},
            "Y3+": {"ksp": 5.0e-26, "x": 2, "y": 3},
            "Yb3+": {"ksp": 3.0e-26, "x": 2, "y": 3},
            "Zn2+": {"ksp": 2.8e-8, "x": 1, "y": 1}
        }
    },


    # ========================================================
    # PHOSPHATE
    # ========================================================

    "Phosphate": {

        "pkas": [2.15, 7.20, 12.35],

        "charge_anion": 3,

        "Metals": {

            "Ag+": {"ksp": 1.4e-16, "x": 3, "y": 1},
            "Al3+": {"ksp": 9.8e-21, "x": 1, "y": 1},
            "Ba2+": {"ksp": 3.4e-23, "x": 3, "y": 2},
            "Be2+": {"ksp": 1.5e-39, "x": 3, "y": 2},
            "Bi3+": {"ksp": 1.3e-23, "x": 1, "y": 1},
            "Ca2+": {"ksp": 2.1e-33, "x": 3, "y": 2},
            "Cd2+": {"ksp": 2.5e-33, "x": 3, "y": 2},
            "Ce3+": {"ksp": 1.0e-23, "x": 1, "y": 1},
            "Ce4+": {"ksp": 1.0e-64, "x": 3, "y": 4},
            "Co2+": {"ksp": 2.0e-35, "x": 3, "y": 2},
            "Cr3+": {"ksp": 2.4e-23, "x": 1, "y": 1},
            "Cu2+": {"ksp": 1.4e-37, "x": 3, "y": 2},
            "Dy3+": {"ksp": 4.0e-25, "x": 1, "y": 1},
            "Er3+": {"ksp": 3.2e-25, "x": 1, "y": 1},
            "Eu3+": {"ksp": 2.5e-25, "x": 1, "y": 1},
            "Fe2+": {"ksp": 1.0e-36, "x": 3, "y": 2},
            "Fe3+": {"ksp": 1.3e-22, "x": 1, "y": 1},
            "Ga3+": {"ksp": 1.0e-21, "x": 1, "y": 1},
            "Gd3+": {"ksp": 3.0e-25, "x": 1, "y": 1},
            "Hf4+": {"ksp": 1.0e-62, "x": 3, "y": 4},
            "Hg2_2+": {"ksp": 1.0e-61, "x": 3, "y": 2},
            "Hg2+": {"ksp": 8.8e-33, "x": 3, "y": 2},
            "Ho3+": {"ksp": 3.5e-25, "x": 1, "y": 1},
            "In3+": {"ksp": 1.0e-22, "x": 1, "y": 1},
            "La3+": {"ksp": 3.7e-23, "x": 1, "y": 1},
            "Lu3+": {"ksp": 5.0e-25, "x": 1, "y": 1},
            "Mg2+": {"ksp": 1.0e-24, "x": 3, "y": 2},
            "Mn2+": {"ksp": 1.0e-22, "x": 3, "y": 2},
            "Nb5+": {"ksp": 1.0e-85, "x": 3, "y": 5},
            "Nd3+": {"ksp": 4.0e-25, "x": 1, "y": 1},
            "Ni2+": {"ksp": 5.0e-31, "x": 3, "y": 2},
            "Pb2+": {"ksp": 7.9e-44, "x": 3, "y": 2},
            "Pr3+": {"ksp": 3.5e-23, "x": 1, "y": 1},
            "Sc3+": {"ksp": 1.0e-24, "x": 1, "y": 1},
            "Sm3+": {"ksp": 2.8e-25, "x": 1, "y": 1},
            "Sn2+": {"ksp": 1.0e-27, "x": 3, "y": 2},
            "Sn4+": {"ksp": 1.0e-60, "x": 3, "y": 4},
            "Sr2+": {"ksp": 4.0e-28, "x": 3, "y": 2},
            "Ta5+": {"ksp": 1.0e-88, "x": 3, "y": 5},
            "Tb3+": {"ksp": 3.2e-25, "x": 1, "y": 1},
            "Th4+": {"ksp": 1.0e-78, "x": 3, "y": 4},
            "Ti4+": {"ksp": 1.0e-65, "x": 3, "y": 4},
            "Tl3+": {"ksp": 1.0e-35, "x": 1, "y": 1},
            "Tm3+": {"ksp": 4.2e-25, "x": 1, "y": 1},
            "U4+": {"ksp": 1.0e-80, "x": 3, "y": 4},
            "V3+": {"ksp": 1.0e-22, "x": 1, "y": 1},
            "Y3+": {"ksp": 1.0e-25, "x": 1, "y": 1},
            "Yb3+": {"ksp": 4.5e-25, "x": 1, "y": 1},
            "Zn2+": {"ksp": 9.0e-33, "x": 3, "y": 2},
            "Zr4+": {"ksp": 1.0e-62, "x": 3, "y": 4}
        }
    },


    # ========================================================
    # CARBONATE
    # ========================================================

    "Carbonate": {

        "pkas": [6.35, 10.33],

        "charge_anion": 2,

        "Metals": {

            "Ag+": {"ksp": 8.5e-12, "x": 2, "y": 1},
            "Ba2+": {"ksp": 2.6e-9, "x": 1, "y": 1},
            "Be2+": {"ksp": 2.0e-10, "x": 1, "y": 1},
            "Ca2+": {"ksp": 4.5e-9, "x": 1, "y": 1},
            "Cd2+": {"ksp": 1.0e-12, "x": 1, "y": 1},
            "Ce4+": {"ksp": 1.0e-36, "x": 1, "y": 2},
            "Co2+": {"ksp": 1.0e-10, "x": 1, "y": 1},
            "Cu2+": {"ksp": 1.4e-10, "x": 1, "y": 1},
            "Fe2+": {"ksp": 3.1e-11, "x": 1, "y": 1},
            "Hg2_2+": {"ksp": 3.6e-17, "x": 1, "y": 1},
            "Hg2+": {"ksp": 3.0e-13, "x": 1, "y": 1},
            "Mg2+": {"ksp": 3.5e-8, "x": 1, "y": 1},
            "Mn2+": {"ksp": 2.2e-11, "x": 1, "y": 1},
            "Ni2+": {"ksp": 1.3e-7, "x": 1, "y": 1},
            "Pb2+": {"ksp": 7.4e-14, "x": 1, "y": 1},
            "Sn2+": {"ksp": 1.0e-20, "x": 1, "y": 1},
            "Sn4+": {"ksp": 1.0e-38, "x": 1, "y": 2},
            "Sr2+": {"ksp": 5.6e-10, "x": 1, "y": 1},
            "Th4+": {"ksp": 1.0e-35, "x": 1, "y": 2},
            "Tl+": {"ksp": 4.3e-4, "x": 2, "y": 1},
            "U4+": {"ksp": 1.0e-36, "x": 1, "y": 2},
            "UO2_2+": {"ksp": 3.0e-14, "x": 1, "y": 1},

            # Vanadyl written explicitly with underscore
            # to distinguish the ionic charge.
            "VO_2+": {"ksp": 1.0e-18, "x": 1, "y": 1},

            "Zn2+": {"ksp": 1.2e-10, "x": 1, "y": 1}
        }
    }
}


# ============================================================
# BUILD MASTER METAL LIST
# ============================================================

all_metals = sorted(
    set(
        metal
        for precipitant in PRECIPITANT_DATABASE.values()
        for metal in precipitant["Metals"]
    )
)


# ============================================================
# TITLE
# ============================================================

st.title("🧪 Selective Precipitation Simulator")

st.write(
    "Thermodynamic screening of metal precipitation using "
    "different precipitants."
)


# ============================================================
# PRECIPITANT SELECTION
# ============================================================

st.subheader("1. Select Precipitant")

precipitant_name = st.selectbox(
    "Precipitant:",
    list(PRECIPITANT_DATABASE.keys())
)


precipitant_data = PRECIPITANT_DATABASE[
    precipitant_name
]

precipitant_pkas = precipitant_data["pkas"]


# ============================================================
# METAL SELECTION
# ============================================================

st.subheader("2. Select Metals")

st.caption(
    "Select the metals you want to include and enter their "
    "initial concentrations."
)


selected_metals = []


# Create three columns
metal_columns = st.columns(3)


for i, metal in enumerate(all_metals):

    with metal_columns[i % 3]:

        selected = st.checkbox(
            metal,
            key=f"select_{metal}"
        )

        if selected:

            concentration = st.number_input(
                f"Initial concentration of {metal} (M)",
                min_value=0.0,
                value=0.05,
                format="%.6g",
                key=f"concentration_{metal}"
            )

            selected_metals.append({

                "name": metal,

                "initial_conc": concentration,

                "database": precipitant_data["Metals"].get(
                    metal
                )

            })


# ============================================================
# SIMULATION SETTINGS
# ============================================================

st.subheader("3. Simulation Settings")

col1, col2 = st.columns(2)


# ------------------------------------------------------------
# Plot 1 settings
# ------------------------------------------------------------

with col1:

    st.markdown(
        "**Plot 1 — Variable pH / Fixed Precipitant**"
    )

    fixed_precipitant_conc = st.number_input(
        f"Fixed {precipitant_name} concentration (M)",
        min_value=0.0,
        value=0.05,
        format="%.6g"
    )

    ph_start, ph_end = st.slider(
        "pH range:",
        min_value=-2.0,
        max_value=14.0,
        value=(0.0, 11.0),
        step=0.5,
        format="%.1f"
    )


# ------------------------------------------------------------
# Plot 2 settings
# ------------------------------------------------------------

with col2:

    st.markdown(
        "**Plot 2 — Variable Precipitant / Fixed pH**"
    )

    fixed_ph_value = st.number_input(
        "Fixed pH:",
        min_value=-2.0,
        max_value=14.0,
        value=5.0,
        step=0.5,
        format="%.1f"
    )

    max_precipitant_addition = st.number_input(
        f"Maximum added {precipitant_name} concentration (M)",
        min_value=0.0,
        value=0.25,
        format="%.6g"
    )


# ============================================================
# SIMULATE BUTTON
# ============================================================

st.markdown("")

simulate = st.button(
    "▶ SIMULATE",
    type="primary",
    use_container_width=True
)


# ============================================================
# CHEMICAL FUNCTIONS
# ============================================================

def calculate_alpha_precipitant(ph_value, kas_list):
    """
    Calculates the alpha fraction of the completely
    deprotonated precipitant active species.
    """

    free_h = 10 ** (-ph_value)

    n_proton_sites = len(kas_list)

    h_terms = [
        free_h ** (n_proton_sites - i)
        for i in range(n_proton_sites + 1)
    ]

    ka_products = [1.0]

    cumulative_product = 1.0

    for ka in kas_list:

        cumulative_product *= ka

        ka_products.append(
            cumulative_product
        )

    denominator = sum(
        term * product
        for term, product in zip(
            h_terms,
            ka_products
        )
    )

    return (
        ka_products[-1]
        / denominator
    )


# ============================================================
# LIGAND MASS BALANCE
# ============================================================

def ligand_mass_balance_equation(
    total_free_ligand,
    total_added_ligand,
    alpha_fraction,
    metal_systems
):
    """
    Calculates the ligand mass balance residual.
    """

    active_free_ligand = max(
        1e-45,
        total_free_ligand * alpha_fraction
    )

    consumed_ligand = 0.0


    for metal in metal_systems:

        # ----------------------------------------------------
        # Metals without a database entry for the selected
        # precipitant do not precipitate.
        # ----------------------------------------------------

        if metal["database"] is None:
            continue


        ksp = metal["database"]["ksp"]
        x = metal["database"]["x"]
        y = metal["database"]["y"]


        equilibrium_metal = (

            ksp
            /
            (active_free_ligand ** y)

        ) ** (1.0 / x)


        precipitated_metal = (

            metal["initial_conc"]

            -

            min(
                metal["initial_conc"],
                equilibrium_metal
            )

        )


        consumed_ligand += (

            precipitated_metal
            *
            (y / x)

        )


    return (
        total_free_ligand
        +
        consumed_ligand
        -
        total_added_ligand
    )


# ============================================================
# LOG-SPACE WRAPPER FOR THE ROOT SOLVER
# ============================================================

# FIX (numerical): the equilibrium root "total_free_ligand" can sit
# anywhere from ~1e-1 M down to ~1e-20 M or lower, depending on how
# insoluble the competing solids are. scipy's bisect was previously
# called with a fixed ABSOLUTE tolerance (xtol=1e-15). That tolerance
# is fine when the root is of order 1e-1, but once the true root is
# several orders of magnitude below 1e-15 itself, bisect has no
# resolution left to distinguish it from numerical noise — it just
# stops as soon as the bracket shrinks below 1e-15, and the value
# reported inside that final bracket is essentially arbitrary.
#
# That noise gets amplified by the (1 / active_ligand ** y) term in
# the precipitation equation: for high-order stoichiometries (e.g.
# y=4 for U4+/Th4+ with phosphate), a ~15% relative error in the
# free-ligand concentration turns into a ~1.75x error in the computed
# equilibrium metal concentration (1.15**4 ≈ 1.75). Right at the
# transition pH/dosage — where a metal is neither fully dissolved nor
# fully precipitated — that's enough to make % precipitated jump
# erratically point to point (the pH 5-7.5 zig-zag).
#
# Fix: solve for log10(total_free_ligand) instead of the raw
# concentration. Bisecting in log-space gives a uniform RELATIVE
# resolution across every scale, so the same xtol behaves just as
# well whether the true root is 1e-1 or 1e-20 — the mismatch between
# solver tolerance and the ligand's concentration scale disappears.

LOG_FREE_LIGAND_FLOOR = -300.0  # log10 of an effectively-zero free ligand concentration
LOG_SOLVER_XTOL = 1e-12         # resolution in log10 units (i.e. ~1e-12 relative precision)


def ligand_mass_balance_equation_log(
    log_total_free_ligand,
    total_added_ligand,
    alpha_fraction,
    metal_systems
):
    """
    Same residual as ligand_mass_balance_equation, but parameterized
    by log10(total_free_ligand) so the root finder operates with
    uniform relative resolution regardless of the concentration scale.
    """

    total_free_ligand = 10 ** log_total_free_ligand

    return ligand_mass_balance_equation(
        total_free_ligand,
        total_added_ligand,
        alpha_fraction,
        metal_systems
    )


def solve_free_ligand(total_added_ligand, alpha_fraction, metal_systems):
    """
    Solves the ligand mass balance for total_free_ligand, bisecting
    in log10-space (see note above). Returns np.nan if the solver
    fails to bracket/converge, so failures are visible as gaps in
    the plots instead of being silently guessed in either direction.
    """

    if total_added_ligand <= 0:
        return 0.0

    # Quick check: is there enough ligand that nothing needs to
    # dissolve at all (residual already >= 0 at zero free ligand)?
    if ligand_mass_balance_equation(
        0,
        total_added_ligand,
        alpha_fraction,
        metal_systems
    ) > 0:
        return 0.0

    try:

        upper_bound = total_added_ligand + 0.1

        log_solution = bisect(

            ligand_mass_balance_equation_log,

            LOG_FREE_LIGAND_FLOOR,

            np.log10(upper_bound),

            args=(
                total_added_ligand,
                alpha_fraction,
                metal_systems
            ),

            xtol=LOG_SOLVER_XTOL

        )

        return 10 ** log_solution

    except Exception:

        # Solver failed to bracket/converge: report as NaN rather
        # than guessing a neutral fallback. Downstream code turns
        # this into a visible gap in the curves (see below).
        return np.nan


# ============================================================
# RUN SIMULATION
# ============================================================

if simulate:

    # --------------------------------------------------------
    # Validate selected metals
    # --------------------------------------------------------

    if len(selected_metals) == 0:

        st.warning(
            "Please select at least one metal."
        )

        st.stop()


    # --------------------------------------------------------
    # Validate concentrations
    # --------------------------------------------------------

    for metal in selected_metals:

        if metal["initial_conc"] <= 0:

            st.error(
                f"Please enter a concentration greater "
                f"than zero for {metal['name']}."
            )

            st.stop()


    # --------------------------------------------------------
    # MODE A
    # Variable pH / Fixed Precipitant Concentration
    # --------------------------------------------------------

    ph_range = np.linspace(
        ph_start,
        ph_end,
        1000
    )

    results_mode_a = []

    # Points where the equilibrium solver failed to converge.
    # Both modes use the SAME neutral (NaN) fallback so that a
    # solver failure is treated identically regardless of which
    # mode produced it.
    failed_points_a = []


    for ph in ph_range:

        alpha_A = calculate_alpha_precipitant(
            ph,
            [
                10 ** (-pka)
                for pka in precipitant_pkas
            ]
        )

        solved_free_ligand = solve_free_ligand(
            fixed_precipitant_conc,
            alpha_A,
            selected_metals
        )

        if np.isnan(solved_free_ligand):
            failed_points_a.append(ph)


        final_active_ligand = (
            solved_free_ligand
            * alpha_A
        )


        data_row = {
            "pH": ph
        }


        if np.isnan(final_active_ligand):

            # Solver failed at this pH: leave every metal as NaN
            # so matplotlib shows a gap rather than a guessed
            # value (same treatment as Mode B, see below).

            for metal in selected_metals:

                data_row[metal["name"]] = np.nan

            results_mode_a.append(data_row)

            continue


        for metal in selected_metals:

            # ------------------------------------------------
            # No insoluble compound in database:
            # precipitation remains 0%.
            # ------------------------------------------------

            if metal["database"] is None:

                data_row[
                    metal["name"]
                ] = 0.0

                continue


            ksp = metal["database"]["ksp"]
            x = metal["database"]["x"]
            y = metal["database"]["y"]


            if final_active_ligand > 1e-45:

                equilibrium_metal = (

                    ksp
                    /
                    (
                        final_active_ligand ** y
                    )

                ) ** (1.0 / x)

            else:

                equilibrium_metal = (
                    metal["initial_conc"]
                )


            real_soluble_metal = min(
                metal["initial_conc"],
                equilibrium_metal
            )


            precipitation_efficiency = (

                (
                    metal["initial_conc"]
                    -
                    real_soluble_metal
                )

                /

                metal["initial_conc"]

            ) * 100


            data_row[
                metal["name"]
            ] = np.clip(
                precipitation_efficiency,
                0,
                100
            )


        results_mode_a.append(
            data_row
        )


    df_mode_a = pd.DataFrame(
        results_mode_a
    )


    # --------------------------------------------------------
    # MODE B
    # Variable Precipitant Concentration / Fixed pH
    # --------------------------------------------------------

    concentration_range = np.linspace(
        0,
        max_precipitant_addition,
        1000
    )


    results_mode_b = []

    # Points where the equilibrium solver failed to converge
    # (same neutral fallback convention as Mode A above).
    failed_points_b = []


    alpha_A_fixed = calculate_alpha_precipitant(
        fixed_ph_value,
        [
            10 ** (-pka)
            for pka in precipitant_pkas
        ]
    )


    for total_conc in concentration_range:

        solved_free_ligand = solve_free_ligand(
            total_conc,
            alpha_A_fixed,
            selected_metals
        )

        if np.isnan(solved_free_ligand):
            failed_points_b.append(total_conc)


        final_active_ligand = (

            solved_free_ligand
            *
            alpha_A_fixed

        )


        data_row = {
            "added_concentration": total_conc
        }


        if np.isnan(final_active_ligand):

            for metal in selected_metals:

                data_row[metal["name"]] = np.nan

            results_mode_b.append(data_row)

            continue


        for metal in selected_metals:

            # ------------------------------------------------
            # No insoluble compound in database:
            # precipitation remains 0%.
            # ------------------------------------------------

            if metal["database"] is None:

                data_row[
                    metal["name"]
                ] = 0.0

                continue


            ksp = metal["database"]["ksp"]
            x = metal["database"]["x"]
            y = metal["database"]["y"]


            if final_active_ligand > 1e-45:

                equilibrium_metal = (

                    ksp
                    /
                    (
                        final_active_ligand ** y
                    )

                ) ** (1.0 / x)

            else:

                equilibrium_metal = (
                    metal["initial_conc"]
                )


            real_soluble_metal = min(
                metal["initial_conc"],
                equilibrium_metal
            )


            precipitation_efficiency = (

                (
                    metal["initial_conc"]
                    -
                    real_soluble_metal
                )

                /

                metal["initial_conc"]

            ) * 100


            data_row[
                metal["name"]
            ] = np.clip(
                precipitation_efficiency,
                0,
                100
            )


        results_mode_b.append(
            data_row
        )


    df_mode_b = pd.DataFrame(
        results_mode_b
    )


    # ========================================================
    # SOLVER FAILURE WARNING
    # ========================================================

    if failed_points_a or failed_points_b:

        st.warning(
            f"The equilibrium solver failed to converge at "
            f"{len(failed_points_a)} point(s) in Plot 1 and "
            f"{len(failed_points_b)} point(s) in Plot 2. "
            f"These points are shown as gaps in the curves "
            f"rather than estimated values."
        )


    # ========================================================
    # VISUALIZATION
    # ========================================================

    st.subheader("Simulation Results")


    # Create figure
    fig, (ax1, ax2) = plt.subplots(
        1,
        2,
        figsize=(15, 6)
    )


    # --------------------------------------------------------
    # Generate distinct colors
    # --------------------------------------------------------

    color_maps = [
        plt.cm.tab20,
        plt.cm.tab20b,
        plt.cm.tab20c
    ]

    colors = []

    for cmap in color_maps:

        colors.extend(
            cmap(
                np.linspace(
                    0,
                    1,
                    20
                )
            )
        )


    # --------------------------------------------------------
    # Plot 1
    # --------------------------------------------------------

    for i, metal in enumerate(selected_metals):

        ax1.plot(

            df_mode_a["pH"],

            df_mode_a[metal["name"]],

            label=metal["name"],

            linewidth=2.5,

            color=colors[
                i % len(colors)
            ]

        )


    ax1.set_title(

        f"Selective Precipitation vs pH\n"
        f"(Fixed {precipitant_name} = "
        f"{fixed_precipitant_conc:g} M)",

        fontsize=11,

        fontweight="bold"

    )


    ax1.set_xlabel(
        "pH",
        fontsize=11
    )


    ax1.set_ylabel(
        "% Precipitated",
        fontsize=11
    )


    ax1.set_xlim(
        ph_start,
        ph_end
    )


    ax1.set_ylim(
        -2,
        102
    )


    ax1.set_xticks(
        np.arange(
            np.ceil(ph_start * 2) / 2,
            ph_end + 0.25,
            0.5
        )
    )


    ax1.grid(
        True,
        linestyle="--",
        alpha=0.4
    )


    ax1.legend(
        loc="best"
    )


    # --------------------------------------------------------
    # Plot 2
    # --------------------------------------------------------

    for i, metal in enumerate(selected_metals):

        ax2.plot(

            df_mode_b["added_concentration"],

            df_mode_b[metal["name"]],

            label=metal["name"],

            linewidth=2.5,

            color=colors[
                i % len(colors)
            ]

        )


    ax2.set_title(

        f"Selective Precipitation vs Dosage\n"
        f"(Fixed pH = {fixed_ph_value:g})",

        fontsize=11,

        fontweight="bold"

    )


    ax2.set_xlabel(

        f"Total Added {precipitant_name} (mol/L)",

        fontsize=11

    )


    ax2.set_ylabel(
        "% Precipitated",
        fontsize=11
    )


    ax2.set_xlim(
        0,
        max_precipitant_addition
    )


    ax2.set_ylim(
        -2,
        102
    )


    # Automatic x-axis ticks
    if max_precipitant_addition > 0:

        tick_step = max(
            max_precipitant_addition / 5,
            0.000001
        )

        ax2.set_xticks(
            np.linspace(
                0,
                max_precipitant_addition,
                6
            )
        )


    ax2.grid(
        True,
        linestyle="--",
        alpha=0.4
    )


    ax2.legend(
        loc="best"
    )


    # --------------------------------------------------------
    # Overall title
    # --------------------------------------------------------

    fig.suptitle(

        f"Thermodynamic Screening: "
        f"Selective Precipitation with "
        f"{precipitant_name}",

        fontsize=14,

        fontweight="bold"

    )


    fig.tight_layout()


    st.pyplot(fig)


    # ========================================================
    # SELECTED METALS SUMMARY
    # ========================================================

    st.subheader("Selected Metals")


    summary_data = []


    for metal in selected_metals:

        database_entry = metal["database"]


        if database_entry is None:

            summary_data.append({

                "Metal": metal["name"],

                "Initial concentration (M)":
                    metal["initial_conc"],

                "Ksp":
                    "No insoluble compound",

                "x":
                    "—",

                "y":
                    "—",

                "Precipitation":
                    "0%"

            })

        else:

            summary_data.append({

                "Metal": metal["name"],

                "Initial concentration (M)":
                    metal["initial_conc"],

                "Ksp":
                    f"{database_entry['ksp']:.2e}",

                "x":
                    database_entry["x"],

                "y":
                    database_entry["y"],

                "Precipitation":
                    "Calculated"

            })


    summary = pd.DataFrame(
        summary_data
    )


    st.dataframe(
        summary,
        use_container_width=True,
        hide_index=True
    )
