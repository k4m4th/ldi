"""Write the supplement's settings tables from the fit settings each example saves.

Every fitting notebook writes its parameters to ``results/fit_settings*.json``. This
script reads those files and writes ``paper_tex/tables/settings.tex``: Supplementary
Table ``tab:settings`` (the LDI fits), which ``supplementary.tex`` inputs. ``paper_tex/latexmkrc`` runs it before every build, so the
tables follow the latest fits; it rewrites the file only when its content changes.

``EXAMPLES`` says which file belongs to which row and how the example is labelled in
the paper. Only experiments that save a settings file have a row. Standard library only.

    python publication/settings_table.py [--check]
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "paper_tex" / "tables" / "settings.tex"

# One row per fit, in table order: its name, the main-text figures it appears in (labels), its
# settings file, dimension, which fit in the file ("ldi" or "nf": the fold-pair file holds
# both), lengths in model units (no km), and ``defaults`` for keys a settings file saved before
# they were recorded may lack (the file wins when it has them). ``sub`` rows are indented under
# the row above.
EXAMPLES = [
    dict(name="Inclined fold pair", figs=["recumbent_comparison"], file="01_recumbent_fold/results/synthetic/fit_settings.json",
         d=2, fit="ldi", ensemble="01_recumbent_fold/results/synthetic/ensemble_members.csv"),
    dict(name="Cylindrical isocline", figs=["cylindrical_isocline"], file="07_cylindrical_isocline/results/fit_settings.json",
         d=3, fit="ldi", defaults=dict(integrator="shear", cylindrical=True, energy_normalised=True, init_std=0.0),
         ensemble_json="07_cylindrical_isocline/results/ensemble/ensemble.json"),
    dict(name="Single layer", figs=["single_layer", "single_layer_prior"],
         file="03_single_layer/results/fit_settings.json", d=2, fit="ldi", model_units=True),
    dict(name="Multilayer fold train", figs=["big_fold"], file="04_multiwavelength_fold/results/shear/fit_settings.json",
         d=2, fit="ldi", model_units=True),
    dict(name="Hartland Quay", figs=["hartland"], file="05_hartland_quay/results/fit_settings.json", d=3, fit="ldi"),
    dict(name="Claudius", figs=["claudius"], file="06_claudius/results/fit_settings_ldi_seis5x5b_w20.json", d=3, fit="ldi"),
]

def sci(x):
    """A rate as a×10^k, as the other tables write them."""
    x = float(x)
    k = math.floor(math.log10(abs(x)))
    m = round(x / 10 ** k, 2)
    if m == 10:
        m, k = 1, k + 1
    return f"$10^{{{k}}}$" if f"{m:g}" == "1" else f"${m:g}\\times10^{{{k}}}$"


def num(x):
    """A number as LaTeX: integers plain, powers of ten as 10^k, small or large as a×10^k."""
    x = float(x)
    if x == 0:
        return "0"
    if x == int(x) and abs(x) < 1e5:
        return f"{int(x)}"
    k = math.floor(math.log10(abs(x)))
    m = round(x / 10 ** k, 2)
    if m == 10:
        m, k = 1, k + 1
    if -2 <= k <= 2:
        return f"{x:g}"
    mant = f"{m:g}"
    return f"$10^{{{k}}}$" if mant == "1" else f"${mant}\\times10^{{{k}}}$"


def first(s, *keys, default=None):
    for k in keys:
        if k in s and s[k] is not None:
            return s[k]
    return default


def construction(s):
    integrator = s.get("integrator", "shear")
    n = int(float(s.get("n_steps", 0)))
    if integrator == "rk4":
        return f"RK4, $N={n}$"
    return "one-way sweep" if n == 0 else f"palindrome, $N={n}$"


def lengths(s, fit, model_units):
    lo, hi = first(s, f"lsr_{fit}", "lsr", "length_scale_range")
    g = lambda v: f"{float(f'{v:.3g}'):g}"           # three significant figures, never in e-notation
    if model_units:
        return f"$[{g(lo)},{g(hi)}]$$^{{\\dagger}}$"
    return f"$[{g(lo / 1000)},{g(hi / 1000)}]$"


def rates(s, fit):
    """(Adam rate, amplitude rate): the second is "--" unless energy-normalised amplitudes start at zero."""
    if fit == "nf":
        return sci(first(s, "lr_adam_nf", "lr_nf", "lr_implicit")), "--"
    shape = first(s, "lr_adam", "lr_shape", "lr_restore", "lr_ldi")
    amp = s.get("lr_sgd", s.get("lr_amp"))
    energy = s.get("energy_normalised", s.get("energy_biased", False)) and float(s.get("init_std", 0) or 0) == 0
    if amp is None or not energy:
        return sci(shape), "--"
    return sci(shape), sci(amp)


def weights(s):
    """Bedding, equal-elevation and strain weights."""
    w = s.get("weights", {})
    grad, eq = (num(w[k]) if k in w else "--" for k in ("grad_loss", "eq_loss"))
    strain = num(s["strain_weight"]) if "strain_weight" in s else "--"
    return grad, eq, strain


def ensemble_size(e):
    if "ensemble_json" in e:
        return json.loads((HERE / e["ensemble_json"]).read_text())["n_ensemble"]
    if "ensemble" in e:
        with open(HERE / e["ensemble"]) as f:
            rows = list(csv.DictReader(f))
        method = {"ldi": "LDI", "nf": "NF"}[e["fit"]]
        return sum(r["method"] == method for r in rows)
    return None


def listing(names):
    names = list(names)
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]


def panel(title, header, rows, spec):
    """One full-width group of the settings table, laid out as the nomenclature table: the group's
    bold title heads the first column of the header row, with a rule under it."""
    n = len(spec)
    return [rf"\begin{{tabular*}}{{\linewidth}}{{@{{\extracolsep{{\fill}}}}{spec}@{{}}}}",
            r"\toprule", rf"\textbf{{{title}}} & {header}", rf"\cmidrule{{1-{n}}}",
            *rows, r"\bottomrule", r"\end{tabular*}"]


def main_table():
    rows_a, rows_b, ensembles, hinged, cylindrical, extra, other_builds = [], [], {}, [], [], [], []
    for i, e in enumerate(EXAMPLES):
        s = {**e.get("defaults", {}), **json.loads((HERE / e["file"]).read_text())}
        fit = e["fit"]
        epochs = first(s, f"epochs_{fit}", "epochs_implicit" if fit == "nf" else "epochs_restore", "epochs")
        if fit == "ldi" and construction(s) != "one-way sweep":
            other_builds.append(f"{e['name']} ({construction(s)})")
        name = (r"\quad " if e.get("sub") else "") + e["name"]
        gap = ""
        figs = ", ".join(rf"\ref{{main-fig:{f}}}" for f in e["figs"])
        rows_a.append(f"{gap}{name} & {figs} & {e['d']} & {first(s, 'n_features_' + fit, 'n_features')} &{s['n_directions']} & "
                      f"{lengths(s, fit, e.get('model_units'))} \\\\")
        rows_b.append(f"{gap}{name} & {epochs} & " + " & ".join(rates(s, fit)) + " & " + " & ".join(weights(s)) + r" \\")
        if fit != "ldi":
            continue
        label = e["name"] if e["name"] in ("Hartland Quay", "Claudius") else "the " + e["name"][0].lower() + e["name"][1:]
        n = s.get("n_ensemble") or ensemble_size(e)
        if n:
            ensembles.setdefault(int(n), []).append(label)
        if s.get("hinge") is not None:
            (cylindrical if s.get("cylindrical") else hinged).append(label)
        if float(s.get("thickness_weight", 0) or 0) > 0:
            extra.append(f"{label[0].upper() + label[1:]} adds an orthogonal-thickness penalty of weight {num(s['thickness_weight'])}.")
        if s.get("eq_sample"):
            extra.append(f"{label[0].upper() + label[1:]} uses {s['eq_sample']} equality samples on each horizon.")
        if s.get("sampling") == "seismic":
            lines = s["lines"]
            extra.append(f"{label[0].upper() + label[1:]} is fitted to the picks within {lines['half_width']:g}~m of "
                         f"{lines['n_x'] + 2} lines in each direction (Supplementary Note~\\ref{{sec:data_claudius}}).")
        elif "cutoff" in s:
            extra.append(f"{label[0].upper() + label[1:]} keeps a fraction {s['cutoff']:g} of the horizon points.")
    notes = ["Ensembles have " + listing(f"{k} members ({listing(v)})" for k, v in sorted(ensembles.items())) + "."]
    if cylindrical:
        notes.append(f"The map of {listing(cylindrical)} does not vary along its fixed hinge (cylindrical).")
    if hinged:
        notes.append(f"For {listing(hinged)}, material moves normal to a fixed hinge and the motion varies along it.")
    notes += extra
    notes.append("The neutral-surface prior fit of the single layer (main text, Fig.~\\ref{main-fig:single_layer_prior}) "
                 "uses its settings with 1000 more iterations, no strain markers and the Hencky-strain prior of "
                 "weight $10^{-3}$ (main text, Methods).")
    sweep = ("Every map is a single one-way sweep of shears." if not other_builds else
             "Every map is a single one-way sweep of shears, except " + listing(other_builds) + ".")
    caption = (r"\caption{\textbf{Settings of the LDI fits.} Basis: the main-text figure, the "
               r"dimension $d$, and $M$ modes on $R$ rays with wavelengths $[L_{\min},L_{\max}]$. " + sweep + r" Training: iterations, "
               r"learning rates, and the relative weights of bedding, equal elevation and strain before each term is "
               r"scaled by its initial value. Adam updates the phases and biases; where an amplitude rate is given, the energy-normalised amplitudes start at zero and are updated "
               r"by gradient descent with momentum. " + " ".join(notes) + " "
               r"$^{\dagger}$In model units.}")
    head_a = r"Fig. & $d$ & $M$ & $R$ & $[L_{\min},L_{\max}]$ (km) \\"
    head_b = r"Iterations & Adam rate & Amplitude rate & $w_{g}$ & $w_{\mathrm{eq}}$ & $w_{\varepsilon}$ \\"
    return "\n".join([
        r"\begin{table}[htbp]", r"\centering", caption, r"\label{tab:settings}",
        r"{\small\renewcommand{\arraystretch}{1.18}",
        *panel("Basis", head_a, rows_a, "lccccc"),
        r"\par\vspace{1em}",
        *panel("Training", head_b, rows_b, "lcccccc"),
        r"}", r"\end{table}"])


def build():
    return ("% Generated by publication/settings_table.py from the saved fit settings. Do not edit by hand.\n"
            + main_table() + "\n")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true", help="print the tables, write nothing")
    args = ap.parse_args(argv)
    text = build()
    if args.check:
        print(text)
        return 0
    OUT.parent.mkdir(parents=True, exist_ok=True)
    if not OUT.exists() or OUT.read_text() != text:
        OUT.write_text(text)
        print(f"settings_table: wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
