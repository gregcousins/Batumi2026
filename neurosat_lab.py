"""Support code for the Georgia workshop NeuroSAT lab.

Provenance
----------
This module was written for the workshop. It is not copied from the official
NeuroSAT repository and it is not the published TensorFlow implementation.

The following ingredients are based on Selsam et al. (ICLR 2019):
* the SR(n) matched-pair generator;
* literal/clause embeddings with a separate complement pairing;
* literal-to-clause and clause-to-literal sum aggregation;
* recurrent message passing, literal votes, and mean formula readout;
* k-means decoding of candidate satisfying assignments.

The CNF utilities, compact DPLL solver, dataset wrappers, and NumPy classroom inference code were implemented specifically for this notebook. The exact solver is for
small teaching examples, not for benchmarking against modern SAT solvers.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import count
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple
from collections import defaultdict
import math
import random
import time

import numpy as np

Clause = Tuple[int, ...]
Clauses = Tuple[Clause, ...]


@dataclass(frozen=True)
class CNF:
    n_vars: int
    clauses: Clauses
    name: str = ""

    def __post_init__(self) -> None:
        if self.n_vars < 0:
            raise ValueError("n_vars must be nonnegative")
        for clause in self.clauses:
            for lit in clause:
                if lit == 0 or abs(lit) > self.n_vars:
                    raise ValueError(f"invalid literal {lit} for {self.n_vars} variables")

    @property
    def n_clauses(self) -> int:
        return len(self.clauses)

    @property
    def n_literals(self) -> int:
        return sum(len(c) for c in self.clauses)


def normalize_clauses(clauses: Sequence[Sequence[int]]) -> Clauses:
    """Remove repeated literals and omit tautological clauses."""
    out: List[Clause] = []
    for clause in clauses:
        lits = set(int(x) for x in clause)
        if 0 in lits:
            raise ValueError("0 is reserved as the DIMACS clause terminator")
        if any(-lit in lits for lit in lits):
            continue
        out.append(tuple(sorted(lits, key=lambda z: (abs(z), z < 0))))
    return tuple(out)


def make_cnf(n_vars: int, clauses: Sequence[Sequence[int]], name: str = "") -> CNF:
    return CNF(n_vars=n_vars, clauses=normalize_clauses(clauses), name=name)


def literal_value(lit: int, assignment: Sequence[bool] | Dict[int, bool]) -> bool:
    var = abs(lit)
    value = assignment[var] if isinstance(assignment, dict) else assignment[var - 1]
    return bool(value) if lit > 0 else not bool(value)


def clause_satisfied(clause: Clause, assignment: Sequence[bool] | Dict[int, bool]) -> bool:
    return any(literal_value(lit, assignment) for lit in clause)


def satisfies(cnf: CNF, assignment: Sequence[bool] | Dict[int, bool]) -> bool:
    return all(clause_satisfied(clause, assignment) for clause in cnf.clauses)


def cnf_to_dimacs(cnf: CNF) -> str:
    lines = [f"p cnf {cnf.n_vars} {len(cnf.clauses)}"]
    lines.extend(" ".join(map(str, clause)) + " 0" for clause in cnf.clauses)
    return "\n".join(lines) + "\n"


def cnf_to_text(cnf: CNF, max_clauses: int = 12) -> str:
    def lit_text(lit: int) -> str:
        return f"x{lit}" if lit > 0 else f"¬x{-lit}"
    pieces = ["(" + " ∨ ".join(lit_text(l) for l in c) + ")" for c in cnf.clauses[:max_clauses]]
    if len(cnf.clauses) > max_clauses:
        pieces.append(f"… [{len(cnf.clauses) - max_clauses} more clauses]")
    return " ∧ ".join(pieces) if pieces else "⊤"


def cnf_to_latex(
    cnf: CNF,
    *,
    max_clauses: Optional[int] = 12,
    clauses_per_line: int = 2,
    lhs: str = "F",
) -> str:
    """Return a LaTeX display of a CNF formula.

    The internal signed-integer representation is converted as follows:
    ``3`` becomes ``x_3`` and ``-3`` becomes ``\\neg x_3``.
    """

    if clauses_per_line < 1:
        raise ValueError("clauses_per_line must be at least 1")

    if max_clauses is None:
        shown_clauses = cnf.clauses
        omitted = 0
    else:
        if max_clauses < 1:
            raise ValueError("max_clauses must be positive or None")
        shown_clauses = cnf.clauses[:max_clauses]
        omitted = max(0, len(cnf.clauses) - len(shown_clauses))

    def literal_latex(lit: int) -> str:
        return f"x_{{{lit}}}" if lit > 0 else f"\\neg x_{{{-lit}}}"

    def clause_latex(clause: Clause) -> str:
        if not clause:
            return r"\bot"
        body = r"\lor ".join(literal_latex(lit) for lit in clause)
        return rf"\left({body}\right)"

    pieces = [clause_latex(clause) for clause in shown_clauses]

    if omitted:
        pieces.append(rf"\cdots\quad\text{{({omitted} more clauses)}}")

    if not pieces:
        return rf"{lhs}=\top"

    chunks = [
        pieces[i : i + clauses_per_line]
        for i in range(0, len(pieces), clauses_per_line)
    ]

    lines = []
    for index, chunk in enumerate(chunks):
        conjunction = r"\land ".join(chunk)
        if index == 0:
            lines.append(rf"{lhs} &= {conjunction}")
        else:
            lines.append(rf"&\quad {{}}\land {conjunction}")

    return r"\begin{aligned}" + r"\\".join(lines) + r"\end{aligned}"


def display_cnf(
    cnf: CNF,
    *,
    max_clauses: Optional[int] = 12,
    clauses_per_line: int = 2,
    lhs: str = "F",
) -> None:
    """Display a generated CNF formula in standard mathematical notation."""
    try:
        from IPython.display import Math, display
    except ImportError as exc:
        raise RuntimeError(
            "display_cnf requires IPython and is intended for use in Jupyter."
        ) from exc

    title = cnf.name if cnf.name else "CNF formula"
    print(
        f"{title}: {cnf.n_vars} variables, "
        f"{cnf.n_clauses} clauses, {cnf.n_literals} literal occurrences"
    )
    display(
        Math(
            cnf_to_latex(
                cnf,
                max_clauses=max_clauses,
                clauses_per_line=clauses_per_line,
                lhs=lhs,
            )
        )
    )


def simplify_with_literal(clauses: Clauses, literal: int) -> Optional[Clauses]:
    simplified: List[Clause] = []
    opposite = -literal
    for clause in clauses:
        if literal in clause:
            continue
        if opposite in clause:
            reduced = tuple(lit for lit in clause if lit != opposite)
            if not reduced:
                return None
            simplified.append(reduced)
        else:
            simplified.append(clause)
    return tuple(simplified)


def _unit_propagate(clauses: Clauses, assignment: Dict[int, bool]) -> Tuple[Optional[Clauses], Dict[int, bool]]:
    formula = clauses
    assignment = assignment.copy()
    while True:
        unit: Optional[int] = None
        for clause in formula:
            if len(clause) == 0:
                return None, assignment
            if len(clause) == 1:
                unit = clause[0]
                break
        if unit is None:
            return formula, assignment
        var, value = abs(unit), unit > 0
        if var in assignment and assignment[var] != value:
            return None, assignment
        assignment[var] = value
        formula = simplify_with_literal(formula, unit)
        if formula is None:
            return None, assignment
        if not formula:
            return formula, assignment


def _pure_literal_assign(clauses: Clauses, assignment: Dict[int, bool]) -> Tuple[Clauses, Dict[int, bool]]:
    polarity: Dict[int, set[bool]] = defaultdict(set)
    for clause in clauses:
        for lit in clause:
            if abs(lit) not in assignment:
                polarity[abs(lit)].add(lit > 0)
    formula = clauses
    assignment = assignment.copy()
    for var, signs in list(polarity.items()):
        if len(signs) == 1:
            value = next(iter(signs))
            assignment[var] = value
            reduced = simplify_with_literal(formula, var if value else -var)
            if reduced is None:
                return tuple([tuple()]), assignment
            formula = reduced
    return formula, assignment


def _choose_jw(clauses: Clauses) -> Tuple[int, bool]:
    pos: Dict[int, float] = defaultdict(float)
    neg: Dict[int, float] = defaultdict(float)
    for clause in clauses:
        w = 2.0 ** (-len(clause))
        for lit in clause:
            (pos if lit > 0 else neg)[abs(lit)] += w
    variables = set(pos) | set(neg)
    var = max(variables, key=lambda v: (pos[v] + neg[v], -v))
    return var, pos[var] >= neg[var]


def solve_exact(cnf: CNF, *, timeout: Optional[float] = None) -> Tuple[bool, Optional[List[bool]]]:
    """A compact complete DPLL solver, intended for classroom-sized formulas."""
    start = time.perf_counter()

    def recurse(clauses: Clauses, assignment: Dict[int, bool]) -> Optional[Dict[int, bool]]:
        if timeout is not None and time.perf_counter() - start > timeout:
            raise TimeoutError("exact solver timeout")
        clauses, assignment = _unit_propagate(clauses, assignment)
        if clauses is None:
            return None
        if not clauses:
            return assignment
        clauses, assignment = _pure_literal_assign(clauses, assignment)
        if any(len(c) == 0 for c in clauses):
            return None
        if not clauses:
            return assignment
        var, preferred = _choose_jw(clauses)
        for value in (preferred, not preferred):
            lit = var if value else -var
            reduced = simplify_with_literal(clauses, lit)
            if reduced is None:
                continue
            child = assignment.copy()
            child[var] = value
            result = recurse(reduced, child)
            if result is not None:
                return result
        return None

    model = recurse(cnf.clauses, {})
    if model is None:
        return False, None
    full = [bool(model.get(i, False)) for i in range(1, cnf.n_vars + 1)]
    if not satisfies(cnf, full):
        raise AssertionError("internal DPLL error: returned assignment does not satisfy formula")
    return True, full


def random_clause(n_vars: int, k: int, rng: random.Random) -> Clause:
    k = min(max(1, int(k)), n_vars)
    variables = rng.sample(range(1, n_vars + 1), k)
    return tuple(v if rng.random() < 0.5 else -v for v in variables)


def random_kcnf(n_vars: int, n_clauses: int, k: int, rng: random.Random, *, name: str = "") -> CNF:
    return make_cnf(n_vars, [random_clause(n_vars, k, rng) for _ in range(n_clauses)], name=name)


def geometric_sample(p: float, rng: random.Random) -> int:
    """Geometric distribution on {1,2,...}: number of trials until first success."""
    if not 0 < p <= 1:
        raise ValueError("p must lie in (0,1]")
    k = 1
    while rng.random() >= p:
        k += 1
    return k


def generate_sr_pair(
    n_vars: int,
    rng: random.Random,
    *,
    p_bernoulli_one: float = 0.7,
    p_geo: float = 0.4,
    max_clauses: int = 1000,
) -> Tuple[CNF, CNF]:
    """Generate one matched pair from the NeuroSAT-style SR(n) distribution.

    For each new clause, draw B ~ Bernoulli(p_bernoulli_one) and
    G ~ Geometric(p_geo) on {1, 2, ...}.  The proposed width is

        K = 1 + B + G.

    The released NeuroSAT generator samples min(K, n_vars) distinct variables
    without replacement, then chooses each sign independently.  Clauses are
    added until the first unsatisfiable prefix.  Flipping the sign of the first
    literal occurrence in the final clause produces the satisfiable partner.

    Returns (sat_formula, unsat_formula).
    """
    if not 0.0 <= p_bernoulli_one <= 1.0:
        raise ValueError("p_bernoulli_one must lie in [0,1]")

    common: List[Clause] = []
    for _ in range(max_clauses):
        bernoulli_increment = 1 if rng.random() < p_bernoulli_one else 0
        proposed_width = 1 + bernoulli_increment + geometric_sample(p_geo, rng)
        actual_width = min(n_vars, proposed_width)
        clause = random_clause(n_vars, actual_width, rng)
        trial = make_cnf(n_vars, common + [clause])
        sat, _ = solve_exact(trial)
        if sat:
            common.append(clause)
            continue
        unsat_clause = clause
        sat_clause = (-unsat_clause[0],) + unsat_clause[1:]
        unsat_formula = make_cnf(n_vars, common + [unsat_clause], name=f"SR({n_vars})-UNSAT")
        sat_formula = make_cnf(n_vars, common + [sat_clause], name=f"SR({n_vars})-SAT")
        sat_label, _ = solve_exact(sat_formula)
        unsat_label, _ = solve_exact(unsat_formula)
        if not sat_label or unsat_label:
            raise AssertionError("SR pair construction failed")
        return sat_formula, unsat_formula
    raise RuntimeError("failed to reach first UNSAT prefix")


def generate_sr_dataset(
    n_pairs: int,
    min_n: int,
    max_n: int,
    seed: int,
) -> Tuple[List[CNF], np.ndarray]:
    rng = random.Random(seed)
    formulas: List[CNF] = []
    labels: List[int] = []
    for _ in range(n_pairs):
        n = rng.randint(min_n, max_n)
        sat_f, unsat_f = generate_sr_pair(n, rng)
        if rng.random() < 0.5:
            formulas.extend([sat_f, unsat_f]); labels.extend([1, 0])
        else:
            formulas.extend([unsat_f, sat_f]); labels.extend([0, 1])
    return formulas, np.asarray(labels, dtype=np.float32)


def permute_variables(cnf: CNF, rng: random.Random) -> CNF:
    perm = list(range(1, cnf.n_vars + 1))
    rng.shuffle(perm)
    mapping = {i + 1: perm[i] for i in range(cnf.n_vars)}
    clauses = [[mapping[abs(lit)] if lit > 0 else -mapping[abs(lit)] for lit in c] for c in cnf.clauses]
    return make_cnf(cnf.n_vars, clauses, name=cnf.name + " [variables permuted]")


def permute_syntax(cnf: CNF, rng: random.Random) -> CNF:
    clauses = [list(c) for c in cnf.clauses]
    for c in clauses:
        rng.shuffle(c)
    rng.shuffle(clauses)
    # preserve literal order at the raw structural level; make_cnf sorts, so instantiate directly
    return CNF(cnf.n_vars, tuple(tuple(c) for c in clauses), name=cnf.name + " [syntax permuted]")


def flip_variable_globally(cnf: CNF, variable: int) -> CNF:
    if not 1 <= variable <= cnf.n_vars:
        raise ValueError("variable out of range")
    clauses = [[-lit if abs(lit) == variable else lit for lit in c] for c in cnf.clauses]
    return make_cnf(cnf.n_vars, clauses, name=cnf.name + f" [x{variable} flipped]")


def duplicate_clause(cnf: CNF, index: int = 0, copies: int = 1) -> CNF:
    if not cnf.clauses:
        return cnf
    clauses = list(cnf.clauses)
    for _ in range(copies):
        clauses.append(cnf.clauses[index % len(cnf.clauses)])
    return CNF(cnf.n_vars, tuple(clauses), name=cnf.name + " [clause duplicated]")


def equality_pair(x: int, y: int) -> List[Clause]:
    # x = y: (x or not y) and (not x or y)
    return [(x, -y), (-x, y)]


def inequality_pair(x: int, y: int) -> List[Clause]:
    # x != y: (x or y) and (not x or not y)
    return [(x, y), (-x, -y)]


def parity_cycle_formula(bits: Sequence[int], *, name: str = "") -> CNF:
    """Encode x_i xor x_{i+1} = bits[i] around a cycle.

    bits[i]=0 encodes equality; bits[i]=1 encodes inequality.
    The formula is satisfiable iff xor(bits)=0.
    """
    r = len(bits)
    if r < 2:
        raise ValueError("cycle needs at least two variables")
    clauses: List[Clause] = []
    for i, b in enumerate(bits):
        x, y = i + 1, ((i + 1) % r) + 1
        clauses.extend(inequality_pair(x, y) if int(b) else equality_pair(x, y))
    return make_cnf(r, clauses, name=name or f"parity-cycle-{''.join(map(str,bits))}")


def parity_cycle_pair(r: int) -> Tuple[CNF, CNF]:
    if r < 3:
        raise ValueError("use r >= 3")
    even_bits = [0] * r
    odd_bits = [0] * (r - 1) + [1]
    sat = parity_cycle_formula(even_bits, name=f"even parity cycle r={r} (SAT)")
    unsat = parity_cycle_formula(odd_bits, name=f"odd parity cycle r={r} (UNSAT)")
    assert solve_exact(sat)[0] and not solve_exact(unsat)[0]
    return sat, unsat


def three_variable_indistinguishable_pair() -> Tuple[CNF, CNF]:
    sat = make_cnf(3, [
        (1, -3), (-1, 3),
        (1, 2), (-1, -2),
        (2, 3), (-2, -3),
    ], name="regular SAT formula")
    unsat = make_cnf(3, [
        (1, -3), (-1, 3),
        (1, 2), (-1, -2),
        (2, -3), (-2, 3),
    ], name="regular UNSAT formula")
    assert solve_exact(sat)[0] and not solve_exact(unsat)[0]
    return sat, unsat


def pigeonhole_cnf(n_pigeons: int, n_holes: int) -> CNF:
    """Standard encoding: each pigeon is in a hole; no hole has two pigeons."""
    if n_pigeons < 1 or n_holes < 1:
        raise ValueError("positive numbers required")
    def var(p: int, h: int) -> int:
        return p * n_holes + h + 1
    clauses: List[Clause] = []
    for p in range(n_pigeons):
        clauses.append(tuple(var(p, h) for h in range(n_holes)))
    for h in range(n_holes):
        for p in range(n_pigeons):
            for q in range(p + 1, n_pigeons):
                clauses.append((-var(p, h), -var(q, h)))
    return make_cnf(n_pigeons * n_holes, clauses, name=f"PHP({n_pigeons},{n_holes})")



@dataclass
class GraphBatch:
    formulas: List[CNF]
    lit_to_clause_lit: np.ndarray
    lit_to_clause_clause: np.ndarray
    flip_index: np.ndarray
    lit_formula: np.ndarray
    literal_offsets: List[int]
    n_lits: int
    n_clauses: int


def batch_formulas(formulas: Sequence[CNF]) -> GraphBatch:
    """Build the batched literal--clause incidence arrays used by the model."""
    lit_indices: List[int] = []
    clause_indices: List[int] = []
    flip_index: List[int] = []
    lit_formula: List[int] = []
    literal_offsets: List[int] = []
    lit_offset = 0
    clause_offset = 0
    for f_idx, cnf in enumerate(formulas):
        literal_offsets.append(lit_offset)
        n = cnf.n_vars
        # Positive literal x_i is paired with negative literal -x_i.
        for i in range(n):
            flip_index.append(lit_offset + n + i)
        for i in range(n):
            flip_index.append(lit_offset + i)
        lit_formula.extend([f_idx] * (2 * n))
        for j, clause in enumerate(cnf.clauses):
            for lit in clause:
                local_lit = (lit - 1) if lit > 0 else n + (-lit - 1)
                lit_indices.append(lit_offset + local_lit)
                clause_indices.append(clause_offset + j)
        lit_offset += 2 * n
        clause_offset += len(cnf.clauses)
    return GraphBatch(
        formulas=list(formulas),
        lit_to_clause_lit=np.asarray(lit_indices, dtype=np.int64),
        lit_to_clause_clause=np.asarray(clause_indices, dtype=np.int64),
        flip_index=np.asarray(flip_index, dtype=np.int64),
        lit_formula=np.asarray(lit_formula, dtype=np.int64),
        literal_offsets=literal_offsets,
        n_lits=lit_offset,
        n_clauses=clause_offset,
    )


def _sigmoid(x: np.ndarray) -> np.ndarray:
    # Stable enough here because the checkpoint activations are modest.
    return 1.0 / (1.0 + np.exp(-x))


def _linear(x: np.ndarray, weight: np.ndarray, bias: np.ndarray) -> np.ndarray:
    # PyTorch Linear stores weight with shape (out_features, in_features).
    return x @ weight.T + bias


def _mlp(x: np.ndarray, params: Dict[str, np.ndarray], prefix: str) -> np.ndarray:
    h = _linear(x, params[f"{prefix}.net.0.weight"], params[f"{prefix}.net.0.bias"])
    h = np.maximum(h, 0.0)
    return _linear(h, params[f"{prefix}.net.2.weight"], params[f"{prefix}.net.2.bias"])


def _gru_cell(
    x: np.ndarray,
    h: np.ndarray,
    weight_ih: np.ndarray,
    weight_hh: np.ndarray,
    bias_ih: np.ndarray,
    bias_hh: np.ndarray,
) -> np.ndarray:
    """NumPy implementation of torch.nn.GRUCell.

    PyTorch stacks reset, update, and new gates in that order.
    """
    gi = x @ weight_ih.T + bias_ih
    gh = h @ weight_hh.T + bias_hh
    i_r, i_z, i_n = np.split(gi, 3, axis=1)
    h_r, h_z, h_n = np.split(gh, 3, axis=1)
    r = _sigmoid(i_r + h_r)
    z = _sigmoid(i_z + h_z)
    n = np.tanh(i_n + r * h_n)
    return (1.0 - z) * n + z * h


class NeuroSATLite:
    """Small NumPy inference implementation for the workshop checkpoint.

    The checkpoint was trained with the classroom PyTorch implementation.  This
    class evaluates the same learned parameters using NumPy only, so the
    worksheet does not require PyTorch or scikit-learn on participant laptops.
    """

    def __init__(self, params: Dict[str, np.ndarray], d: int, n_rounds: int):
        self.params = params
        self.d = int(d)
        self.n_rounds = int(n_rounds)

    def forward(self, batch: GraphBatch, *, n_rounds: Optional[int] = None, return_embeddings: bool = False):
        rounds = self.n_rounds if n_rounds is None else int(n_rounds)
        p = self.params
        L = np.repeat(p["L_init"][None, :], batch.n_lits, axis=0).copy()
        C = np.repeat(p["C_init"][None, :], batch.n_clauses, axis=0).copy()
        lit_idx = batch.lit_to_clause_lit
        clause_idx = batch.lit_to_clause_clause

        for _ in range(rounds):
            clause_in = np.zeros((batch.n_clauses, self.d), dtype=L.dtype)
            if lit_idx.size:
                np.add.at(clause_in, clause_idx, _mlp(L[lit_idx], p, "L_msg"))
            C = _gru_cell(
                clause_in, C,
                p["C_update.weight_ih"], p["C_update.weight_hh"],
                p["C_update.bias_ih"], p["C_update.bias_hh"],
            )

            literal_in = np.zeros((batch.n_lits, self.d), dtype=L.dtype)
            if lit_idx.size:
                np.add.at(literal_in, lit_idx, _mlp(C[clause_idx], p, "C_msg"))
            literal_features = np.concatenate([literal_in, L[batch.flip_index]], axis=1)
            L = _gru_cell(
                literal_features, L,
                p["L_update.weight_ih"], p["L_update.weight_hh"],
                p["L_update.bias_ih"], p["L_update.bias_hh"],
            )

        votes = _mlp(L, p, "L_vote").reshape(-1)
        n_formulas = len(batch.formulas)
        logits = np.zeros(n_formulas, dtype=votes.dtype)
        np.add.at(logits, batch.lit_formula, votes)
        counts = np.bincount(batch.lit_formula, minlength=n_formulas).astype(votes.dtype)
        logits = logits / np.maximum(counts, 1) + float(np.asarray(p["vote_bias"]))
        if return_embeddings:
            return logits, L, votes, batch
        return logits


def load_checkpoint(path: str) -> Tuple[NeuroSATLite, Dict]:
    """Load the portable NumPy checkpoint distributed with the worksheet."""
    import json
    payload = np.load(path, allow_pickle=False)
    metadata = json.loads(str(payload["__metadata_json__"]))
    params = {k: payload[k] for k in payload.files if k != "__metadata_json__"}
    model = NeuroSATLite(params=params, d=int(metadata["d"]), n_rounds=int(metadata["n_rounds"]))
    return model, dict(metadata.get("metadata", {}))


def predict_probabilities(
    model: NeuroSATLite,
    formulas: Sequence[CNF],
    *,
    batch_size: int = 64,
    n_rounds: Optional[int] = None,
) -> np.ndarray:
    out: List[np.ndarray] = []
    for start in range(0, len(formulas), batch_size):
        batch = batch_formulas(formulas[start:start + batch_size])
        logits = model.forward(batch, n_rounds=n_rounds)
        out.append(_sigmoid(logits))
    return np.concatenate(out) if out else np.empty(0)


def _two_means(x: np.ndarray, max_iter: int = 100) -> np.ndarray:
    """Tiny deterministic two-means routine used only for assignment decoding."""
    if len(x) < 2:
        raise ValueError("need at least two points")
    # Start with the pair of points that are farthest apart.
    d2 = ((x[:, None, :] - x[None, :, :]) ** 2).sum(axis=2)
    i, j = np.unravel_index(np.argmax(d2), d2.shape)
    centres = np.stack([x[i], x[j]], axis=0).copy()
    labels = np.zeros(len(x), dtype=np.int64)
    for _ in range(max_iter):
        distances = ((x[:, None, :] - centres[None, :, :]) ** 2).sum(axis=2)
        new_labels = distances.argmin(axis=1)
        if np.array_equal(new_labels, labels):
            labels = new_labels
            break
        labels = new_labels
        new_centres = centres.copy()
        for k in (0, 1):
            mask = labels == k
            if mask.any():
                new_centres[k] = x[mask].mean(axis=0)
        if np.allclose(new_centres, centres):
            centres = new_centres
            break
        centres = new_centres
    return centres


def decode_assignment(
    model: NeuroSATLite,
    cnf: CNF,
    *,
    n_rounds: Optional[int] = None,
) -> Optional[List[bool]]:
    batch = batch_formulas([cnf])
    _, L, votes, _ = model.forward(batch, n_rounds=n_rounds, return_embeddings=True)
    n = cnf.n_vars

    # First try a cheap orientation derived from the literal votes.
    cheap = [bool(votes[i] > votes[n + i]) for i in range(n)]
    for candidate in (cheap, [not x for x in cheap]):
        if satisfies(cnf, candidate):
            return candidate

    # Then mimic the paper's clustering idea with a small NumPy two-means.
    centres = _two_means(L)
    dist2 = ((L[:, None, :] - centres[None, :, :]) ** 2).sum(axis=2)
    assignment: List[bool] = []
    for i in range(n):
        d1 = dist2[i, 0] + dist2[n + i, 1]
        d2 = dist2[i, 1] + dist2[n + i, 0]
        assignment.append(bool(d1 < d2))
    for candidate in (assignment, [not x for x in assignment]):
        if satisfies(cnf, candidate):
            return candidate
    return None


def evaluate_classifier(
    model: NeuroSATLite,
    formulas: Sequence[CNF],
    labels: Sequence[int | float],
    *,
    n_rounds: Optional[int] = None,
) -> Dict[str, float]:
    probs = predict_probabilities(model, formulas, n_rounds=n_rounds)
    y = np.asarray(labels, dtype=np.int64)
    pred = (probs >= 0.5).astype(np.int64)
    sat_mask = y == 1
    unsat_mask = y == 0
    return {
        "n": float(len(y)),
        "accuracy": float(np.mean(pred == y)),
        "sat_accuracy": float(np.mean(pred[sat_mask] == y[sat_mask])) if sat_mask.any() else float("nan"),
        "unsat_accuracy": float(np.mean(pred[unsat_mask] == y[unsat_mask])) if unsat_mask.any() else float("nan"),
        "mean_probability": float(np.mean(probs)),
    }


def literal_occurrence_degrees(cnf: CNF) -> Dict[int, int]:
    """Return the number of clauses containing each signed literal."""
    degrees = {lit: 0 for v in range(1, cnf.n_vars + 1) for lit in (v, -v)}
    for clause in cnf.clauses:
        for lit in clause:
            degrees[lit] += 1
    return degrees


def balanced_random_kcnf_dataset(
    *,
    n_vars: int,
    k: int,
    clause_ratio: float,
    n_per_class: int,
    seed: int,
    max_attempts: int = 100000,
) -> Tuple[List[CNF], np.ndarray]:
    """Rejection-sample an exactly balanced random k-CNF dataset."""
    rng = random.Random(seed)
    n_clauses = max(1, int(round(clause_ratio * n_vars)))
    sat_formulas: List[CNF] = []
    unsat_formulas: List[CNF] = []
    for attempt in range(max_attempts):
        if len(sat_formulas) >= n_per_class and len(unsat_formulas) >= n_per_class:
            break
        cnf = random_kcnf(n_vars, n_clauses, k, rng, name=f"random {k}-CNF")
        is_sat, _ = solve_exact(cnf)
        target = sat_formulas if is_sat else unsat_formulas
        if len(target) < n_per_class:
            target.append(cnf)
    if len(sat_formulas) < n_per_class or len(unsat_formulas) < n_per_class:
        raise RuntimeError(
            f"Could not collect {n_per_class} examples of each class; "
            f"got {len(sat_formulas)} SAT and {len(unsat_formulas)} UNSAT. "
            "Try a different clause ratio."
        )
    formulas: List[CNF] = []
    labels: List[int] = []
    for sat_f, unsat_f in zip(sat_formulas, unsat_formulas):
        if rng.random() < 0.5:
            formulas.extend([sat_f, unsat_f]); labels.extend([1, 0])
        else:
            formulas.extend([unsat_f, sat_f]); labels.extend([0, 1])
    return formulas, np.asarray(labels, dtype=np.int64)


def flip_random_literal_occurrence(cnf: CNF, rng: random.Random) -> CNF:
    """Negate one randomly chosen literal occurrence."""
    if not cnf.clauses:
        return cnf
    clause_idx = rng.randrange(len(cnf.clauses))
    if not cnf.clauses[clause_idx]:
        return cnf
    lit_idx = rng.randrange(len(cnf.clauses[clause_idx]))
    clauses = [list(c) for c in cnf.clauses]
    clauses[clause_idx][lit_idx] *= -1
    return make_cnf(cnf.n_vars, clauses, name=cnf.name + " [one sign flipped]")


def wrong_answer_confidence(prob_sat: float, true_label: int) -> float:
    """Probability mass assigned to the wrong class."""
    return float(prob_sat if true_label == 0 else 1.0 - prob_sat)
