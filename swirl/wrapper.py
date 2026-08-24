#!/usr/bin/env python3
# SWIRL index-selection wrapper (long-lived HTTP server).
#
# Endpoints: GET /health, POST /recommend, POST /shutdown, GET /state.
# A PPO2 policy runs a gym_db episode; what-if costs use HypoPG hypothetical
# indexes on --eval-dsn (defaults to --dsn). Real indexes are created only with
# --apply. Training runs in a background subprocess on the eval DSN.

import os
import sys
import json
import time
import copy
import threading
import tempfile
import subprocess
import importlib
import traceback
import contextlib
import io
import re
import warnings
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
warnings.filterwarnings("ignore")

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import numpy as np
import psycopg2

# --------------------------------------------------------------------------- #
# Runtime patches for the target database connector.                          #
# --------------------------------------------------------------------------- #
from index_selection_evaluation.selection.dbms.postgres_dbms import (
    PostgresDatabaseConnector as _PGC,
)
from index_selection_evaluation.selection import what_if_index_creation as _wi_mod
from index_selection_evaluation.selection.workload import (
    Query,
    Workload,
    Column,
    Table,
)
import swirl.utils as _swirl_utils
from gym_db.common import EnvironmentType

# Accept a full DSN in place of a bare database name.
def _create_connection_dsn(self):
    if self._connection:
        self.close()
    db = str(self.db_name) if self.db_name is not None else ""
    if db.startswith("postgres") or "://" in db:
        self._connection = psycopg2.connect(db)
    else:
        self._connection = psycopg2.connect("dbname={}".format(db))
    self._connection.autocommit = self.autocommit
    self._cursor = self._connection.cursor()
_PGC.create_connection = _create_connection_dsn

# Never drop real indexes on the target DSN.
def _drop_indexes_noop(self):
    _log("drop_indexes() neutered (non-invasive on user DSN)")
_PGC.drop_indexes = _drop_indexes_noop

# Idempotent HypoPG extension setup.
def _enable_simulation_safe(self):
    self.exec_only("CREATE EXTENSION IF NOT EXISTS hypopg")
    self.commit()
_PGC.enable_simulation = _enable_simulation_safe

# Fallback when hypopg_list_indexes() is unavailable.
def _all_simulated_indexes_safe(self):
    try:
        return self.db_connector.exec_fetch("select * from hypopg_list_indexes()", one=False)
    except Exception:
        return [(oid, name) for oid, name in self.simulated_indexes.items()]
_wi_mod.WhatIfIndexCreation.all_simulated_indexes = _all_simulated_indexes_safe


def _log(msg):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print("[swirl-wrapper {}] {}".format(ts, msg), flush=True)


# --------------------------------------------------------------------------- #
# Config / defaults.                                                          #
# --------------------------------------------------------------------------- #
DEFAULTS = {
    "model_dir": os.path.join(REPO, "skill_state"),
    "max_index_width": 2,
    "max_steps_per_episode": 10,
    "time_budget_seconds": 60,
    "train_trigger": 1,
    "timesteps": 2000,
    "training_workloads": 20,
    "training_workload_size": 10,
    "action_manager": "MultiColumnIndexActionManager",
    "observation_manager": "SingleColumnIndexObservationManager",
    "reward_calculator": "RelativeDifferenceRelativeToStorageReward",
    "reenable_indexes": False,
    "random_seed": 60,
    "obs_query_classes": 256,
    "candidate_columns": None,
}

DEFAULT_STORAGE_MB = 1000  # used when --budget omits storage_mb

STATE_LOCK = threading.RLock()
PERSIST_LOCK = threading.Lock()
# Serializes /recommend (shared STATE/model/schema).
RECOMMEND_LOCK = threading.Lock()
_training_in_progress = {"value": False}
_shutdown_requested = {"value": False}
_untrained_counter = {"value": 0}


class SkillState:
    def __init__(self):
        self.dsn = None
        self.eval_dsn = None
        self.config = dict(DEFAULTS)
        self.model_dir = self.config["model_dir"]
        self.model = None
        self.vec_normalize = None
        self.model_mtime = 0
        self.schema_columns = None
        self.schema_tables = None
        self.last_recommend = {}
        self.size_cache = {}
        self.training_loss_log = []

    def reload_if_changed(self):
        path = os.path.join(self.model_dir, "model.zip")
        if not os.path.exists(path):
            return
        mtime = os.path.getmtime(path)
        if mtime > self.model_mtime:
            _log("Model file changed (mtime {} > {}); will reload".format(mtime, self.model_mtime))
            self.model = None
            self.vec_normalize = None
            self.model_mtime = mtime

STATE = SkillState()


def _model_meta():
    try:
        with open(os.path.join(STATE.model_dir, "model_meta.json")) as f:
            return json.load(f)
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# Schema introspection from the target database catalog.                      #
# --------------------------------------------------------------------------- #
def introspect_schema(dsn):
    conn = psycopg2.connect(dsn)
    conn.autocommit = True
    cur = conn.cursor()
    cur.execute(
        "SELECT table_name, column_name FROM information_schema.columns "
        "WHERE table_schema='public' ORDER BY table_name, ordinal_position"
    )
    tables = {}
    columns = []
    for tname, cname in cur.fetchall():
        tname_l = tname.lower()
        cname_l = cname.lower()
        if tname_l not in tables:
            tables[tname_l] = Table(tname_l)
        col = Column(cname_l)
        tables[tname_l].add_column(col)
        columns.append(col)
    cur.close()
    conn.close()
    return list(tables.values()), columns


def _base_table_names(dsn):
    try:
        conn = psycopg2.connect(dsn)
        conn.autocommit = True
        cur = conn.cursor()
        cur.execute("SELECT table_name FROM information_schema.tables "
                    "WHERE table_schema='public' AND table_type='BASE TABLE'")
        names = {t.lower() for (t,) in cur.fetchall()}
        cur.close()
        conn.close()
        return names
    except Exception as e:
        _log("base-table introspection failed: {}".format(e))
        return set()


def _universe_columns(dsn, schema_columns, config):
    """Schema-scoped indexable candidate universe (deterministic order).

    Default: all base-table columns, or the --config candidate_columns list
    of "table.column" entries. The resulting action/observation dimensions
    are workload-independent."""
    requested = (config or {}).get("candidate_columns")
    if requested:
        by_key = {(c.table.name, c.name): c for c in schema_columns}
        out = []
        for item in requested:
            try:
                if isinstance(item, str) and "." in item:
                    t, cname = item.split(".", 1)
                else:
                    t, cname = item[0], item[1]
            except Exception:
                continue
            c = by_key.get((t.lower(), cname.lower()))
            if c is not None and c not in out:
                out.append(c)
            else:
                _log("candidate_columns: unknown {}.{}".format(t, cname))
        if out:
            return out
        _log("candidate_columns: none resolved; using schema columns")
    base = _base_table_names(dsn)
    cols = [c for c in schema_columns if (not base) or c.table.name in base]
    return sorted(cols, key=lambda c: (c.table.name, c.name))


def _build_universe_candidates(dsn, config, schema_columns):
    """Width-1 schema-scoped action space (multi-column widths stay
    workload-scoped: permutations over the whole schema explode)."""
    cols = _universe_columns(dsn, schema_columns, config)
    flat = [(c,) for c in cols]
    return [flat], flat


def _action_sizes(flat, dsn):
    """Cached HypoPG size estimates for the candidate list (schema-scoped
    candidate sets are stable, so this is computed once per DSN/universe)."""
    key = (dsn, tuple(tuple((c.table.name, c.name) for c in cc) for cc in flat))
    if key not in STATE.size_cache:
        STATE.size_cache[key] = _swirl_utils.predict_index_sizes(flat, dsn)
    return STATE.size_cache[key]


def existing_index_set(dsn):
    conn = psycopg2.connect(dsn)
    conn.autocommit = True
    cur = conn.cursor()
    cur.execute(
        "SELECT schemaname, tablename, indexname, indexdef FROM pg_indexes "
        "WHERE schemaname='public'"
    )
    out = set()
    names = set()
    for schemaname, tablename, indexname, indexdef in cur.fetchall():
        names.add(indexname)
        cols = _parse_indexdef_columns(indexdef)
        if cols:
            out.add((tablename.lower(), tuple(c.lower() for c in cols)))
    cur.close()
    conn.close()
    return out, names


def _existing_prefixes(dsn):
    """Prefixes of existing real indexes: a candidate combination matching one
    is redundant (a btree also serves leftmost-prefix predicates), so it is
    masked as an ACTION rather than being emitted and skipped later."""
    try:
        existing, _ = existing_index_set(dsn)
    except Exception as e:
        _log("existing-index introspection failed: {}".format(e))
        return set()
    prefixes = set()
    for table, cols in existing:
        for k in range(1, len(cols) + 1):
            prefixes.add((table, tuple(cols[:k])))
    return prefixes


_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_$]*$")


_QUALIFIED_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_$]*(\.[A-Za-z_][A-Za-z0-9_$]*)?$")
_QUOTED_IDENT_RE = re.compile(r'^"((?:[^"]|"")*)"(.*)$', re.DOTALL)


def _parse_index_key_part(part):
    """One key-column entry of a pg_indexes.indexdef -> lower-case column name,
    or None for an expression. Strips [COLLATE <name>] [opclass]
    [ASC|DESC] [NULLS FIRST|LAST]."""
    s = part.strip()
    if not s:
        return None
    m = _QUOTED_IDENT_RE.match(s)
    if m:
        name = m.group(1).replace('""', '"')
        rest = m.group(2).strip()
        tokens = rest.split() if rest else []
        quoted = True
    else:
        tokens = s.split()
        name = tokens[0]
        tokens = tokens[1:]
        quoted = False
    opclass_seen = False
    i = 0
    while i < len(tokens):
        tok = tokens[i].upper()
        if tok == "COLLATE":
            i += 2            # COLLATE <collation name>
            continue
        if tok == "NULLS":
            i += 2            # NULLS FIRST|LAST
            continue
        if tok in ("ASC", "DESC"):
            i += 1
            continue
        if not opclass_seen and _QUALIFIED_IDENT_RE.match(tokens[i].strip('"')):
            opclass_seen = True   # e.g. text_pattern_ops / pg_catalog.int4_ops
            i += 1
            continue
        return None
    if not name or (not quoted and not _IDENT_RE.match(name)):
        return None
    return name.lower()


def _parse_indexdef_key_columns(indexdef):
    """Key (btree) columns of a pg_indexes.indexdef; [] when the key cannot be
    read with certainty (expression or non-btree index). INCLUDE is not part
    of the key."""
    if not indexdef:
        return []
    m = re.search(r"\bON\b\s+\S+\s+USING\s+btree\s*\(([^)]*)\)", indexdef, re.I)
    if not m:
        m = re.search(r"\bON\b\s+\S+\s*\(([^)]*)\)", indexdef, re.I)
    if not m:
        return []
    cols = []
    for part in m.group(1).split(","):
        if not part.strip():
            continue
        name = _parse_index_key_part(part)
        if name is None:
            return []  # expression key — do not guess
        cols.append(name)
    return cols

def _parse_indexdef_columns(indexdef):
    return _parse_indexdef_key_columns(indexdef)


# --------------------------------------------------------------------------- #
# Workload parsing (from --workload: .sql dir/file or JSON list of entries).  #
# --------------------------------------------------------------------------- #
_DIRECTIVE_RE = re.compile(r"^:[A-Za-z]$")


def _clean_stmt(text):
    """Strip SQL line comments and dbgen-style directives (:x/:o) so the
    remaining text begins with the actual SQL verb. Returns the cleaned
    statement (comments/directives removed)."""
    kept = []
    for ln in text.splitlines():
        s = ln.strip()
        if not s or s.startswith("--") or _DIRECTIVE_RE.match(s):
            continue
        kept.append(ln)
    return "\n".join(kept).strip()


def parse_workload(workload_arg, schema_columns, query_classes=256):
    entries = _load_workload_entries(workload_arg)
    queries = []
    notes = []
    for i, (text, freq) in enumerate(entries):
        text = _clean_stmt((text or "").strip())
        if not text:
            continue
        text = text.rstrip(";").strip()
        if not (text.lstrip().lower().startswith("select") or text.lstrip().lower().startswith("with")):
            notes.append("Q{}: non-SELECT statement excluded from read-cost analysis".format(i + 1))
            continue
        # Query numbers are pinned to the fixed query-class width.
        nr = (i % query_classes) + 1
        q = Query(nr, text, frequency=freq)
        n = _populate_query_columns(q, schema_columns)
        if not n:
            notes.append("Q{}: no indexable predicate columns matched; excluded".format(q.nr))
            continue
        queries.append(q)
    if not queries:
        notes.append("workload yielded no indexable queries")
    return Workload(queries), notes


def _load_workload_entries(workload_arg):
    if workload_arg is None:
        return []
    # Accept a workload supplied directly as a JSON list of entries (not only as
    # a JSON-list string, file path, or bare SQL string).
    if isinstance(workload_arg, list):
        out = []
        for e in workload_arg:
            if isinstance(e, str):
                out.append((e, 1))
            elif isinstance(e, dict):
                sql = e.get("sql") or e.get("query") or e.get("text") or ""
                fr = e.get("weight", e.get("frequency", 1))
                try:
                    fr = float(fr)
                except Exception:
                    fr = 1
                out.append((sql, fr))
        return out
    if isinstance(workload_arg, str) and workload_arg.lstrip().startswith("["):
        try:
            data = json.loads(workload_arg)
        except Exception:
            data = None
        if isinstance(data, list):
            out = []
            for e in data:
                if isinstance(e, str):
                    out.append((e, 1))
                elif isinstance(e, dict):
                    sql = e.get("sql") or e.get("query") or e.get("text") or ""
                    fr = e.get("weight", e.get("frequency", 1))
                    try:
                        fr = float(fr)
                    except Exception:
                        fr = 1
                    out.append((sql, fr))
            return out
    if isinstance(workload_arg, str) and os.path.exists(workload_arg):
        files = []
        if os.path.isdir(workload_arg):
            for root, _, fs in os.walk(workload_arg):
                for f in sorted(fs):
                    if f.lower().endswith(".sql"):
                        files.append(os.path.join(root, f))
        else:
            files = [workload_arg]
        out = []
        for fp in files:
            try:
                with open(fp, "r") as fh:
                    content = fh.read()
            except Exception as e:
                _log("could not read {}: {}".format(fp, e))
                continue
            for stmt in _split_sql_stmts(content):
                if stmt.strip():
                    out.append((stmt, 1))
        return out
    return [(workload_arg, 1)]


def _split_sql_stmts(content):
    content = content.strip()
    if ";" not in content:
        return [content] if content else []
    stmts = [s.strip() for s in content.split(";")]
    return [s for s in stmts if s]


def _populate_query_columns(query, schema_columns):
    # Strip string/number literals so a column name that happens to appear
    # inside a 'string constant' is not treated as a predicate column.
    text = re.sub(r"'[^']*'", " ", query.text)
    text = re.sub(r"\b\d+\b", " ", text)
    tl = text.lower()
    where_idx = tl.find(" where ")
    if where_idx < 0:
        where_idx = tl.find("\nwhere ")
    if where_idx < 0:
        before, after = tl, ""
    else:
        before, after = tl[:where_idx], tl[where_idx:]
    attached = 0
    for col in schema_columns:
        tname = col.table.name
        tname_in_from = bool(re.search(r"\b" + re.escape(tname) + r"\b", before))
        if where_idx >= 0:
            if re.search(r"\b" + re.escape(col.name) + r"\b", after) and tname_in_from:
                query.columns.append(col)
                attached += 1
        else:
            if re.search(r"\b" + re.escape(col.name) + r"\b", tl) and tname_in_from:
                query.columns.append(col)
                attached += 1
    return attached


# --------------------------------------------------------------------------- #
# Candidate / model component construction.                                   #
# --------------------------------------------------------------------------- #
def _build_candidates(queries, max_index_width):
    indexable_columns = []
    seen = set()
    for q in queries:
        for c in q.columns:
            if c not in seen:
                seen.add(c)
                indexable_columns.append(c)
    combos = _swirl_utils.create_column_permutation_indexes(indexable_columns, max_index_width)
    flat = [item for sub in combos for item in sub]
    # create_column_permutation_indexes() returns set order (process-random);
    # sort deterministically so action indices are stable across processes.
    flat = sorted(flat, key=lambda cc: (cc[0].table.name,
                                        tuple(c.name for c in cc)))
    combos = _regroup_by_width(flat, max_index_width)
    # env.globally_indexable_columns and the action manager must use the same
    # order: width-1 actions first, then each wider group.
    flat = [item for sub in combos for item in sub]
    for gid, column_combination in enumerate(flat):
        column_combination[0].global_column_id = gid
    return combos, flat


def _regroup_by_width(flat, max_index_width):
    by_width = [[] for _ in range(max_index_width)]
    for column_combination in flat:
        by_width[len(column_combination) - 1].append(column_combination)
    return by_width


def _mask_existing_actions(action_manager, existing_prefixes):
    """Mask actions whose column combination is a prefix of an existing real
    index."""
    orig = action_manager.get_initial_valid_actions

    def _masked(workload, budget):
        va = orig(workload, budget)
        remaining = []
        for idx in list(action_manager._remaining_valid_actions):
            cc = action_manager.indexable_column_combinations_flat[idx]
            key = (cc[0].table.name, tuple(c.name for c in cc))
            if key in existing_prefixes:
                action_manager.valid_actions[idx] = action_manager.FORBIDDEN_ACTION
            else:
                remaining.append(idx)
        action_manager._remaining_valid_actions = remaining
        return np.array(action_manager.valid_actions)

    action_manager.get_initial_valid_actions = _masked


def _build_action_manager(combos, flat, dsn, config):
    consumptions = _action_sizes(flat, dsn)
    cls = getattr(importlib.import_module("swirl.action_manager"), config["action_manager"])
    return cls(
        indexable_column_combinations=combos,
        action_storage_consumptions=consumptions,
        sb_version=2,
        max_index_width=config["max_index_width"],
        reenable_indexes=config["reenable_indexes"],
    )


def _build_observation_manager(action_manager, config):
    cls = getattr(importlib.import_module("swirl.observation_manager"), config["observation_manager"])
    # Fixed observation dimension (query-class width is zero-padded).
    # action_status has one entry per single-column action, not per
    # combination (matches the source's experiment.make_env).
    return cls(
        action_manager.number_of_columns,
        {
            "number_of_query_classes": config["obs_query_classes"],
            "workload_embedder": None,
            "workload_size": config["training_workload_size"],
        },
    )


def _build_reward_calculator(config):
    cls = getattr(importlib.import_module("swirl.reward_calculator"), config["reward_calculator"])
    return cls()


def _make_env(dsn, workload, budget_mb, max_steps, config, environment_type, env_id=0,
                 candidate_workload=None, schema_columns=None):
    from gym_db.envs.db_env_v1 import DBEnvV1
    if schema_columns is None:
        _, schema_columns = introspect_schema(dsn)
    if int(config.get("max_index_width", 1)) == 1:
        combos, flat = _build_universe_candidates(dsn, config, schema_columns)
    else:
        cand_src = candidate_workload if candidate_workload is not None else workload
        combos, flat = _build_candidates(cand_src.queries, config["max_index_width"])
    action_manager = _build_action_manager(combos, flat, dsn, config)
    _mask_existing_actions(action_manager, _existing_prefixes(dsn))
    observation_manager = _build_observation_manager(action_manager, config)
    reward_calculator = _build_reward_calculator(config)
    workload.budget = budget_mb
    env = DBEnvV1(
        environment_type=environment_type,
        config={
            "database_name": dsn,
            "globally_indexable_columns": flat,
            "workloads": [workload],
            "random_seed": config["random_seed"] + env_id,
            "max_steps_per_episode": max_steps,
            "action_manager": action_manager,
            "observation_manager": observation_manager,
            "reward_calculator": reward_calculator,
            "env_id": env_id,
            "similar_workloads": False,
        },
    )
    return env, action_manager


def _close_env(env):
    """Release a per-request env's database session.

    Drops the hypothetical indexes via complete_cost_estimation(), then
    closes the env's own DBEnvV1 connector so the session and its server-side
    HypoPG state are released. Idempotent and failure-tolerant.
    """
    try:
        if getattr(env, "cost_evaluation", None) is not None:
            env.cost_evaluation.complete_cost_estimation()
    except Exception:
        pass
    try:
        if getattr(env, "connector", None) is not None:
            env.connector.close()
    except Exception:
        pass


# --------------------------------------------------------------------------- #
# PPO2 model load / bootstrap / reload.                                       #
# --------------------------------------------------------------------------- #
# Non-dimension knobs that can change per request.
_PER_REQUEST_KNOBS = ("max_steps_per_episode", "train_trigger", "timesteps",
                      "training_workloads", "training_workload_size", "random_seed",
                      "time_budget_seconds")


def _ensure_state_for_dsn(dsn, eval_dsn, config):
    config = config or {}
    try:
        width = int(config.get("max_index_width", STATE.config.get("max_index_width", 1)))
    except Exception:
        width = STATE.config.get("max_index_width", 1)
    try:
        query_classes = int(config.get("obs_query_classes",
                                       STATE.config.get("obs_query_classes", 256)))
    except Exception:
        query_classes = STATE.config.get("obs_query_classes", 256)
    model_dir = config.get("model_dir") or STATE.config.get("model_dir")
    candidate_columns = config.get("candidate_columns") or None
    # Dimension changes and DSN/model_dir switches reset in-memory state.
    changed = (STATE.dsn != dsn
               or STATE.eval_dsn != (eval_dsn or dsn)
               or STATE.config.get("max_index_width") != width
               or STATE.config.get("obs_query_classes") != query_classes
               or STATE.config.get("candidate_columns") != candidate_columns
               or STATE.model_dir != model_dir)
    if changed:
        STATE.dsn = dsn
        STATE.eval_dsn = eval_dsn or dsn
        STATE.config = dict(DEFAULTS)
        STATE.config.update(config)
        # Manager/reward classes are not configurable.
        STATE.config["action_manager"] = DEFAULTS["action_manager"]
        STATE.config["observation_manager"] = DEFAULTS["observation_manager"]
        STATE.config["reward_calculator"] = DEFAULTS["reward_calculator"]
        STATE.config["reenable_indexes"] = DEFAULTS["reenable_indexes"]
        STATE.config["max_index_width"] = width
        STATE.config["obs_query_classes"] = query_classes
        STATE.model_dir = model_dir
        os.makedirs(STATE.model_dir, exist_ok=True)
        STATE.schema_columns = None
        STATE.model = None
        STATE.vec_normalize = None
        STATE.model_mtime = 0
    # Non-dimension knobs apply per request (and after a state reset).
    for key in _PER_REQUEST_KNOBS:
        if key in config:
            try:
                if key == "time_budget_seconds":
                    STATE.config[key] = float(config[key])
                else:
                    STATE.config[key] = int(config[key])
            except Exception:
                pass


def _bootstrap_model(env, config):
    from stable_baselines import PPO2
    from stable_baselines.common.vec_env import DummyVecEnv, VecNormalize
    venv = VecNormalize(
        DummyVecEnv([lambda: env]),
        norm_obs=True,
        norm_reward=True,
        gamma=0.5,
        training=True,
    )
    model = PPO2(
        policy="MlpPolicy",
        env=venv,
        verbose=0,
        seed=config["random_seed"],
        gamma=0.5,
        n_steps=64,
        policy_kwargs={"net_arch": [{"vf": [256, 256], "pi": [256, 256]}]},
    )
    return model, venv


def _load_model_from_disk():
    from stable_baselines import PPO2
    from stable_baselines.common.vec_env import DummyVecEnv, VecNormalize
    path = os.path.join(STATE.model_dir, "model.zip")
    if not os.path.exists(path):
        STATE.model = None
        STATE.vec_normalize = None
        return
    try:
        wl_path = os.path.join(STATE.model_dir, "last_workload.json")
        workload_arg = None
        if os.path.exists(wl_path):
            try:
                with open(wl_path) as f:
                    workload_arg = json.load(f).get("workload")
            except Exception:
                workload_arg = None
        if STATE.schema_columns is None:
            STATE.schema_tables, STATE.schema_columns = introspect_schema(STATE.dsn)
        workload, _ = parse_workload(workload_arg or "select 1", STATE.schema_columns,
                                     query_classes=STATE.config["obs_query_classes"])
        if not workload.queries:
            raise RuntimeError("persisted workload has no indexable queries")
        # Check the persisted workload against the trained model's dimensions
        # before building an env.
        meta = _model_meta()
        if meta is not None:
            if int(STATE.config.get("max_index_width", 1)) == 1:
                _, flat_chk = _build_universe_candidates(
                    STATE.eval_dsn or STATE.dsn, STATE.config, STATE.schema_columns)
            else:
                _, flat_chk = _build_candidates(workload.queries,
                                                STATE.config["max_index_width"])
            if meta.get("n_actions") != len(flat_chk):
                _log("persisted workload ({} candidates) does not match trained model "
                     "({} actions); bootstrapping fresh".format(len(flat_chk), meta.get("n_actions")))
                STATE.model = None
                STATE.vec_normalize = None
                return
        env, _ = _make_env(STATE.eval_dsn or STATE.dsn, workload, None,
                            STATE.config["max_steps_per_episode"], STATE.config,
                            EnvironmentType.TESTING,
                            schema_columns=STATE.schema_columns)
        if meta is not None and meta.get("n_features") not in (
                None, int(env.observation_space.shape[0])):
            _log("model n_features={} does not match env observation {}; "
                 "bootstrapping fresh".format(meta.get("n_features"),
                                              env.observation_space.shape[0]))
            _close_env(env)
            STATE.model = None
            STATE.vec_normalize = None
            return
        venv = VecNormalize(DummyVecEnv([lambda: env]), norm_obs=True, norm_reward=False, training=False)
        # Restore the normalization statistics saved with the model
        # (VecNormalize.load is a staticmethod).
        try:
            import pickle
            with open(os.path.join(STATE.model_dir, "vecnormalize.pkl"), "rb") as fh:
                saved_norm = pickle.load(fh)
            if np.shape(saved_norm.obs_rms.mean) == np.shape(venv.obs_rms.mean):
                venv.obs_rms = copy.deepcopy(saved_norm.obs_rms)
                venv.ret_rms = copy.deepcopy(saved_norm.ret_rms)
            else:
                _log("vecnormalize stats shape mismatch; skipped")
        except Exception as e:
            _log("vecnormalize stats not loaded: {}".format(e))
        model = PPO2.load(path, env=venv)
        STATE.model = model
        STATE.vec_normalize = venv
        _log("trained model loaded from disk")
    except Exception as e:
        _log("model load failed ({}); will bootstrap on next recommend".format(e))
        STATE.model = None
        STATE.vec_normalize = None


# --------------------------------------------------------------------------- #
# Recommendation.                                                             #
# --------------------------------------------------------------------------- #
def _as_dict(value):
    """Accept a JSON object or a JSON-encoded object; anything else -> {}."""
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except Exception:
            return {}
    return value if isinstance(value, dict) else {}


def recommend(dsn, eval_dsn, workload_arg, budget, config, recommend_only, apply):
    with RECOMMEND_LOCK:
        return _recommend_locked(dsn, eval_dsn, workload_arg, budget, config,
                                 recommend_only, apply)


def _recommend_locked(dsn, eval_dsn, workload_arg, budget, config, recommend_only, apply):
    t0 = time.time()
    _ensure_state_for_dsn(dsn, eval_dsn, config or {})
    cfg = STATE.config
    STATE.reload_if_changed()

    b = budget or {}
    storage_mb = b.get("storage_mb", DEFAULT_STORAGE_MB)
    try:
        budget_mb = float(storage_mb) if storage_mb else None
    except Exception:
        budget_mb = float(DEFAULT_STORAGE_MB)
    try:
        max_indexes = int(b.get("max_indexes", cfg["max_steps_per_episode"]))
    except Exception:
        max_indexes = cfg["max_steps_per_episode"]
    try:
        write_overhead_pct = float(b["max_write_overhead_pct"])
    except (KeyError, TypeError, ValueError):
        write_overhead_pct = None
    eff_budget = dict(b)
    eff_budget["storage_mb"] = budget_mb
    eff_budget["max_indexes"] = max_indexes
    eff_budget["max_write_overhead_pct"] = write_overhead_pct
    try:
        time_budget = float(cfg.get("time_budget_seconds") or 0)
    except (TypeError, ValueError):
        time_budget = float(DEFAULTS["time_budget_seconds"])
    deadline = (t0 + time_budget) if time_budget > 0 else None

    if STATE.schema_columns is None:
        _log("introspecting schema for {}".format(dsn))
        try:
            STATE.schema_tables, STATE.schema_columns = introspect_schema(dsn)
        except Exception as e:
            _log("schema introspection failed: {}".format(traceback.format_exc()))
            return _degrade_response(["schema introspection failed: {}".format(e)],
                                     cfg, t0, 0, eff_budget, recommend_only)

    workload, parse_notes = parse_workload(workload_arg, STATE.schema_columns,
                                           query_classes=cfg["obs_query_classes"])
    num_queries_analyzed = len(workload.queries)
    degrade_notes = list(parse_notes)
    if num_queries_analyzed == 0:
        return _degrade_response(degrade_notes or ["workload yielded no indexable queries"],
                                 cfg, t0, num_queries_analyzed, eff_budget,
                                 recommend_only)

    eval_dsn_resolved = STATE.eval_dsn
    eval_mode = "standby" if eval_dsn and eval_dsn != dsn else "hypothetical"
    try:
        env, action_manager = _make_env(
            eval_dsn_resolved, workload, budget_mb, max_indexes, cfg, EnvironmentType.TESTING,
            schema_columns=STATE.schema_columns,
        )
    except Exception as e:
        _log("env build failed: {}".format(traceback.format_exc()))
        return _degrade_response(degrade_notes + ["env build failed: {}".format(e)],
                                 cfg, t0, num_queries_analyzed, eff_budget,
                                 recommend_only)

    # Reuse the trained model only when its dimensions match this env.
    meta = _model_meta()
    try:
        cur_n_actions = env.action_space.n
        cur_n_features = env.observation_space.shape[0]
    except Exception:
        cur_n_actions = cur_n_features = None
    meta_match = (meta is not None and cur_n_actions is not None
                  and meta.get("max_index_width") == cfg["max_index_width"]
                  and meta.get("n_actions") == cur_n_actions
                  and meta.get("n_features") == cur_n_features)
    # Validate the in-memory model against this env's dimensions.
    if STATE.model is not None:
        try:
            in_mem_match = (STATE.model.action_space.n == cur_n_actions and
                            tuple(STATE.model.observation_space.shape) == (cur_n_features,))
        except Exception:
            in_mem_match = False
        if not in_mem_match:
            _log("in-memory model does not match current env dimensions; discarding")
            STATE.model = None
            STATE.vec_normalize = None
    if STATE.model is not None and not meta_match:
        _log("trained model does not match current workload/config; bootstrapping fresh")
        STATE.model = None
        STATE.vec_normalize = None
    if STATE.model is None and os.path.exists(os.path.join(STATE.model_dir, "model.zip")) and meta_match:
        _load_model_from_disk()
        if STATE.model is not None:
            try:
                sp_match = (STATE.model.action_space.n == cur_n_actions and
                            tuple(STATE.model.observation_space.shape) == (cur_n_features,))
            except Exception:
                sp_match = False
            if not sp_match:
                STATE.model = None
                STATE.vec_normalize = None
    model_was_loaded = STATE.model is not None
    if STATE.model is None:
        _log("bootstrapping fresh PPO2 model (cold start)")
        try:
            model, venv = _bootstrap_model(env, cfg)
            STATE.model = model
            STATE.vec_normalize = venv
        except Exception as e:
            _log("bootstrap failed: {}".format(traceback.format_exc()))
            _close_env(env)
            return _degrade_response(degrade_notes + ["model bootstrap failed: {}".format(e)],
                                     cfg, t0, num_queries_analyzed, eff_budget,
                                     recommend_only)
    model = STATE.model
    model_reused = model_was_loaded

    rec_indexes, episode_note, achieved_cost = _run_episode(model, env, cfg, deadline)
    _close_env(env)
    env = None
    if episode_note:
        degrade_notes.append(episode_note)

    existing, existing_names = existing_index_set(dsn)
    col_types = _column_types(dsn)
    rec_list, drop_notes = _indexes_to_ddl(rec_indexes, existing, existing_names, dsn,
                                           budget_mb, max_indexes, write_overhead_pct,
                                           col_types=col_types)
    degrade_notes.extend(drop_notes)

    apply_results = None
    if apply:
        apply_results = _apply_ddl(dsn, rec_list)

    _persist_last_workload(workload_arg)
    mode = "inference-only" if recommend_only else "online-training"
    # train_trigger counts online-mode requests only.
    if not recommend_only:
        with STATE_LOCK:
            _untrained_counter["value"] += 1
        _maybe_spawn_training()

    optimization_time = round(time.time() - t0, 3)
    metadata = {
        "strategy_type": "learned",
        "optimization_time": optimization_time,
        "num_queries_analyzed": num_queries_analyzed,
        "budget_used": _budget_used(eff_budget, rec_list),
        "eval_mode": eval_mode,
        "mode": mode,
        "training_trigger": cfg["train_trigger"],
        "untrained_since_last_train": _untrained_counter["value"],
    }
    if achieved_cost is not None:
        # achieved_cost = current/initial workload cost * 100 (what-if estimates)
        metadata["estimated_impact"] = round(100.0 - achieved_cost, 3)
        metadata["achieved_cost_pct"] = round(achieved_cost, 3)
    if degrade_notes:
        metadata["degrade_note"] = "; ".join(degrade_notes)
    if not rec_list and "degrade_note" not in metadata:
        metadata["degrade_note"] = "no beneficial index produced"

    response = {"recommended_indexes": rec_list, "metadata": metadata}
    if apply_results is not None:
        response["apply_results"] = apply_results
    with STATE_LOCK:
        STATE.last_recommend = response
    _log("[req] dsn={} queries={} recs={} model_reused={} "
         "budget[storage_mb={}, max_indexes={}, write_pct={}] time={:.2f}s".format(
             (dsn or "").rsplit("/", 1)[-1], num_queries_analyzed, len(rec_list),
             model_reused, eff_budget.get("storage_mb"),
             eff_budget.get("max_indexes"),
             eff_budget.get("max_write_overhead_pct"), time.time() - t0))
    return response


def _run_episode(model, env, cfg, deadline=None):
    from stable_baselines.common.vec_env import DummyVecEnv, VecNormalize
    venv = VecNormalize(DummyVecEnv([lambda: env]), norm_obs=True, norm_reward=False, training=False)
    if STATE.vec_normalize is not None:
        try:
            # Reuse normalization stats only for a matching observation shape.
            if np.shape(STATE.vec_normalize.obs_rms.mean) == np.shape(venv.obs_rms.mean):
                venv.obs_rms = copy.deepcopy(STATE.vec_normalize.obs_rms)
                venv.ret_rms = copy.deepcopy(STATE.vec_normalize.ret_rms)
        except Exception:
            pass
    try:
        obs = venv.reset()
        indexes = set()
        achieved_cost = None
        no_action_note = None
        for _ in range(cfg["max_steps_per_episode"] + 5):
            if deadline is not None and time.time() > deadline:
                no_action_note = ("time budget of {}s exceeded; returning the best-so-far "
                                  "configuration ({} indexes)".format(
                                      cfg.get("time_budget_seconds"), len(indexes)))
                break
            try:
                remaining = list(env.action_manager._remaining_valid_actions)
            except Exception:
                remaining = []
            if not remaining:
                no_action_note = ("no valid candidate action remains "
                                  "(workload candidates already indexed, or budget over-tight)")
                break
            try:
                va = env.valid_actions
            except Exception:
                va = None
            action, _ = model.predict(obs, deterministic=True,
                                       action_mask=[va] if va is not None else None)
            a = int(action[0]) if isinstance(action, (np.ndarray, list)) else int(action)
            try:
                obs, _r, dones, _infos = venv.step([a])
            except Exception as e:
                _log("episode step failed:\n{}".format(traceback.format_exc()))
                no_action_note = "episode step failed: {}: {}".format(type(e).__name__, e)
                break
            try:
                if env.episode_performances:
                    # Appended before the VecEnv auto-reset, so it survives.
                    perf = env.episode_performances[-1]
                    indexes = set(perf.get("indexes",
                                           perf.get("indexes_creation_order", set())))
                    if perf.get("achieved_cost") is not None:
                        achieved_cost = float(perf["achieved_cost"])
            except Exception:
                indexes = set(getattr(env, "current_indexes", set()))
            if dones[0]:
                break
        if no_action_note and not indexes:
            return set(), no_action_note, achieved_cost
        if not indexes:
            indexes = set(getattr(env, "current_indexes", set()))
        return indexes, None, achieved_cost
    except Exception as e:
        _log("episode run failed: {}".format(traceback.format_exc()))
        try:
            om = env.observation_manager
            wl_debug = [(q.nr, len(q.columns)) for q in env.current_workload.queries]
            _log("episode debug: n_actions={} n_columns={} n_features={} "
                 "K={} n_queries={} wl_debug={}".format(
                     getattr(env.action_manager, "number_of_actions", None),
                     getattr(om, "number_of_actions", None),
                     getattr(om, "number_of_features", None),
                     getattr(om, "number_of_query_classes", None),
                     len(env.current_workload.queries), wl_debug[:3]))
        except Exception:
            pass
        try:
            indexes = set(getattr(env, "current_indexes", set()))
        except Exception:
            indexes = set()
        return indexes, "episode terminated early: {}: {}".format(type(e).__name__, e), None


def _indexes_to_ddl(indexes, existing_set, existing_names, dsn, budget_mb, max_indexes, write_pct,
                    col_types=None):
    out = []
    notes = []
    seen = set()
    for idx in sorted(indexes, key=lambda i: (i.table().name, [c.name for c in i.columns])):
        table = idx.table().name
        cols = [c.name for c in idx.columns]
        cols_t = tuple(c.lower() for c in cols)
        key = (table, cols_t)
        if key in seen or key in existing_set:
            notes.append("{} on {}({}): duplicates an existing index; skipped".format(idx.index_idx(), table, ",".join(cols)))
            continue
        seen.add(key)
        buildable, build_note = _index_buildable(dsn, table, cols, col_types)
        if not buildable:
            notes.append(build_note)
            continue
        idx_name = idx.index_idx()
        base = idx_name
        n = 1
        while idx_name in existing_names:
            n += 1
            idx_name = "{}_{}".format(base, n)
        size_bytes = idx.estimated_size or 0
        storage_mb_val = round(size_bytes / 1024.0 / 1024.0, 4) if size_bytes else None
        who = _estimate_write_overhead(table, dsn, size_bytes)
        if budget_mb is not None and storage_mb_val is not None and storage_mb_val > budget_mb:
            notes.append("{}: single-index size {}MB exceeds storage budget; dropped".format(idx_name, storage_mb_val))
            continue
        if write_pct is not None and who is not None and who > write_pct:
            notes.append("{}: write overhead {:.1f}% exceeds limit; dropped".format(idx_name, who))
            continue
        if len(out) >= max_indexes:
            notes.append("max_indexes ({}) reached; remaining dropped".format(max_indexes))
            break
        ddl = "CREATE INDEX {} ON {} ({})".format(idx_name, table, ", ".join(cols))
        existing_names.add(idx_name)
        entry = {"ddl": ddl, "table": table, "columns": cols, "index_name": idx_name}
        if storage_mb_val is not None:
            entry["storage_mb"] = storage_mb_val
        if who is not None:
            entry["write_overhead_estimate"] = round(who, 3)
        out.append(entry)
    return out, notes


def _column_types(dsn):
    """(table, column) -> (data_type, character_maximum_length), re-read per
    request (cheap single query; used by the buildability probe)."""
    out = {}
    try:
        conn = psycopg2.connect(dsn)
        conn.autocommit = True
        cur = conn.cursor()
        cur.execute(
            "SELECT table_name, column_name, data_type, character_maximum_length "
            "FROM information_schema.columns WHERE table_schema='public'"
        )
        for tname, cname, dtype, cml in cur.fetchall():
            out[(tname.lower(), cname.lower())] = (dtype, cml)
        cur.close()
        conn.close()
    except Exception as e:
        _log("column type introspection failed: {}".format(e))
    return out


# Btree key-size limit; HypoPG does not validate buildability.
_BTREE_KEY_LIMIT = 2700
_max_bytes_cache = {}


def _max_bytes(dsn, table, col):
    """Cached max octet_length of a column, for buildability probing."""
    key = (dsn, table, col)
    if key in _max_bytes_cache:
        return _max_bytes_cache[key]
    val = _BTREE_KEY_LIMIT + 1
    try:
        conn = psycopg2.connect(dsn)
        conn.autocommit = True
        cur = conn.cursor()
        cur.execute("set statement_timeout = 60000")
        cur.execute('select max(octet_length("{}")) from "{}"'.format(col, table))
        row = cur.fetchone()
        val = (row[0] or 0) if row else 0
        cur.execute("set statement_timeout = 0")
        cur.close()
        conn.close()
    except Exception:
        # Timeout / error -> assume large (conservative skip) so we never emit a
        # DDL that would fail at CREATE INDEX time.
        val = _BTREE_KEY_LIMIT + 1
    _max_bytes_cache[key] = val
    return val


def _index_buildable(dsn, table, cols, col_types):
    """(buildable, reason). Skip indexes whose key columns include an unbounded
    text-family column with a value exceeding the btree index-row limit."""
    if not col_types:
        return True, None
    for c in cols:
        dtype, cml = col_types.get((table, c.lower()), (None, None))
        if dtype in ("text", "bytea", "json", "xml") or (
            dtype == "character varying" and cml is None
        ):
            if _max_bytes(dsn, table, c.lower()) > _BTREE_KEY_LIMIT:
                return False, (
                    "index on {}({}) has unbounded {} column '{}' exceeding "
                    "btree key-size limit; skipped".format(table, ",".join(cols), dtype, c)
                )
    return True, None


def _estimate_write_overhead(table, dsn, index_size_bytes):
    if index_size_bytes is None or index_size_bytes <= 0:
        return None
    try:
        conn = psycopg2.connect(dsn)
        conn.autocommit = True
        cur = conn.cursor()
        cur.execute("SELECT pg_total_relation_size(%s)", (table,))
        row = cur.fetchone()
        cur.close()
        conn.close()
        table_bytes = row[0] if row and row[0] else 0
        if table_bytes <= 0:
            return None
        ratio = index_size_bytes / float(table_bytes)
        # Base 1% maintenance + 5x the index/table size ratio, capped at 100.
        return min(100.0, 1.0 + ratio * 5.0)
    except Exception:
        return None


def _budget_used(requested, rec_list):
    consumed_mb = sum((e.get("storage_mb") or 0) for e in rec_list)
    who_max = max([e.get("write_overhead_estimate") or 0 for e in rec_list], default=0)
    return {
        "storage_mb": (requested or {}).get("storage_mb"),
        "max_indexes": (requested or {}).get("max_indexes"),
        "max_write_overhead_pct": (requested or {}).get("max_write_overhead_pct"),
        "consumed_storage_mb": round(consumed_mb, 4),
        "consumed_indexes": len(rec_list),
        "consumed_write_overhead_pct": round(who_max, 3),
    }


def _apply_ddl(dsn, rec_list):
    results = []
    conn = psycopg2.connect(dsn)
    conn.autocommit = True
    cur = conn.cursor()
    for e in rec_list:
        try:
            cur.execute(e["ddl"])
            results.append({"index_name": e["index_name"], "status": "ok"})
        except Exception as ex:
            conn.rollback()
            results.append({"index_name": e["index_name"], "status": "error", "error": str(ex)})
    cur.close()
    conn.close()
    return results


def _degrade_response(notes, cfg, t0, num_queries, budget, recommend_only=False):
    metadata = {
        "strategy_type": "learned",
        "optimization_time": round(time.time() - t0, 3),
        "num_queries_analyzed": num_queries,
        "budget_used": _budget_used(budget, []),
        "mode": "inference-only" if recommend_only else "online-training",
        "degrade_note": "; ".join(notes) if notes else "no recommendation produced",
    }
    return {"recommended_indexes": [], "metadata": metadata}


# --------------------------------------------------------------------------- #
# Background training.                                                        #
# --------------------------------------------------------------------------- #
def _persist_last_workload(workload_arg):
    try:
        path = os.path.join(STATE.model_dir, "last_workload.json")
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump({"workload": workload_arg}, f)
        os.replace(tmp, path)
    except Exception as e:
        _log("could not persist last workload: {}".format(e))


def _maybe_spawn_training():
    with STATE_LOCK:
        if _training_in_progress["value"]:
            return
        if _untrained_counter["value"] < STATE.config["train_trigger"]:
            return
        _training_in_progress["value"] = True
    wrapper_path = os.path.abspath(__file__)
    cmd = [
        sys.executable, wrapper_path, "--train",
        "--model-dir", STATE.model_dir,
        "--dsn", STATE.dsn,
        "--eval-dsn", STATE.eval_dsn,
        "--timesteps", str(STATE.config["timesteps"]),
        "--config-json", json.dumps(STATE.config),
    ]
    try:
        log_path = os.path.join(STATE.model_dir, "training.log")
        with open(log_path, "a") as logf:
            proc = subprocess.Popen(cmd, stdout=logf, stderr=logf, cwd=REPO,
                                    env=dict(os.environ,
                                             PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION="python",
                                             TF_CPP_MIN_LOG_LEVEL="3"))
        threading.Thread(target=_training_waiter, args=(proc,), daemon=True).start()
    except Exception as e:
        _log("training spawn failed: {}".format(e))
        with STATE_LOCK:
            _training_in_progress["value"] = False


def _training_waiter(proc):
    try:
        proc.wait()
    finally:
        with STATE_LOCK:
            _training_in_progress["value"] = False
            _untrained_counter["value"] = 0


def run_training_subprocess(model_dir, dsn, eval_dsn, timesteps_override=None,
                            config_json=None):
    _log("training subprocess start (model_dir={})".format(model_dir))
    cfg = dict(DEFAULTS)
    if config_json:
        try:
            extra = json.loads(config_json)
            if isinstance(extra, dict):
                cfg.update(extra)
        except Exception:
            _log("training: could not parse config-json; using defaults")
    cfg["model_dir"] = model_dir
    if timesteps_override is not None:
        cfg["timesteps"] = timesteps_override
    os.makedirs(model_dir, exist_ok=True)

    wl_path = os.path.join(model_dir, "last_workload.json")
    workload_arg = None
    if os.path.exists(wl_path):
        try:
            with open(wl_path) as f:
                workload_arg = json.load(f).get("workload")
        except Exception:
            workload_arg = None

    tables, columns = introspect_schema(dsn)
    if not columns:
        _log("training: schema empty; aborting")
        return
    workload, _ = parse_workload(workload_arg or "select 1", columns,
                                 query_classes=cfg["obs_query_classes"])
    if not workload.queries:
        _log("training: no indexable queries in workload; aborting")
        return

    training_workloads = _sample_training_workloads(workload, cfg)
    if not training_workloads:
        _log("training: could not build training workloads; aborting")
        return

    first_wl = training_workloads[0]
    env, _ = _make_env(eval_dsn or dsn, first_wl, None, cfg["max_steps_per_episode"],
                       cfg, EnvironmentType.TRAINING, candidate_workload=workload,
                       schema_columns=columns)
    env.workloads = training_workloads

    try:
        meta = {
            "max_index_width": cfg["max_index_width"],
            "n_actions": env.action_space.n,
            "n_features": env.observation_space.shape[0],
            "obs_query_classes": cfg["obs_query_classes"],
            "candidate_scope": "schema" if int(cfg["max_index_width"]) == 1 else "workload",
            "candidate_columns": ["{}.{}".format(c.table.name, c.name)
                                  for c in _universe_columns(dsn, columns, cfg)]
                                 if int(cfg["max_index_width"]) == 1 else None,
            "timestamp": time.time(),
        }
        mp = os.path.join(model_dir, "model_meta.json")
        tmp = mp + ".tmp"
        with open(tmp, "w") as f:
            json.dump(meta, f)
        os.replace(tmp, mp)
    except Exception as e:
        _log("could not persist model_meta: {}".format(e))

    model_path = os.path.join(model_dir, "model.zip")
    if os.path.exists(model_path):
        try:
            from stable_baselines import PPO2
            from stable_baselines.common.vec_env import DummyVecEnv, VecNormalize
            venv = VecNormalize(DummyVecEnv([lambda: env]), norm_obs=True, norm_reward=True,
                                 gamma=0.5, training=True)
            # Restore the training normalization statistics.
            try:
                import pickle
                with open(os.path.join(model_dir, "vecnormalize.pkl"), "rb") as fh:
                    saved_norm = pickle.load(fh)
                if np.shape(saved_norm.obs_rms.mean) == np.shape(venv.obs_rms.mean):
                    venv.obs_rms = copy.deepcopy(saved_norm.obs_rms)
                    venv.ret_rms = copy.deepcopy(saved_norm.ret_rms)
            except Exception:
                pass
            model = PPO2.load(model_path, env=venv)
            _log("training: loaded existing model for further training")
        except Exception as e:
            _log("training: load failed ({}); bootstrapping fresh".format(e))
            model, venv = _bootstrap_model(env, cfg)
    else:
        model, venv = _bootstrap_model(env, cfg)

    params_before = list(model.get_parameters().values())
    norm_before = float(np.sqrt(sum(float(np.sum(np.square(v))) for v in params_before)))

    # Capture per-update PPO losses.
    losses = []
    achieved = []
    _PPO2 = model.__class__
    _orig_train_step = _PPO2._train_step
    def _patched_train_step(self, *a, **kw):
        res = _orig_train_step(self, *a, **kw)
        try:
            losses.append((float(res[0]), float(res[1])))
        except Exception:
            pass
        try:
            for plist in self.env.get_attr("episode_performances"):
                if plist:
                    achieved.append(float(plist[-1].get("achieved_cost", 0)))
        except Exception:
            pass
        return res
    _PPO2._train_step = _patched_train_step

    t0 = time.time()
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            model.learn(total_timesteps=cfg["timesteps"], callback=lambda l, g: True,
                        reset_num_timesteps=False)
    except Exception as e:
        _log("training: learn failed: {}".format(traceback.format_exc()))
        _PPO2._train_step = _orig_train_step
        return
    finally:
        _PPO2._train_step = _orig_train_step

    params_after = list(model.get_parameters().values())
    norm_after = float(np.sqrt(sum(float(np.sum(np.square(v))) for v in params_after)))
    weights_changed = abs(norm_after - norm_before) > 1e-9

    tmp_dir = tempfile.mkdtemp(dir=model_dir)
    try:
        model.save(os.path.join(tmp_dir, "model.zip"))
        venv.save(os.path.join(tmp_dir, "vecnormalize.pkl"))
        for name in ("model.zip", "vecnormalize.pkl"):
            os.replace(os.path.join(tmp_dir, name), os.path.join(model_dir, name))
        sanity = {
            "timestamp": time.time(),
            "timesteps": cfg["timesteps"],
            "duration_sec": round(time.time() - t0, 3),
            "weights_norm_before": norm_before,
            "weights_norm_after": norm_after,
            "weights_changed": weights_changed,
            "policy_loss_samples": [round(x[0], 6) for x in losses[-20:]],
            "value_loss_samples": [round(x[1], 6) for x in losses[-20:]],
            "achieved_cost_samples": achieved[-20:],
            "status": "ok",
        }
        with open(os.path.join(tmp_dir, "training_loss.json"), "w") as f:
            json.dump(sanity, f, indent=2)
        os.replace(os.path.join(tmp_dir, "training_loss.json"),
                   os.path.join(model_dir, "training_loss.json"))
        _log("training complete: weights_changed={}, loss_updates={}, achieved_n={}, saved to {}".format(
            weights_changed, len(losses), len(achieved), model_dir))
    finally:
        try:
            os.rmdir(tmp_dir)
        except Exception:
            pass


def _sample_training_workloads(workload, cfg):
    import random
    rng = random.Random(cfg["random_seed"])
    qs = workload.queries
    if not qs:
        return []
    size = max(2, min(cfg["training_workload_size"], len(qs)))
    out = []
    for _ in range(cfg["training_workloads"]):
        chosen = list(qs) if len(qs) <= size else rng.sample(qs, size)
        queries = []
        for i, q in enumerate(chosen):
            queries.append(Query(i + 1, q.text, columns=list(q.columns),
                                 frequency=rng.randint(1, 5)))
        out.append(Workload(queries))
    return out


# --------------------------------------------------------------------------- #
# HTTP server.                                                                #
# --------------------------------------------------------------------------- #
class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        _log(" ".join(str(a) for a in args))

    def _send(self, code, obj):
        body = json.dumps(obj).encode()
        try:
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def do_GET(self):
        if self.path.startswith("/health"):
            self._send(200, {"status": "ok", "model_loaded": STATE.model is not None})
            return
        if self.path.startswith("/state"):
            self._send(200, _state_summary())
            return
        self._send(404, {"error": "not found"})

    def do_POST(self):
        if self.path.startswith("/shutdown"):
            _handle_shutdown()
            self._send(200, {"status": "shutting down"})
            threading.Thread(target=lambda: (time.sleep(0.3), os._exit(0)), daemon=True).start()
            return
        if self.path.startswith("/recommend"):
            self._handle_recommend()
            return
        self._send(404, {"error": "not found"})

    def _handle_recommend(self):
        length = int(self.headers.get("Content-Length", "0"))
        try:
            raw = self.rfile.read(length) if length else b"{}"
        except (BrokenPipeError, ConnectionResetError):
            return
        try:
            req = json.loads(raw.decode() or "{}")
        except Exception as e:
            self._send(400, {"error": "bad json: {}".format(e)})
            return
        try:
            if _shutdown_requested["value"]:
                self._send(503, {"error": "shutting down"})
                return
            resp = recommend(
                dsn=req.get("dsn"),
                eval_dsn=req.get("eval_dsn"),
                workload_arg=req.get("workload"),
                budget=_as_dict(req.get("budget")),
                config=_as_dict(req.get("config")),
                recommend_only=bool(req.get("recommend_only", False)),
                apply=bool(req.get("apply", False)),
            )
            self._send(200, resp)
        except Exception as e:
            _log("/recommend error: {}".format(traceback.format_exc()))
            self._send(200, {
                "recommended_indexes": [],
                "metadata": {"degrade_note": "server error: {}: {}".format(type(e).__name__, e)},
            })


def _state_summary():
    with STATE_LOCK:
        return {
            "dsn": STATE.dsn,
            "eval_dsn": STATE.eval_dsn,
            "model_loaded": STATE.model is not None,
            "model_dir": STATE.model_dir,
            "model_mtime": STATE.model_mtime,
            "untrained_since_last_train": _untrained_counter["value"],
            "training_in_progress": _training_in_progress["value"],
            "idle": (not _training_in_progress["value"]
                     and _untrained_counter["value"] < STATE.config.get("train_trigger", 1)),
            "last_recommend": STATE.last_recommend,
            "config": STATE.config,
        }


def _handle_shutdown():
    _shutdown_requested["value"] = True
    _persist_state()


def _persist_state():
    got = False
    try:
        got = PERSIST_LOCK.acquire(blocking=False)
        if not got:
            return
        path = os.path.join(STATE.model_dir, "counters.json")
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump({
                "untrained_counter": _untrained_counter["value"],
                "dsn": STATE.dsn,
                "eval_dsn": STATE.eval_dsn,
                "last_recommend": STATE.last_recommend,
            }, f, indent=2)
        os.replace(tmp, path)
        _log("state persisted")
    except Exception as e:
        _log("persist failed: {}".format(e))
    finally:
        if got:
            PERSIST_LOCK.release()


def _rehydrate():
    try:
        path = os.path.join(STATE.model_dir, "counters.json")
        if os.path.exists(path):
            with open(path) as f:
                d = json.load(f)
            _untrained_counter["value"] = d.get("untrained_counter", 0)
            _log("rehydrated counters (untrained={})".format(_untrained_counter["value"]))
    except Exception as e:
        _log("rehydrate failed: {}".format(e))


def _arg(argv, name):
    if name in argv:
        i = argv.index(name)
        if i + 1 < len(argv):
            return argv[i + 1]
    return None


def main():
    if "--train" in sys.argv:
        model_dir = _arg(sys.argv, "--model-dir") or DEFAULTS["model_dir"]
        dsn = _arg(sys.argv, "--dsn")
        eval_dsn = _arg(sys.argv, "--eval-dsn") or dsn
        timesteps = _arg(sys.argv, "--timesteps")
        try:
            timesteps = int(timesteps) if timesteps else None
        except Exception:
            timesteps = None
        os.makedirs(model_dir, exist_ok=True)
        run_training_subprocess(model_dir, dsn, eval_dsn, timesteps,
                                config_json=_arg(sys.argv, "--config-json"))
        return

    port = int(os.environ.get("PORT", "7080"))
    server = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    server.daemon_threads = True
    STATE.model_dir = DEFAULTS["model_dir"]
    os.makedirs(STATE.model_dir, exist_ok=True)
    _rehydrate()
    _log("SWIRL index-selection wrapper listening on :{}".format(port))
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        _persist_state()


if __name__ == "__main__":
    main()
