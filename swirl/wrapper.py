#!/usr/bin/env python3
# SWIRL index-selection skill wrapper (long-lived HTTP server).
#
# Wraps github.com/hyrise/rl_index_selection (SWIRL: Selection of Workload-aware
# Indexes using Reinforcement Learning) into the index_selection skill I/O spec.
# Endpoints: GET /health, POST /recommend, POST /shutdown, GET /state.
#
# The recommendation path runs a trained/bootstrapped PPO2 policy through a
# SWIRL gym_db episode to obtain a recommended index configuration. What-if cost
# evaluation uses HypoPG hypothetical indexes on --eval-dsn (defaults to --dsn);
# no real indexes are created on --dsn unless --apply is set. Online training
# (PPO2.learn) runs in a background subprocess on the eval DSN and is
# infrequently triggered; the foreground recommendation is cost-only and fast.

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
# Source repairs (minimal; core RL logic untouched).                          #
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
from index_selection_evaluation.selection.index import Index
from index_selection_evaluation.selection.cost_evaluation import CostEvaluation
from index_selection_evaluation.selection import candidate_generation as _cand
import swirl.utils as _swirl_utils
from gym_db.common import EnvironmentType

# Repair 1: accept a full DSN (the source uses `psycopg2.connect("dbname=...")`
# which cannot reach a non-local socket / custom port / credentials). The skill
# passes the full DSN through db_name; connect with it verbatim when it looks
# like a DSN.
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

# Repair 2: never drop real indexes on the user's online DSN. The source env
# calls connector.drop_indexes() at construction to start from a clean slate on
# its own managed benchmark DBs; against the skill's external --dsn this would
# destroy the user's existing indexes (non-invasiveness violation).
def _drop_indexes_noop(self):
    _log("drop_indexes() neutered (non-invasive on user DSN)")
_PGC.drop_indexes = _drop_indexes_noop

# Repair 3: CREATE EXTENSION IF NOT EXISTS (the source's `create extension
# hypopg` errors once the extension already exists).
def _enable_simulation_safe(self):
    self.exec_only("CREATE EXTENSION IF NOT EXISTS hypopg")
    self.commit()
_PGC.enable_simulation = _enable_simulation_safe

# Repair 4: guard the absent hypopg_list_indexes() (only used by debug helpers).
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
    "max_index_width": 1,
    "max_steps_per_episode": 10,
    "train_trigger": 5,
    "timesteps": 2000,
    "parallel_environments": 1,
    "training_workloads": 20,
    "training_workload_size": 10,
    "connection_pool_size": 4,
    "action_manager": "MultiColumnIndexActionManager",
    "observation_manager": "SingleColumnIndexObservationManager",
    "reward_calculator": "RelativeDifferenceRelativeToStorageReward",
    "reenable_indexes": False,
    "random_seed": 60,
    "obs_query_classes": 256,
}

STATE_LOCK = threading.RLock()
PERSIST_LOCK = threading.Lock()
# Serializes /recommend: a request resets/swaps shared STATE (model,
# schema_columns, config) while another request's episode may still be
# running; concurrent episodes would race that shared state.
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
# Schema introspection (replaces the source's benchmark-specific schema/data  #
# generation, which manages its own benchmark databases).                    #
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


def _parse_indexdef_columns(indexdef):
    m = re.search(r"\bON\b\s+\S+\s+USING\s+\w+\s*\((.*)\)\s*$", indexdef, re.IGNORECASE)
    if not m:
        m = re.search(r"\bON\b\s+\S+\s*\((.*)\)\s*$", indexdef, re.IGNORECASE)
    if not m:
        return []
    inner = m.group(1)
    parts = re.split(r",", inner)
    cols = []
    for p in parts:
        p = p.strip()
        p = re.split(r"\s+(COLLATE|DESC|ASC|NULLS|_pattern_ops|_ops|gist_|gin_|spgist_|brin_)\b", p)[0]
        p = p.strip().strip('"')
        if p:
            cols.append(p)
    return cols


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
        # Query numbers are pinned to the observation manager's fixed
        # query-class width; the frequency vector is zero-padded to it.
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
# SWIRL component construction.                                               #
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
    return combos, flat


def _build_action_manager(combos, flat, dsn, config):
    consumptions = _swirl_utils.predict_index_sizes(flat, dsn)
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
    # Fixed observation dimension so a model trained on one workload can predict
    # for another (the query-class frequency vector is zero-padded to this width).
    return cls(
        action_manager.number_of_actions,
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
                 candidate_workload=None):
    from gym_db.envs.db_env_v1 import DBEnvV1
    cand_src = candidate_workload if candidate_workload is not None else workload
    combos, flat = _build_candidates(cand_src.queries, config["max_index_width"])
    action_manager = _build_action_manager(combos, flat, dsn, config)
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

    _make_env builds a DBEnvV1 that opens its own PostgresDatabaseConnector
    (one DB session per /recommend). Previously nothing ever closed it, so
    every request leaked an idle backend. Cleanup mirrors what the source's
    SelectionAlgorithm does at the end of a run: drop the simulated
    hypothetical indexes via complete_cost_estimation(), then close the
    connector so the session (and all server-side HypoPG state) dies with
    it. Idempotent and failure-tolerant; only called on envs the request is
    finished with (STATE.model / STATE.vec_normalize only reuse the model
    weights and obs stats, never this connector).
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
# Knobs that can be changed per request without invalidating the in-memory
# model (they do not resize the action/observation spaces).
_PER_REQUEST_KNOBS = ("max_steps_per_episode", "train_trigger", "timesteps",
                      "training_workloads", "training_workload_size", "random_seed")


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
    # A dimension change (index width / query-class width), a DSN/eval-DSN
    # change, or a model_dir switch invalidates the in-memory model/state.
    changed = (STATE.dsn != dsn
               or STATE.eval_dsn != (eval_dsn or dsn)
               or STATE.config.get("max_index_width") != width
               or STATE.config.get("obs_query_classes") != query_classes
               or STATE.model_dir != model_dir)
    if changed:
        STATE.dsn = dsn
        STATE.eval_dsn = eval_dsn or dsn
        STATE.config = dict(DEFAULTS)
        STATE.config.update(config)
        # The SWIRL manager/reward classes are fixed parts of the skill, not
        # user-selectable knobs.
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
    else:
        # Non-dimension knobs apply per request.
        for key in _PER_REQUEST_KNOBS:
            if key in config:
                try:
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
        # Cheap signature pre-check: the persisted workload's candidate count
        # must match the trained model's recorded dimensions.
        # last_workload.json is rewritten on every request but model.zip only
        # on training, so they drift — building an env for a mismatched
        # workload wastes a full size-estimation pass and fails late inside
        # PPO2.load.
        meta = _model_meta()
        if meta is not None:
            _, flat_chk = _build_candidates(workload.queries, STATE.config["max_index_width"])
            if meta.get("n_actions") != len(flat_chk):
                _log("persisted workload ({} candidates) does not match trained model "
                     "({} actions); bootstrapping fresh".format(len(flat_chk), meta.get("n_actions")))
                STATE.model = None
                STATE.vec_normalize = None
                return
        env, _ = _make_env(STATE.eval_dsn or STATE.dsn, workload, None,
                            STATE.config["max_steps_per_episode"], STATE.config,
                            EnvironmentType.TESTING)
        venv = VecNormalize(DummyVecEnv([lambda: env]), norm_obs=True, norm_reward=False, training=False)
        # Restore the normalization statistics the model was trained with.
        # VecNormalize.load is a STATICMETHOD (load_path, venv) — the
        # venv.load(path) form raises and the stats were silently never
        # restored (predictions then ran on unnormalized observations after a
        # restart). Copy obs/ret rms from the pickled object instead.
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


def _dummy_env_for_spaces(dsn, config):
    if STATE.schema_columns is None:
        STATE.schema_tables, STATE.schema_columns = introspect_schema(dsn)
    if STATE.schema_columns:
        real = STATE.schema_columns[0]
        q = Query(1, "select * from {} where {} = 1".format(real.table.name, real.name), columns=[real])
        env, _ = _make_env(dsn, Workload([q]), None, 1, config, EnvironmentType.TESTING)
        return env
    raise RuntimeError("schema has no columns")


# --------------------------------------------------------------------------- #
# Recommendation.                                                             #
# --------------------------------------------------------------------------- #
def recommend(dsn, eval_dsn, workload_arg, budget, config, recommend_only, apply):
    # Serialized: a request resets/swaps shared STATE (model, schema_columns,
    # config) that another in-flight request's episode may still be using.
    with RECOMMEND_LOCK:
        return _recommend_locked(dsn, eval_dsn, workload_arg, budget, config,
                                 recommend_only, apply)


def _recommend_locked(dsn, eval_dsn, workload_arg, budget, config, recommend_only, apply):
    t0 = time.time()
    _ensure_state_for_dsn(dsn, eval_dsn, config or {})
    cfg = STATE.config
    STATE.reload_if_changed()

    if STATE.schema_columns is None:
        _log("introspecting schema for {}".format(dsn))
        STATE.schema_tables, STATE.schema_columns = introspect_schema(dsn)

    workload, parse_notes = parse_workload(workload_arg, STATE.schema_columns,
                                           query_classes=cfg["obs_query_classes"])
    num_queries_analyzed = len(workload.queries)
    degrade_notes = list(parse_notes)
    if num_queries_analyzed == 0:
        return _degrade_response(degrade_notes or ["workload yielded no indexable queries"],
                                 cfg, t0, num_queries_analyzed, budget)

    b = budget or {}
    storage_mb = b.get("storage_mb")
    max_indexes = b.get("max_indexes", cfg["max_steps_per_episode"])
    write_overhead_pct = b.get("max_write_overhead_pct")
    try:
        max_indexes = int(max_indexes)
    except Exception:
        max_indexes = cfg["max_steps_per_episode"]
    budget_mb = float(storage_mb) if storage_mb else None

    eval_dsn_resolved = STATE.eval_dsn
    eval_mode = "standby" if eval_dsn and eval_dsn != dsn else "hypothetical"
    try:
        env, action_manager = _make_env(
            eval_dsn_resolved, workload, budget_mb, max_indexes, cfg, EnvironmentType.TESTING,
        )
    except Exception as e:
        _log("env build failed: {}".format(traceback.format_exc()))
        return _degrade_response(degrade_notes + ["env build failed: {}".format(e)],
                                 cfg, t0, num_queries_analyzed, budget)

    # Decide whether the persisted trained model applies to THIS workload/config
    # (same candidate count + observation width + index width). The saved
    # policy network has a fixed input size; a workload/width change means it
    # cannot predict, so bootstrap fresh rather than feed mismatched obs.
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
    # Validate the IN-MEMORY model against the current env directly — the disk
    # meta says nothing about an in-memory model that was bootstrapped fresh
    # for a different workload whose dimensions happen to match the meta (such
    # a model's obs normalization broadcasts-fail or predicts garbage).
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
        # The disk meta does not describe the in-memory model's training run;
        # drop it so a matching disk model (below) takes precedence.
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
                                     cfg, t0, num_queries_analyzed, budget)
    model = STATE.model
    model_reused = model_was_loaded

    rec_indexes, episode_note, achieved_cost = _run_episode(model, env, cfg)
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
    with STATE_LOCK:
        _untrained_counter["value"] += 1

    mode = "inference-only" if recommend_only else "online-training"
    if not recommend_only:
        _maybe_spawn_training()

    optimization_time = round(time.time() - t0, 3)
    metadata = {
        "strategy_type": "learned",
        "optimization_time": optimization_time,
        "num_queries_analyzed": num_queries_analyzed,
        "budget_used": _budget_used(b, rec_list),
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
    _log("[req] dsn={} queries={} recs={} model_reused={} budget={} time={:.2f}s".format(
        (dsn or "").rsplit("/", 1)[-1], num_queries_analyzed, len(rec_list),
        model_reused, budget_mb, time.time() - t0))
    return response


def _run_episode(model, env, cfg):
    from stable_baselines.common.vec_env import DummyVecEnv, VecNormalize
    venv = VecNormalize(DummyVecEnv([lambda: env]), norm_obs=True, norm_reward=False, training=False)
    if STATE.vec_normalize is not None:
        try:
            # Only reuse normalization stats recorded for the SAME observation
            # shape (a mismatched obs_rms broadcasts-fail on this env's obs).
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
            try:
                remaining = list(env.action_manager._remaining_valid_actions)
            except Exception:
                remaining = []
            if not remaining:
                no_action_note = "no valid candidate action remains (budget may be over-tight)"
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
                no_action_note = "episode step failed: {}: {}".format(type(e).__name__, e)
                break
            try:
                if env.episode_performances:
                    # The VecEnv auto-resets and clears env.current_indexes
                    # after done, but this entry is appended before the reset
                    # and survives. ("indexes" is the DBEnvV1 key;
                    # "indexes_creation_order" the DBEnvV3 key.)
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
            _log("episode debug: M={} K={} n_queries={} wl_debug={}".format(
                om.number_of_actions, getattr(om, "number_of_query_classes", None),
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


# PG btree index-row safety threshold (~8191 B block / ~2704 B key). HypoPG does
# NOT validate buildability: an index on an unbounded text-family column whose
# values exceed this limit fails at real CREATE INDEX (—apply) time.
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


def _degrade_response(notes, cfg, t0, num_queries, budget):
    metadata = {
        "strategy_type": "learned",
        "optimization_time": round(time.time() - t0, 3),
        "num_queries_analyzed": num_queries,
        "budget_used": _budget_used(budget, []),
        "mode": "inference-only",
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


def run_training_subprocess(model_dir, dsn, eval_dsn, timesteps_override=None):
    _log("training subprocess start (model_dir={})".format(model_dir))
    cfg = dict(DEFAULTS)
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
                       cfg, EnvironmentType.TRAINING, candidate_workload=workload)
    env.workloads = training_workloads

    try:
        meta = {
            "max_index_width": cfg["max_index_width"],
            "n_actions": env.action_space.n,
            "n_features": env.observation_space.shape[0],
            "obs_query_classes": cfg["obs_query_classes"],
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
            # Restore the training normalization statistics (see the note in
            # _load_model_from_disk: VecNormalize.load is a staticmethod, so the
            # venv.load(path) form silently never restored anything).
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

    # Capture per-update PPO loss by wrapping _train_step (returns
    # policy_loss, value_loss, ... as numpy floats).
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
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

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
        raw = self.rfile.read(length) if length else b"{}"
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
                budget=req.get("budget"),
                config=req.get("config") or {},
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
                     and _untrained_counter["value"] < STATE.config.get("train_trigger", 5)),
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
        run_training_subprocess(model_dir, dsn, eval_dsn, timesteps)
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
