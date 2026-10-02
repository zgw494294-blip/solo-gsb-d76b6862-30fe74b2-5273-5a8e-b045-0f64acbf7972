"""读取一致性回归测试：GET 返回的版本/blank/样本必须来自同一完整提交。

复现“批次读取拼接不同提交状态”：GET 先读到版本 1 及旧 blank，此时另一
SQLite 连接以 ``expected_version=1`` 合法提交版本 2，原请求再读样本——
修复前会拼出“旧版本 + 旧 blank + 新结果”（或版本与 confirmed 错配）；
修复后读事务持有 WAL 快照，只能返回完整旧状态或完整新状态。

交错是可控的：包装存储连接，在批次行 SELECT 之后暂停读取线程，待写入方
（真实 HTTP 接口、另一事务）提交后再放行。读取不取存储实例的进程锁，
写入方不受影响——锁不住另一连接的合法写入，一致性只能由读事务保证。
"""

from __future__ import annotations

import threading
from contextlib import contextmanager
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app import batches
from app.main import app


@pytest.fixture(autouse=True)
def isolated_db(tmp_path: str):
    # 每个测试一个全新的临时 SQLite 文件
    db = str(Path(tmp_path) / "batches.db")
    batches.reset_store(db)
    yield db
    batches.reset_store(db)


client = TestClient(app)

REFS_2 = [
    {"id": "A", "time_ms": 3},
    {"id": "B", "time_ms": 7},
]


def run(times, intensities, apex_indices=(1, 3), is1=0, is2=1, **over):
    body = {
        "samples": [
            {"time_ms": t, "intensity": y} for t, y in zip(times, intensities)
        ],
        "start_index": 0,
        "end_index": len(times) - 1,
        "apex_indices": list(apex_indices),
        "is_apex_idx_1": is1,
        "is_apex_idx_2": is2,
    }
    body.update(over)
    return body


SAMPLE = [0, 3, 4, 7, 10], [1, 4, 2, 5, 0]
BLANK = [0, 5, 6, 9, 14], [1, 2, 1, 3, 1]
# 空白更强的版本：A 峰面积变大，会改变全部样本的扣除结果
BLANK_STRONG = [0, 5, 6, 9, 14], [1, 6, 1, 3, 1]


def batch_payload():
    return {
        "reference_peaks": REFS_2,
        "is_ref_id_1": "A",
        "is_ref_id_2": "B",
        "tolerance_ms": 0,
        "blank": run(*BLANK),
        "samples": [
            {"id": "s1", **run(*SAMPLE)},
            {"id": "s2", **run(*SAMPLE)},
        ],
    }


def create():
    r = client.post("/v1/blank-subtraction-batches", json=batch_payload())
    assert r.status_code == 201, r.text
    return r.json()


# 批次行 SELECT 的识别片段（区别于版本校验的 SELECT version FROM batches）
BATCH_ROW_SQL = "params, blank FROM batches"


class _HookedConn:
    """委托 sqlite3.Connection：execute 后触发钩子，并记录语句首词。"""

    def __init__(self, conn, hook=None, log=None):
        self._conn = conn
        self._hook = hook
        self._log = log

    def execute(self, sql, parameters=()):
        cur = self._conn.execute(sql, parameters)
        if self._log is not None:
            self._log.append(sql.strip().split()[0].upper())
        if self._hook is not None:
            self._hook(sql)
        return cur

    def __getattr__(self, name):
        return getattr(self._conn, name)


def _patch_connect(store, hook=None, log=None):
    """把 store 的连接换成包装版（仅本实例生效），返回还原函数。"""
    real_connect = store._connect

    @contextmanager
    def patched():
        with real_connect() as conn:
            try:
                yield _HookedConn(conn, hook=hook, log=log)
            finally:
                # 退出 _connect 上下文即释放连接（其 finally 执行 close）
                if log is not None:
                    log.append("CLOSE")

    store._connect = patched
    return lambda: setattr(store, "_connect", real_connect)


def _read_paused_after_batch_row(write):
    """读取线程在批次行 SELECT 后暂停；主线程执行 write(bid) 后放行读取。

    返回 (创建响应 JSON, 读取线程的 GET 响应 JSON)。
    """
    created = create()
    bid = created["batch_id"]
    store = batches.get_store()

    row_read = threading.Event()  # 批次行已读（读事务快照已建立）
    resume = threading.Event()    # 写入方已提交，放行读取
    fired = threading.Event()     # 钩子只触发一次（读取方），写入方不受影响

    def hook(sql):
        if BATCH_ROW_SQL in sql and not fired.is_set():
            fired.set()
            row_read.set()
            assert resume.wait(timeout=10), "读取线程等待放行超时"

    _patch_connect(store, hook=hook)

    outcome: dict = {}

    def reader():
        try:
            r = client.get(f"/v1/blank-subtraction-batches/{bid}")
            outcome["status"] = r.status_code
            outcome["json"] = r.json()
        except BaseException as exc:  # 读取线程的异常回传主线程断言
            outcome["error"] = exc

    t = threading.Thread(target=reader)
    t.start()
    try:
        assert row_read.wait(timeout=10), "未等到读取线程进入读事务"
        write(bid)  # 另一连接/事务在读取暂停期间合法提交
    finally:
        resume.set()
        t.join(timeout=10)
    assert not t.is_alive(), "读取线程未结束"
    assert "error" not in outcome, outcome.get("error")
    assert outcome["status"] == 200
    return created, outcome["json"]


# ---------------------------------------------------------------------------
# 读写交错：读取只能得到完整旧状态或完整新状态
# ---------------------------------------------------------------------------

def test_read_paused_mid_read_then_replace_blank_commits_sees_complete_old_state():
    def write(bid):
        # 读取暂停期间，另一连接以 expected_version=1 合法提交版本 2
        # （WAL 下读不阻塞写；写事务正常拿到进程锁与 BEGIN IMMEDIATE）
        r = client.post(
            f"/v1/blank-subtraction-batches/{bid}/replace-blank",
            json={"blank": run(*BLANK_STRONG), "expected_version": 1},
        )
        assert r.status_code == 200, r.text

    created, got = _read_paused_after_batch_row(write)

    # 完整旧状态：版本 1、旧 blank、旧结果、全部未确认——绝不拼接新旧
    assert got == created
    assert got["version"] == 1
    assert got["blank"]["samples"][1]["intensity"] == 2
    assert [s["confirmed"] for s in got["samples"]] == [False, False]
    assert got["samples"][0]["result"]["sample_peaks"] == [
        {"ref_id": "A", "corrected_area": {"n": 3, "d": 2}},
        {"ref_id": "B", "corrected_area": {"n": 4, "d": 1}},
    ]

    # 提交之后的读取看到完整新状态：版本 2、新 blank、新结果
    fresh = client.get(f"/v1/blank-subtraction-batches/{created['batch_id']}")
    assert fresh.status_code == 200
    fresh = fresh.json()
    assert fresh["version"] == 2
    assert fresh["blank"]["samples"][1]["intensity"] == 6
    assert all(
        s["result"]["sample_peaks"][0]["corrected_area"] == {"n": 0, "d": 1}
        for s in fresh["samples"]
    )


def test_read_paused_mid_read_then_confirm_commits_sees_complete_old_state():
    def write(bid):
        r = client.post(
            f"/v1/blank-subtraction-batches/{bid}/confirm",
            json={"sample_id": "s1", "expected_version": 1},
        )
        assert r.status_code == 200, r.text

    created, got = _read_paused_after_batch_row(write)

    # 版本与确认状态来自同一提交：版本 1 必须配全部未确认
    assert got == created
    assert got["version"] == 1
    assert [s["confirmed"] for s in got["samples"]] == [False, False]

    # 提交之后的读取：版本 2 且 s1 已确认，同步前进
    fresh = client.get(f"/v1/blank-subtraction-batches/{created['batch_id']}")
    fresh = fresh.json()
    assert fresh["version"] == 2
    assert [s["confirmed"] for s in fresh["samples"]] == [True, False]


# ---------------------------------------------------------------------------
# 资源释放：读成功 / 批次不存在 / 中途异常都结束事务并关闭连接
# ---------------------------------------------------------------------------

def test_read_transaction_commits_and_closes_on_success():
    d = create()
    log: list[str] = []
    _patch_connect(batches.get_store(), log=log)

    r = client.get(f"/v1/blank-subtraction-batches/{d['batch_id']}")
    assert r.status_code == 200
    assert log == ["BEGIN", "SELECT", "SELECT", "COMMIT", "CLOSE"]


def test_read_transaction_rolls_back_and_closes_when_batch_missing():
    log: list[str] = []
    _patch_connect(batches.get_store(), log=log)

    r = client.get("/v1/blank-subtraction-batches/nope")
    assert r.status_code == 404
    assert log == ["BEGIN", "SELECT", "ROLLBACK", "CLOSE"]


def test_read_transaction_rolls_back_and_closes_on_mid_read_error():
    d = create()
    log: list[str] = []

    def boom(sql):
        if "FROM batch_samples" in sql:
            raise RuntimeError("injected mid-read failure")

    restore = _patch_connect(batches.get_store(), log=log, hook=boom)
    # raise_server_exceptions=False：异常透传为 500 而不是在测试里抛出
    r = TestClient(app, raise_server_exceptions=False).get(
        f"/v1/blank-subtraction-batches/{d['batch_id']}"
    )
    assert r.status_code == 500
    assert log == ["BEGIN", "SELECT", "SELECT", "ROLLBACK", "CLOSE"]

    # 事务与连接已释放：还原钩子后同一存储立即可用
    restore()
    r = client.get(f"/v1/blank-subtraction-batches/{d['batch_id']}")
    assert r.status_code == 200
    assert r.json()["version"] == 1
