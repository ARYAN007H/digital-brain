from datetime import datetime, timedelta, timezone

from brain.queue import WriteQueue


def test_timestamp_migration_normalizes_mixed_formats(tmp_path):
    db_path = tmp_path / "queue.db"
    q = WriteQueue(db_path=db_path)
    conn = q._get_conn()

    base = datetime(2026, 5, 10, 12, 0, 0, tzinfo=timezone.utc)
    iso = base.isoformat()
    sqlite_text = base.strftime("%Y-%m-%d %H:%M:%S")
    epoch = int(base.timestamp())

    conn.execute(
        """
        INSERT INTO pending_writes (target, operation, payload, created_at, status, completed_at, last_attempt_at)
        VALUES ('supabase', 'upsert', '{}', ?, 'done', ?, ?)
        """,
        (iso, sqlite_text, str(epoch)),
    )
    conn.commit()
    q.close()

    q2 = WriteQueue(db_path=db_path)
    row = q2._get_conn().execute(
        "SELECT created_at, completed_at, last_attempt_at FROM pending_writes"
    ).fetchone()

    assert str(row["created_at"]).isdigit()
    assert str(row["completed_at"]).isdigit()
    assert str(row["last_attempt_at"]).isdigit()
    assert int(row["created_at"]) == epoch
    assert int(row["completed_at"]) == epoch
    assert int(row["last_attempt_at"]) == epoch


def test_purge_completed_handles_migrated_and_timezone_values(tmp_path):
    db_path = tmp_path / "queue.db"
    q = WriteQueue(db_path=db_path)
    conn = q._get_conn()

    now = datetime.now(timezone.utc)
    old = now - timedelta(hours=30)
    recent = now - timedelta(hours=1)

    conn.execute(
        """
        INSERT INTO pending_writes (target, operation, payload, created_at, status, completed_at)
        VALUES ('supabase', 'upsert', '{}', ?, 'done', ?)
        """,
        (old.isoformat(), old.strftime("%Y-%m-%d %H:%M:%S")),
    )
    conn.execute(
        """
        INSERT INTO pending_writes (target, operation, payload, created_at, status, completed_at)
        VALUES ('supabase', 'upsert', '{}', ?, 'done', ?)
        """,
        (recent.isoformat(), recent.astimezone(timezone(timedelta(hours=-7))).isoformat()),
    )
    conn.commit()

    # Trigger one-time migration after legacy-format inserts.
    q._migrate_timestamps_to_epoch(conn)
    conn.commit()

    q.purge_completed(older_than_hours=24)

    rows = conn.execute("SELECT status, completed_at FROM pending_writes").fetchall()
    assert len(rows) == 1
    assert str(rows[0]["completed_at"]).isdigit()
    assert int(rows[0]["completed_at"]) == int(recent.timestamp())
