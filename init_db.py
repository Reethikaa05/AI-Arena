import os
import sqlite3
from pathlib import Path

def init_db(db_path: Path, seed_sql_path: Path):
    # Ensure parent directory exists
    db_path.parent.mkdir(parents=True, exist_ok=True)
    # Connect to SQLite (creates file if not exists)
    conn = sqlite3.connect(db_path)
    try:
        with open(seed_sql_path, "r", encoding="utf-8") as f:
            sql_script = f.read()
        conn.executescript(sql_script)
        print(f"Database initialized at {db_path}")
    finally:
        conn.close()

if __name__ == "__main__":
    base_dir = Path(__file__).parent
    db_file = base_dir / "data" / "app.db"
    seed_file = base_dir / "data" / "seed.sql"
    init_db(db_file, seed_file)
