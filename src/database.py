from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

from utils import DB_PATH, ensure_directories, resolve_project_path, to_project_relative_path


class ProfileDatabase:
    def __init__(self, db_path: str | Path = DB_PATH) -> None:
        self.db_path = Path(db_path)

    def connect(self) -> sqlite3.Connection:
        ensure_directories()
        return sqlite3.connect(self.db_path)

    def initialize(self) -> None:
        with self.connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS people (
                    person_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    age INTEGER NOT NULL,
                    nationality TEXT NOT NULL,
                    career TEXT NOT NULL,
                    role TEXT NOT NULL DEFAULT 'visitor',
                    profile_image_path TEXT,
                    created_at TEXT NOT NULL
                )
                """
            )
            self._ensure_column(connection, "people", "profile_image_path", "TEXT")
            self._ensure_column(
                connection,
                "people",
                "profile_image_source",
                "TEXT NOT NULL DEFAULT 'first_sample'",
            )
            self._ensure_column(connection, "people", "role", "TEXT NOT NULL DEFAULT 'visitor'")
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS face_samples (
                    sample_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    person_id INTEGER NOT NULL,
                    image_path TEXT NOT NULL,
                    embedding_path TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(person_id) REFERENCES people(person_id) ON DELETE CASCADE
                )
                """
            )
            connection.execute(
                """
                UPDATE people
                SET role = 'visitor'
                WHERE role IS NULL OR TRIM(role) = ''
                """
            )
            connection.execute(
                """
                UPDATE people
                SET profile_image_path = (
                    SELECT face_samples.image_path
                    FROM face_samples
                    WHERE face_samples.person_id = people.person_id
                    ORDER BY face_samples.sample_id
                    LIMIT 1
                )
                WHERE profile_image_path IS NULL
                  AND EXISTS (
                    SELECT 1
                    FROM face_samples
                    WHERE face_samples.person_id = people.person_id
                  )
                """
            )
            self._normalize_paths(connection)
            connection.execute(
                """
                UPDATE people
                SET profile_image_source = 'first_sample'
                WHERE profile_image_source IS NULL OR TRIM(profile_image_source) = ''
                """
            )
            self._consolidate_sample_images(connection)
            connection.commit()

    def insert_person(
        self,
        name: str,
        age: int,
        nationality: str,
        career: str,
        created_at: str,
        role: str = "visitor",
        profile_image_path: str | None = None,
    ) -> int:
        with self.connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO people (name, age, nationality, career, role, profile_image_path, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (name, age, nationality, career, role, profile_image_path, created_at),
            )
            connection.commit()
            return int(cursor.lastrowid)

    def update_profile_image(
        self,
        person_id: int,
        profile_image_path: str,
        source: str = "manual",
    ) -> None:
        with self.connect() as connection:
            connection.execute(
                """
                UPDATE people
                SET profile_image_path = ?, profile_image_source = ?
                WHERE person_id = ?
                """,
                (str(to_project_relative_path(profile_image_path)), source, person_id),
            )
            connection.commit()

    def update_person(
        self,
        person_id: int,
        name: str,
        age: int,
        nationality: str,
        career: str,
        role: str = "visitor",
    ) -> None:
        with self.connect() as connection:
            connection.execute(
                """
                UPDATE people
                SET name = ?, age = ?, nationality = ?, career = ?, role = ?
                WHERE person_id = ?
                """,
                (name, age, nationality, career, role, person_id),
            )
            connection.commit()

    def add_face_sample(
        self,
        person_id: int,
        image_path: str,
        embedding_path: str,
        created_at: str,
    ) -> int:
        with self.connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO face_samples (person_id, image_path, embedding_path, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (
                    person_id,
                    str(to_project_relative_path(image_path)),
                    str(to_project_relative_path(embedding_path)),
                    created_at,
                ),
            )
            connection.commit()
            return int(cursor.lastrowid)

    def get_person_by_id(self, person_id: int) -> dict[str, Any] | None:
        with self.connect() as connection:
            connection.row_factory = sqlite3.Row
            row = connection.execute(
                "SELECT * FROM people WHERE person_id = ?",
                (person_id,),
            ).fetchone()
        return dict(row) if row else None

    def get_person_by_name(self, name: str) -> list[dict[str, Any]]:
        with self.connect() as connection:
            connection.row_factory = sqlite3.Row
            rows = connection.execute(
                "SELECT * FROM people WHERE LOWER(name) = LOWER(?) ORDER BY person_id",
                (name,),
            ).fetchall()
        return [dict(row) for row in rows]

    def get_all_face_samples(self) -> list[dict[str, Any]]:
        with self.connect() as connection:
            connection.row_factory = sqlite3.Row
            rows = connection.execute(
                """
                SELECT
                    people.person_id,
                    people.name,
                    people.age,
                    people.nationality,
                    people.career,
                    people.role,
                    people.profile_image_path,
                    people.created_at,
                    face_samples.sample_id,
                    face_samples.image_path,
                    face_samples.embedding_path
                FROM face_samples
                JOIN people ON people.person_id = face_samples.person_id
                ORDER BY people.person_id, face_samples.sample_id
                """
            ).fetchall()
        return [dict(row) for row in rows]

    def get_all_people(self) -> list[dict[str, Any]]:
        with self.connect() as connection:
            connection.row_factory = sqlite3.Row
            rows = connection.execute(
                """
                SELECT
                    people.person_id,
                    people.name,
                    people.age,
                    people.nationality,
                    people.career,
                    people.role,
                    people.profile_image_path,
                    people.created_at,
                    COUNT(face_samples.sample_id) AS sample_count
                FROM people
                LEFT JOIN face_samples ON face_samples.person_id = people.person_id
                GROUP BY people.person_id
                ORDER BY people.person_id
                """
            ).fetchall()
        return [dict(row) for row in rows]

    @staticmethod
    def _ensure_column(
        connection: sqlite3.Connection,
        table_name: str,
        column_name: str,
        column_type: str,
    ) -> None:
        columns = {
            row[1]
            for row in connection.execute(f"PRAGMA table_info({table_name})").fetchall()
        }
        if column_name not in columns:
            connection.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_type}")

    def delete_person(self, person_id: int) -> int:
        with self.connect() as connection:
            paths = connection.execute(
                """
                SELECT DISTINCT path_value
                FROM (
                    SELECT profile_image_path AS path_value
                    FROM people
                    WHERE person_id = ?
                      AND profile_image_path IS NOT NULL
                      AND TRIM(profile_image_path) <> ''
                    UNION ALL
                    SELECT image_path AS path_value
                    FROM face_samples
                    WHERE person_id = ?
                      AND image_path IS NOT NULL
                      AND TRIM(image_path) <> ''
                )
                """,
                (person_id, person_id),
            ).fetchall()
            sample_count = connection.execute(
                "SELECT COUNT(*) FROM face_samples WHERE person_id = ?",
                (person_id,),
            ).fetchone()[0]
            person_count = connection.execute(
                "SELECT COUNT(*) FROM people WHERE person_id = ?",
                (person_id,),
            ).fetchone()[0]
            connection.execute("DELETE FROM face_samples WHERE person_id = ?", (person_id,))
            connection.execute("DELETE FROM people WHERE person_id = ?", (person_id,))
            connection.commit()
        for (path_value,) in paths:
            self._delete_file(path_value)
        return int(sample_count + person_count)

    def has_people(self) -> bool:
        with self.connect() as connection:
            row = connection.execute("SELECT COUNT(*) FROM people").fetchone()
        return bool(row and row[0] > 0)

    def _consolidate_sample_images(self, connection: sqlite3.Connection) -> None:
        connection.row_factory = sqlite3.Row
        people = connection.execute(
            """
            SELECT person_id, profile_image_path, profile_image_source
            FROM people
            ORDER BY person_id
            """
        ).fetchall()
        for person in people:
            samples = connection.execute(
                """
                SELECT sample_id, image_path
                FROM face_samples
                WHERE person_id = ?
                ORDER BY sample_id
                """,
                (person["person_id"],),
            ).fetchall()
            if not samples:
                continue

            first_sample_path = str(samples[0]["image_path"])
            stale_paths = {
                str(sample["image_path"])
                for sample in samples[1:]
                if sample["image_path"] and str(sample["image_path"]).strip()
            }
            current_profile_path = person["profile_image_path"]
            profile_source = str(person["profile_image_source"] or "first_sample")
            if profile_source != "manual":
                if current_profile_path and current_profile_path != first_sample_path:
                    stale_paths.add(str(current_profile_path))
                connection.execute(
                    """
                    UPDATE people
                    SET profile_image_path = ?, profile_image_source = 'first_sample'
                    WHERE person_id = ?
                    """,
                    (str(to_project_relative_path(first_sample_path)), person["person_id"]),
                )
            connection.execute(
                "UPDATE face_samples SET image_path = ? WHERE person_id = ?",
                (str(to_project_relative_path(first_sample_path)), person["person_id"]),
            )
            for path_value in stale_paths:
                if path_value != first_sample_path:
                    self._delete_file(path_value)

    def _normalize_paths(self, connection: sqlite3.Connection) -> None:
        people = connection.execute(
            "SELECT person_id, profile_image_path FROM people"
        ).fetchall()
        for person_id, profile_image_path in people:
            if profile_image_path:
                connection.execute(
                    "UPDATE people SET profile_image_path = ? WHERE person_id = ?",
                    (str(to_project_relative_path(profile_image_path)), person_id),
                )

        samples = connection.execute(
            "SELECT sample_id, image_path, embedding_path FROM face_samples"
        ).fetchall()
        for sample_id, image_path, embedding_path in samples:
            connection.execute(
                """
                UPDATE face_samples
                SET image_path = ?, embedding_path = ?
                WHERE sample_id = ?
                """,
                (
                    str(to_project_relative_path(image_path)),
                    str(to_project_relative_path(embedding_path)),
                    sample_id,
                ),
            )

    @staticmethod
    def _delete_file(path_value: str) -> None:
        path = resolve_project_path(path_value)
        try:
            if path.exists():
                path.unlink()
        except OSError:
            pass
