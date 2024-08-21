import sqlite3
import numpy as np
import hnswlib
import uuid
import hashlib
import shutil
import os
import time


def generate_hash_id(input_id):
    return hashlib.sha256(input_id.encode()).hexdigest()


class VectorDBManager:
    def __init__(self, db_directory="vector_dbs", log_db_file="logs_db.sqlite"):
        self.db_directory = db_directory
        self.log_db_file = log_db_file
        if not os.path.exists(self.db_directory):
            os.makedirs(self.db_directory)
        self.connection = sqlite3.connect(self.log_db_file)
        self.cursor = self.connection.cursor()
        self.cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS vector_databases (
                db_id TEXT PRIMARY KEY,
                dim INTEGER,
                space TEXT,
                max_elements INTEGER,
                ef_construction INTEGER,
                M INTEGER
            )
        """
        )
        self.cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS history_logs (
                log_id TEXT PRIMARY KEY,
                db_id TEXT,
                action TEXT,
                vector_id TEXT,
                details TEXT,
                timestamp TEXT,
                FOREIGN KEY(db_id) REFERENCES vector_databases(db_id)
            )
        """
        )
        self.connection.commit()
        self.databases = {}

    def _get_db_path(self, db_id):
        return os.path.join(self.db_directory, f"{db_id}.sqlite")

    def create_database(
        self, dim, space="l2", max_elements=10000, ef_construction=200, M=16
    ):
        db_id = str(uuid.uuid4())
        db_path = self._get_db_path(db_id)
        connection = sqlite3.connect(db_path)
        cursor = connection.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS vectors (
                vector_id TEXT PRIMARY KEY,
                vector BLOB
            )
        """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS vector_metadata (
                vector_id TEXT PRIMARY KEY,
                metadata TEXT
            )
        """
        )
        connection.commit()
        new_db = VectorDB(
            dim,
            space,
            max_elements,
            ef_construction,
            M,
            db_id,
            connection,
            self.log_db_file,
        )
        self.databases[db_id] = new_db
        self.cursor.execute(
            "INSERT INTO vector_databases (db_id, dim, space, max_elements, ef_construction, M) VALUES (?, ?, ?, ?, ?, ?)",
            (db_id, dim, space, max_elements, ef_construction, M),
        )
        self.connection.commit()
        return db_id

    def get_database(self, db_id):
        if db_id in self.databases:
            return self.databases[db_id]
        else:
            db_path = self._get_db_path(db_id)
            connection = sqlite3.connect(db_path)
            cursor = self.connection.cursor()
            cursor.execute(
                "SELECT dim, space, max_elements, ef_construction, M FROM vector_databases WHERE db_id = ?",
                (db_id,),
            )
            row = cursor.fetchone()
            if row:
                dim, space, max_elements, ef_construction, M = row
                new_db = VectorDB(
                    dim,
                    space,
                    max_elements,
                    ef_construction,
                    M,
                    db_id,
                    connection,
                    self.log_db_file,
                )
                self.databases[db_id] = new_db
                return new_db
            else:
                raise ValueError("Database ID not found")

    def add_vector(self, db_id, vector_id, vector, metadata=None):
        db = self.get_database(db_id)
        db.add(vector, vector_id)
        if metadata:
            db.add_metadata(vector_id, metadata)
        print(f"Added vector with ID: {vector_id}, in database ID: {db_id}")

    def get_vector(self, db_id, vector_id):
        db = self.get_database(db_id)
        return db.get(vector_id)

    def get_log(self, db_id):
        cursor = self.connection.cursor()
        cursor.execute(
            "SELECT log_id, action, vector_id, details, timestamp FROM history_logs WHERE db_id = ?",
            (db_id,),
        )
        logs = cursor.fetchall()
        return logs

    def print_database_ids(self):
        for db_id in self.databases:
            print(f"Database ID: {db_id}")


class VectorDB:
    def __init__(
        self,
        dim,
        space,
        max_elements,
        ef_construction,
        M,
        db_id,
        connection,
        log_db_file,
    ):
        self.dim = dim
        self.space = space
        self.index = hnswlib.Index(space=space, dim=dim)
        self.index.init_index(
            max_elements=max_elements, ef_construction=ef_construction, M=M
        )
        self.id_map = {}
        self.next_id = 0
        self.index_set_ef_before_query = False
        self.db_id = db_id
        self.connection = connection
        self.cursor = self.connection.cursor()  # Initialize cursor here
        self.log_connection = sqlite3.connect(log_db_file)
        self.log_cursor = self.log_connection.cursor()
        self.backup_interval = 300  # Backup every 300 seconds (5 minutes)
        self.last_backup_time = time.time()
        self.index_backup_file = f"{db_id}_index.hnsw"
        self.sqlite_backup_file = f"{db_id}_vectrs_dbs_log.sqlite"

    def check_and_backup(self):
        if time.time() - self.last_backup_time >= self.backup_interval:
            self.backup_index()
            self.backup_sqlite_db()
            self.last_backup_time = time.time()

    def backup_index(self):
        self.index.save_index(self.index_backup_file)
        print(f"Index backed up to {self.index_backup_file}")

    def backup_sqlite_db(self):
        db_path = self.connection.execute("PRAGMA database_list;").fetchone()[2]
        backup_path = self.sqlite_backup_file
        shutil.copyfile(db_path, backup_path)
        print(f"SQLite database backed up to {backup_path}")

    def add(self, vector, id):
        hash_id = generate_hash_id(id)
        if hash_id not in self.id_map:
            self.id_map[hash_id] = self.next_id
            self.next_id += 1
        numerical_id = self.id_map[hash_id]
        self.index.add_items(np.array([vector]), np.array([numerical_id]))
        self.log_action("add", hash_id, f"Added vector with hash ID {hash_id}")
        self.check_and_backup()
        print(
            f"Added vector with ID: {id}, Hash ID: {hash_id}, Numerical ID: {numerical_id}"
        )
        self.cursor.execute(
            "INSERT INTO vectors (vector_id, vector) VALUES (?, ?)",
            (id, vector.tobytes()),
        )
        self.connection.commit()

    def get(self, id):
        hash_id = generate_hash_id(id)
        if hash_id in self.id_map:
            numerical_id = self.id_map[hash_id]
            print(
                f"Retrieving vector with ID: {id}, Hash ID: {hash_id}, Numerical ID: {numerical_id}"
            )
            vector_data = self.index.get_items([numerical_id])[0]
            return np.frombuffer(vector_data, dtype=np.float32)
        else:
            print(f"Vector ID {id} with Hash ID {hash_id} not found in id_map")
            raise ValueError("Vector ID not found")

    def query(self, vector, k=10):
        if not self.index_set_ef_before_query:
            raise ValueError("Set 'ef' parameter before querying the index.")
        if k > self.next_id:
            k = max(1, self.next_id)
        try:
            labels, distances = self.index.knn_query(vector, k=k)
            self.check_and_backup()
            return labels[0], distances[0]
        except RuntimeError:
            return [], []

    def update(self, id, new_vector):
        hash_id = generate_hash_id(id)
        if hash_id in self.id_map:
            numerical_id = self.id_map[hash_id]
            self.index.mark_deleted(numerical_id)
            self.index.add_items(np.array([new_vector]), np.array([numerical_id]))
            self.log_action("update", hash_id, f"Updated vector for hash ID {hash_id}")
            self.check_and_backup()
            self.cursor.execute(
                "UPDATE vectors SET vector = ? WHERE vector_id = ?",
                (new_vector.tobytes(), id),
            )
            self.connection.commit()
        else:
            raise ValueError("Error: Hash ID not found for update")

    def delete(self, id):
        hash_id = generate_hash_id(id)
        if hash_id in self.id_map:
            numerical_id = self.id_map[hash_id]
            self.index.mark_deleted(numerical_id)
            del self.id_map[hash_id]
            self.log_action("delete", hash_id, f"Deleted vector with hash ID {hash_id}")
            self.check_and_backup()
            self.cursor.execute("DELETE FROM vectors WHERE vector_id = ?", (id,))
            self.connection.commit()
        else:
            raise ValueError("Error: Hash ID not found for deletion")

    def log_action(self, action, vector_id, details):
        log_id = str(uuid.uuid4())
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        self.log_cursor.execute(
            "INSERT INTO history_logs (log_id, db_id, action, vector_id, details, timestamp) VALUES (?, ?, ?, ?, ?, ?)",
            (log_id, self.db_id, action, vector_id, details, timestamp),
        )
        self.log_connection.commit()
        return log_id

    def set_ef(self, ef):
        """Sets the 'ef' parameter for the index, which controls the size of the dynamic candidate list during the query."""
        self.index.set_ef(ef)
        self.index_set_ef_before_query = True

    def get_logs(self):
        """Fetches and returns all history logs from the database."""
        self.log_cursor.execute(
            "SELECT log_id, action, vector_id, details, timestamp FROM history_logs WHERE db_id = ?",
            (self.db_id,),
        )
        logs = self.log_cursor.fetchall()
        return logs

    def get_logs_by_hash(self, hash_id):
        """Fetches and returns history logs for a specific hash ID."""
        self.log_cursor.execute(
            "SELECT log_id, action, vector_id, details, timestamp FROM history_logs WHERE db_id = ? AND vector_id = ?",
            (self.db_id, hash_id),
        )
        logs = self.log_cursor.fetchall()
        return logs

    def knn_query(self, vector, k=10):
        """Queries the k nearest neighbors of the given vector."""
        labels, distances = self.index.knn_query(vector, k=k)
        return labels, distances

    def add_metadata(self, vector_id, metadata):
        self.cursor.execute(
            """
            INSERT OR REPLACE INTO vector_metadata (vector_id, metadata)
            VALUES (?, ?)
        """,
            (vector_id, metadata),
        )
        self.connection.commit()

    def get_metadata(self, vector_id):
        self.cursor.execute(
            """
            SELECT metadata FROM vector_metadata WHERE vector_id = ?
        """,
            (vector_id,),
        )
        result = self.cursor.fetchone()
        return result[0] if result else None

    def delete_metadata(self, vector_id):
        self.cursor.execute(
            """
            DELETE FROM vector_metadata WHERE vector_id = ?
        """,
            (vector_id,),
        )
        self.connection.commit()

    def update_metadata(self, vector_id, metadata):
        self.add_metadata(vector_id, metadata)
