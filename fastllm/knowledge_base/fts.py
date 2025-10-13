import sqlite3
import json
from typing import List, Dict, Any
import os
import re
from fastllm.knowledge_base.knowledge_interface import KnowledgeBaseInterface


class FullTextSearchBase(KnowledgeBaseInterface):
    """
    A document indexing class with full-text search capabilities using SQLite FTS5.
    Similar to ChromaDB but without vector search/indexing.
    """

    def __init__(self, path: str):
        """Initialize the document index with SQLite FTS5."""
        self.path = path
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row  # Enable column access by name
        self._initialize_database()

    def _initialize_database(self):
        """Initialize the database with FTS5 support."""
        # Enable foreign keys and WAL mode for better concurrency
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.conn.execute("PRAGMA journal_mode = WAL")

        # Create tables for collections and metadata
        self.conn.execute(
            """                                                                                     
            CREATE TABLE IF NOT EXISTS collections (                                                              
                id INTEGER PRIMARY KEY AUTOINCREMENT,                                                             
                name TEXT UNIQUE NOT NULL,                                                                        
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP                                                    
            )                                                                                                     
        """
        )

        self.conn.commit()

    def _sanitize_table_name(self, collection_name: str) -> str:
        """Sanitize collection name for use as table name."""
        # Remove any characters that aren't alphanumeric or underscore
        sanitized = re.sub(r"[^a-zA-Z0-9_]", "_", collection_name)
        # Ensure it doesn't start with a number
        if sanitized and sanitized[0].isdigit():
            sanitized = f"_{sanitized}"
        # Add prefix to avoid conflicts
        return f"fts_{sanitized}" if sanitized else "fts_default"

    def _create_collection_table(self, collection_name: str):
        """Create FTS5 table for a collection if it doesn't exist."""
        table_name = self._sanitize_table_name(collection_name)

        # Create FTS5 virtual table
        self.conn.execute(
            f"""                                                                                    
            CREATE VIRTUAL TABLE IF NOT EXISTS {table_name}                                                       
            USING fts5(                                                                                           
                content,                                                                                          
                metadata,                                                                                         
                collection_id UNINDEXED,                                                                          
                id UNINDEXED,                                                                                     
                tokenize='porter unicode61'                                                                       
            )                                                                                                     
        """
        )

        # Create collections entry if not exists
        self.conn.execute(
            """                                                                                     
            INSERT OR IGNORE INTO collections (name) VALUES (?)                                                   
        """,
            (collection_name,),
        )

        self.conn.commit()
        return table_name

    def _get_collection_table(self, collection_name: str) -> str:
        """Get the FTS table name for a collection."""
        # Check if collection exists
        cursor = self.conn.execute(
            """                                                                            
            SELECT id FROM collections WHERE name = ?                                                             
        """,
            (collection_name,),
        )

        if not cursor.fetchone():
            raise ValueError(f"Collection '{collection_name}' does not exist")

        return self._sanitize_table_name(collection_name)

    def get_collection(self, collection_name: str):
        """Get a collection (returns self for chaining in this implementation)."""
        # Verify collection exists, create if needed
        self._create_collection_table(collection_name)
        return self  # Following the interface pattern, though typically would return a collection object

    def get_collection_names(self) -> List[str]:
        """Get all collection names."""
        cursor = self.conn.execute("SELECT name FROM collections")
        return [row[0] for row in cursor.fetchall()]

    def insert(
        self,
        collection_name: str,
        texts: List[str],
        metadatas: List[Dict] = None,
    ):
        """Insert documents into a collection."""
        if not texts:
            return

        if metadatas is None:
            metadatas = [{}] * len(texts)

        if len(texts) != len(metadatas):
            raise ValueError("Number of texts must match number of metadatas")

        table_name = self._create_collection_table(collection_name)

        # Get collection ID
        cursor = self.conn.execute(
            """                                                                            
            SELECT id FROM collections WHERE name = ?                                                             
        """,
            (collection_name,),
        )
        collection_id_row = cursor.fetchone()

        if not collection_id_row:
            raise ValueError(f"Collection '{collection_name}' does not exist")

        collection_id = collection_id_row[0]

        # Insert documents
        for text, metadata in zip(texts, metadatas):
            metadata_json = json.dumps(metadata) if metadata else "{}"
            self.conn.execute(
                f"""                                                                                
                INSERT INTO {table_name} (content, metadata, collection_id)                                       
                VALUES (?, ?, ?)                                                                                  
            """,
                (text, metadata_json, collection_id),
            )

        self.conn.commit()

    def query(
        self, collection_name: str, query: str, k: int = 5
    ) -> List[Dict[str, Any]]:
        """Query documents in a collection."""
        if k <= 0:
            return []

        try:
            table_name = self._get_collection_table(collection_name)
        except ValueError:
            return []  # Return empty list if collection doesn't exist

        # Get collection ID
        cursor = self.conn.execute(
            """                                                                            
            SELECT id FROM collections WHERE name = ?                                                             
        """,
            (collection_name,),
        )
        collection_id_row = cursor.fetchone()

        if not collection_id_row:
            return []

        collection_id = collection_id_row[0]

        # Handle empty query
        if not query or not query.strip():
            # Return top k documents by ID (most recent)
            cursor = self.conn.execute(
                f"""                                                                       
                SELECT rowid, content, metadata                                                                   
                FROM {table_name}                                                                                 
                WHERE collection_id = ?                                                                           
                ORDER BY rowid DESC                                                                               
                LIMIT ?                                                                                           
            """,
                (collection_id, k),
            )
        else:
            # Perform FTS5 search with BM25 ranking
            cursor = self.conn.execute(
                f"""                                                                       
                SELECT rowid, content, metadata, rank                                                             
                FROM {table_name}                                                                                 
                WHERE {table_name} MATCH ? AND collection_id = ?                                                  
                ORDER BY rank                                                                                     
                LIMIT ?                                                                                           
            """,
                (query, collection_id, k),
            )

        results = []
        for row in cursor.fetchall():
            metadata = json.loads(row["metadata"]) if row["metadata"] else {}
            result = {
                "id": row["rowid"],
                "content": row["content"],
                "metadata": metadata,
            }

            # Add score if available (from FTS search)
            if "rank" in row.keys():
                result["score"] = -row["rank"]  # Convert to positive score

            results.append(result)

        return results

    def wipe(self, collection_name: str):
        """Delete all documents from a collection."""
        try:
            table_name = self._get_collection_table(collection_name)
        except ValueError:
            return  # Collection doesn't exist, nothing to wipe

        # Get collection ID
        cursor = self.conn.execute(
            """                                                                            
            SELECT id FROM collections WHERE name = ?                                                             
        """,
            (collection_name,),
        )
        collection_id_row = cursor.fetchone()

        if collection_id_row:
            # Delete all documents in this collection
            self.conn.execute(
                f"""                                                                                
                DELETE FROM {table_name} WHERE collection_id = ?                                                  
            """,
                (collection_id_row[0],),
            )
            self.conn.commit()

    def delete_index(self):
        """Delete the entire database."""
        self.conn.close()
        if os.path.exists(self.path):
            os.remove(self.path)
