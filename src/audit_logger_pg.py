import hashlib
import json
import psycopg2
from datetime import datetime

class PGAuditLogger:
    def __init__(self):
        self.db_config = {
            "host": "localhost",
            "database": "gazeloop",
            "user": "postgres",
            "password": "123456"
        }

    def log_event(self, snapshot_path: str, metadata: dict, llm_response: dict, hitl_status: str = "PENDING") -> str:
        """
        Calculates a SHA-256 cryptographic hash chaining to the previous record 
        and securely inserts the alarm event into PostgreSQL.
        """
        timestamp = datetime.utcnow().isoformat()

        with psycopg2.connect(**self.db_config) as conn:
            with conn.cursor() as cur:
                # Row-level lock (FOR UPDATE) preserves strict chronological chain order 
                # even if concurrent alarm events trigger.
                cur.execute("SELECT current_hash FROM audit_logs ORDER BY id DESC LIMIT 1 FOR UPDATE")
                row = cur.fetchone()
                prev_hash = row[0] if row else "0" * 64

                # Canonicalize dictionaries into sorted JSON strings for deterministic hashing
                meta_str = json.dumps(metadata, sort_keys=True)
                payload_str = json.dumps(llm_response, sort_keys=True)

                # Generate cryptographic hash linking this event to the chain history
                raw_data = f"{timestamp}{snapshot_path}{meta_str}{payload_str}{hitl_status}{prev_hash}"
                current_hash = hashlib.sha256(raw_data.encode('utf-8')).hexdigest()

                # Insert into PostgreSQL using native JSONB casting
                cur.execute('''
                    INSERT INTO audit_logs 
                    (timestamp, snapshot_path, model_metadata, llm_payload, hitl_decision, prev_hash, current_hash)
                    VALUES (%s, %s, %s::jsonb, %s::jsonb, %s, %s, %s)
                ''', (
                    timestamp, 
                    snapshot_path, 
                    json.dumps(metadata), 
                    json.dumps(llm_response), 
                    hitl_status, 
                    prev_hash, 
                    current_hash
                ))
            conn.commit()
            
        print(f"[AuditLogger] Alarm securely logged to SQL. Hash: {current_hash[:12]}...")
        return current_hash