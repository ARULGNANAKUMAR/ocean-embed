"""
SCIENTIST HUB
Collaborative platform for ocean scientists
OCEANVERSE Phase 5
"""

import json
import logging
import hashlib
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Tuple
from enum import Enum
import sqlite3
from pathlib import Path

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# DATA STRUCTURES
# ─────────────────────────────────────────────────────────────

class ReviewStatus(Enum):
    """Prediction review status."""
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    FLAGGED = "flagged"
    REVISION_REQUESTED = "revision_requested"


class UserRole(Enum):
    """Scientist roles."""
    ADMIN = "admin"
    LEAD_SCIENTIST = "lead_scientist"
    REVIEWER = "reviewer"
    CONTRIBUTOR = "contributor"
    STUDENT = "student"


@dataclass
class Scientist:
    """Scientist profile."""
    id: str
    name: str
    email: str
    institution: str
    specialty: str  # e.g., "Physical Oceanography", "Marine Biology"
    phone: str = ""
    orcid: str = ""  # Open Researcher & Contributor ID
    role: UserRole = UserRole.CONTRIBUTOR
    is_active: bool = True
    created_at: datetime = field(default_factory=datetime.now)
    last_login: Optional[datetime] = None
    profile_url: str = ""
    bio: str = ""


@dataclass
class PredictionReview:
    """Scientist's review of AI prediction."""
    id: str
    prediction_id: str
    scientist_id: str
    status: ReviewStatus
    confidence_in_prediction: float  # 0-1
    comments: str
    time_spent_minutes: int
    data_sources_consulted: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    approved_at: Optional[datetime] = None
    reasoning: str = ""


@dataclass
class Observation:
    """Ground truth observation uploaded by scientist."""
    id: str
    scientist_id: str
    location: Tuple[float, float]  # (lat, lon)
    depth: float
    timestamp: datetime
    variable: str  # temperature, salinity, etc.
    value: float
    unit: str
    source: str  # "manual", "instrument", "cruise"
    quality_flag: str  # "good", "questionable", "bad"
    metadata: Dict = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class Comment:
    """Threaded comment on prediction, model, or dataset."""
    id: str
    entity_id: str  # prediction_id, model_id, or dataset_id
    entity_type: str  # "prediction", "model", "dataset"
    scientist_id: str
    text: str
    thread_id: Optional[str] = None  # for nested comments
    likes: int = 0
    created_at: datetime = field(default_factory=datetime.now)
    edited_at: Optional[datetime] = None
    mentions: List[str] = field(default_factory=list)  # scientist IDs


@dataclass
class Dashboard:
    """Personalized scientist dashboard."""
    scientist_id: str
    pending_reviews: List[Dict]
    recent_approvals: List[Dict]
    recent_observations: List[Dict]
    collaboration_activity: List[Dict]
    model_performance: List[Dict]
    upcoming_deadlines: List[Dict]
    research_publications: List[Dict]


# ─────────────────────────────────────────────────────────────
# AUTHENTICATION
# ─────────────────────────────────────────────────────────────

class AuthenticationManager:
    """Manage scientist authentication."""
    
    def __init__(self, db_path: str = 'oceanverse.db'):
        self.db_path = db_path
        self.logger = logger
        self.sessions: Dict[str, Dict] = {}
        self._init_db()
    
    def _init_db(self):
        """Initialize authentication database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS scientists (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                institution TEXT,
                specialty TEXT,
                phone TEXT,
                orcid TEXT,
                role TEXT DEFAULT 'contributor',
                is_active BOOLEAN DEFAULT 1,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                last_login DATETIME
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                scientist_id TEXT NOT NULL,
                login_time DATETIME,
                logout_time DATETIME,
                ip_address TEXT,
                user_agent TEXT,
                FOREIGN KEY(scientist_id) REFERENCES scientists(id)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def hash_password(self, password: str) -> str:
        """Hash password using SHA-256."""
        return hashlib.sha256(password.encode()).hexdigest()
    
    def register_scientist(self,
                          name: str,
                          email: str,
                          password: str,
                          institution: str,
                          specialty: str) -> str:
        """Register new scientist."""
        
        scientist_id = f"sci_{hashlib.md5(email.encode()).hexdigest()[:8]}"
        password_hash = self.hash_password(password)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                INSERT INTO scientists
                (id, name, email, password_hash, institution, specialty, role)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (scientist_id, name, email, password_hash, institution, specialty, 'contributor'))
            
            conn.commit()
            self.logger.info(f"Registered scientist: {scientist_id}")
            return scientist_id
            
        except sqlite3.IntegrityError:
            self.logger.warning(f"Registration failed: email already registered")
            raise ValueError("Email already registered")
        finally:
            conn.close()
    
    def login(self,
              email: str,
              password: str,
              ip_address: str = "0.0.0.0") -> Tuple[str, Scientist]:
        """
        Authenticate scientist and create session.
        
        Returns:
            (session_id, scientist_data)
        """
        
        password_hash = self.hash_password(password)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            'SELECT id, name, email, institution, specialty, role FROM scientists '
            'WHERE email = ? AND password_hash = ? AND is_active = 1',
            (email, password_hash)
        )
        row = cursor.fetchone()
        
        if not row:
            self.logger.warning(f"Login failed: {email}")
            raise ValueError("Invalid credentials")
        
        scientist_id = row[0]
        session_id = hashlib.sha256(f"{scientist_id}{datetime.now()}".encode()).hexdigest()
        
        # Create session
        cursor.execute('''
            INSERT INTO sessions (session_id, scientist_id, login_time, ip_address)
            VALUES (?, ?, ?, ?)
        ''', (session_id, scientist_id, datetime.now().isoformat(), ip_address))
        
        # Update last login
        cursor.execute(
            'UPDATE scientists SET last_login = ? WHERE id = ?',
            (datetime.now().isoformat(), scientist_id)
        )
        
        conn.commit()
        conn.close()
        
        scientist = Scientist(
            id=scientist_id,
            name=row[1],
            email=row[2],
            institution=row[3],
            specialty=row[4],
            role=UserRole(row[5])
        )
        
        self.logger.info(f"Login successful: {scientist_id}")
        return session_id, scientist
    
    def logout(self, session_id: str):
        """End scientist session."""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            'UPDATE sessions SET logout_time = ? WHERE session_id = ?',
            (datetime.now().isoformat(), session_id)
        )
        conn.commit()
        conn.close()
        
        self.logger.info(f"Logout: {session_id}")
    
    def validate_session(self, session_id: str) -> Optional[str]:
        """Check if session is valid. Returns scientist_id if valid."""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            SELECT scientist_id FROM sessions
            WHERE session_id = ? AND logout_time IS NULL
            AND login_time > datetime('now', '-24 hours')
        ''', (session_id,))
        row = cursor.fetchone()
        conn.close()
        
        return row[0] if row else None


# ─────────────────────────────────────────────────────────────
# PREDICTION REVIEW
# ─────────────────────────────────────────────────────────────

class PredictionReviewManager:
    """Manage scientist review of predictions."""
    
    def __init__(self, db_path: str = 'oceanverse.db'):
        self.db_path = db_path
        self.logger = logger
        self._init_db()
    
    def _init_db(self):
        """Initialize review database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS prediction_reviews (
                id TEXT PRIMARY KEY,
                prediction_id TEXT NOT NULL,
                scientist_id TEXT NOT NULL,
                status TEXT DEFAULT 'pending',
                confidence REAL,
                comments TEXT,
                time_spent_minutes INTEGER,
                data_sources TEXT,
                reasoning TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                approved_at DATETIME,
                FOREIGN KEY(scientist_id) REFERENCES scientists(id)
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS approvals (
                id TEXT PRIMARY KEY,
                prediction_id TEXT,
                scientist_id TEXT,
                approval_time DATETIME,
                deployment_status TEXT,
                FOREIGN KEY(scientist_id) REFERENCES scientists(id)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def get_pending_reviews(self, scientist_id: str = None) -> List[Dict]:
        """Get predictions awaiting review."""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        if scientist_id:
            cursor.execute('''
                SELECT id, prediction_id, status, created_at
                FROM prediction_reviews
                WHERE scientist_id = ? AND status = 'pending'
                ORDER BY created_at
            ''', (scientist_id,))
        else:
            cursor.execute('''
                SELECT id, prediction_id, scientist_id, status, created_at
                FROM prediction_reviews
                WHERE status = 'pending'
                ORDER BY created_at
            ''')
        
        rows = cursor.fetchall()
        conn.close()
        
        return [
            {
                'review_id': row[0],
                'prediction_id': row[1],
                'status': row[2] if len(row) > 2 else row[3],
                'created_at': row[3] if len(row) > 3 else row[4],
            }
            for row in rows
        ]
    
    def submit_review(self,
                     prediction_id: str,
                     scientist_id: str,
                     status: ReviewStatus,
                     confidence: float,
                     comments: str,
                     time_spent_minutes: int = 0) -> str:
        """Submit review of prediction."""
        
        review_id = f"rev_{hashlib.md5(f'{prediction_id}{scientist_id}'.encode()).hexdigest()[:8]}"
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT OR REPLACE INTO prediction_reviews
            (id, prediction_id, scientist_id, status, confidence, comments, time_spent_minutes, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (review_id, prediction_id, scientist_id, status.value, confidence, comments, time_spent_minutes,
              datetime.now().isoformat()))
        conn.commit()
        conn.close()
        
        self.logger.info(f"Review submitted: {review_id} ({status.value})")
        return review_id
    
    def approve_prediction(self,
                          prediction_id: str,
                          scientist_id: str,
                          comments: str = "") -> str:
        """Approve prediction for deployment."""
        
        approval_id = f"app_{hashlib.md5(f'{prediction_id}{scientist_id}'.encode()).hexdigest()[:8]}"
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Update review status
        cursor.execute('''
            UPDATE prediction_reviews
            SET status = 'approved', approved_at = ?
            WHERE prediction_id = ? AND scientist_id = ?
        ''', (datetime.now().isoformat(), prediction_id, scientist_id))
        
        # Record approval
        cursor.execute('''
            INSERT INTO approvals
            (id, prediction_id, scientist_id, approval_time, deployment_status)
            VALUES (?, ?, ?, ?, ?)
        ''', (approval_id, prediction_id, scientist_id, datetime.now().isoformat(), 'approved'))
        
        conn.commit()
        conn.close()
        
        self.logger.info(f"Prediction approved: {prediction_id}")
        return approval_id
    
    def reject_prediction(self,
                         prediction_id: str,
                         scientist_id: str,
                         reason: str) -> str:
        """Reject prediction."""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            UPDATE prediction_reviews
            SET status = 'rejected', comments = ?
            WHERE prediction_id = ? AND scientist_id = ?
        ''', (reason, prediction_id, scientist_id))
        conn.commit()
        conn.close()
        
        self.logger.info(f"Prediction rejected: {prediction_id}")
        return prediction_id


# ─────────────────────────────────────────────────────────────
# OBSERVATIONS
# ─────────────────────────────────────────────────────────────

class ObservationManager:
    """Manage scientist observations and ground truth data."""
    
    def __init__(self, db_path: str = 'oceanverse.db'):
        self.db_path = db_path
        self.logger = logger
        self._init_db()
    
    def _init_db(self):
        """Initialize observations database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS observations (
                id TEXT PRIMARY KEY,
                scientist_id TEXT NOT NULL,
                lat REAL, lon REAL, depth REAL,
                timestamp DATETIME NOT NULL,
                variable TEXT NOT NULL,
                value REAL NOT NULL,
                unit TEXT,
                source TEXT,
                quality_flag TEXT DEFAULT 'good',
                metadata TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(scientist_id) REFERENCES scientists(id)
            )
        ''')
        
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_obs_time ON observations(timestamp)
        ''')
        
        conn.commit()
        conn.close()
    
    def upload_observation(self,
                          scientist_id: str,
                          location: Tuple[float, float],
                          depth: float,
                          timestamp: datetime,
                          variable: str,
                          value: float,
                          unit: str = "°C",
                          source: str = "manual",
                          quality_flag: str = "good") -> str:
        """Upload observation data."""
        
        obs_id = f"obs_{hashlib.md5(f'{scientist_id}{timestamp}'.encode()).hexdigest()[:8]}"
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO observations
            (id, scientist_id, lat, lon, depth, timestamp, variable, value, unit, source, quality_flag, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (obs_id, scientist_id, location[0], location[1], depth, timestamp.isoformat(),
              variable, value, unit, source, quality_flag, datetime.now().isoformat()))
        conn.commit()
        conn.close()
        
        self.logger.info(f"Observation recorded: {obs_id}")
        return obs_id
    
    def bulk_upload_observations(self,
                                scientist_id: str,
                                observations: List[dict]) → List[str]:
        """Bulk upload multiple observations from CSV."""
        
        obs_ids = []
        for obs in observations:
            obs_id = self.upload_observation(
                scientist_id=scientist_id,
                location=(obs['lat'], obs['lon']),
                depth=obs['depth'],
                timestamp=datetime.fromisoformat(obs['timestamp']),
                variable=obs['variable'],
                value=float(obs['value']),
                unit=obs.get('unit', '°C'),
                source=obs.get('source', 'csv_import'),
                quality_flag=obs.get('quality', 'good'),
            )
            obs_ids.append(obs_id)
        
        self.logger.info(f"Bulk uploaded {len(obs_ids)} observations")
        return obs_ids
    
    def get_observations(self,
                        scientist_id: str = None,
                        variable: str = None,
                        date_range: Tuple[datetime, datetime] = None,
                        region: dict = None) -> List[Observation]:
        """Query observations with filters."""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        query = 'SELECT * FROM observations WHERE 1=1'
        params = []
        
        if scientist_id:
            query += ' AND scientist_id = ?'
            params.append(scientist_id)
        
        if variable:
            query += ' AND variable = ?'
            params.append(variable)
        
        if date_range:
            query += ' AND timestamp BETWEEN ? AND ?'
            params.extend([date_range[0].isoformat(), date_range[1].isoformat()])
        
        if region:
            query += ' AND lat BETWEEN ? AND ? AND lon BETWEEN ? AND ?'
            params.extend([region['lat_min'], region['lat_max'], 
                          region['lon_min'], region['lon_max']])
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()
        
        return [
            Observation(
                id=row[0],
                scientist_id=row[1],
                location=(row[2], row[3]),
                depth=row[4],
                timestamp=datetime.fromisoformat(row[5]),
                variable=row[6],
                value=row[7],
                unit=row[8],
                source=row[9],
                quality_flag=row[10],
            )
            for row in rows
        ]


# ─────────────────────────────────────────────────────────────
# COLLABORATION
# ─────────────────────────────────────────────────────────────

class CollaborationManager:
    """Manage scientist collaboration and comments."""
    
    def __init__(self, db_path: str = 'oceanverse.db'):
        self.db_path = db_path
        self.logger = logger
        self._init_db()
    
    def _init_db(self):
        """Initialize collaboration database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS comments (
                id TEXT PRIMARY KEY,
                entity_id TEXT NOT NULL,
                entity_type TEXT NOT NULL,
                scientist_id TEXT NOT NULL,
                text TEXT NOT NULL,
                thread_id TEXT,
                likes INTEGER DEFAULT 0,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                edited_at DATETIME,
                FOREIGN KEY(scientist_id) REFERENCES scientists(id)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def add_comment(self,
                   entity_id: str,
                   entity_type: str,
                   scientist_id: str,
                   text: str,
                   thread_id: str = None) -> str:
        """Add comment to prediction, model, or dataset."""
        
        comment_id = f"com_{hashlib.md5(f'{entity_id}{scientist_id}{datetime.now()}'.encode()).hexdigest()[:8]}"
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO comments
            (id, entity_id, entity_type, scientist_id, text, thread_id, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (comment_id, entity_id, entity_type, scientist_id, text, thread_id, datetime.now().isoformat()))
        conn.commit()
        conn.close()
        
        self.logger.info(f"Comment added: {comment_id}")
        return comment_id
    
    def get_comments(self,
                    entity_id: str,
                    entity_type: str) -> List[Comment]:
        """Retrieve all comments on entity."""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            SELECT id, scientist_id, text, thread_id, likes, created_at
            FROM comments
            WHERE entity_id = ? AND entity_type = ?
            ORDER BY created_at DESC
        ''', (entity_id, entity_type))
        
        rows = cursor.fetchall()
        conn.close()
        
        return [
            Comment(
                id=row[0],
                entity_id=entity_id,
                entity_type=entity_type,
                scientist_id=row[1],
                text=row[2],
                thread_id=row[3],
                likes=row[4],
                created_at=datetime.fromisoformat(row[5]),
            )
            for row in rows
        ]
    
    def like_comment(self, comment_id: str) -> int:
        """Increment comment likes."""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('UPDATE comments SET likes = likes + 1 WHERE id = ?', (comment_id,))
        cursor.execute('SELECT likes FROM comments WHERE id = ?', (comment_id,))
        likes = cursor.fetchone()[0]
        conn.commit()
        conn.close()
        
        return likes


# ─────────────────────────────────────────────────────────────
# SCIENTIST HUB ORCHESTRATOR
# ─────────────────────────────────────────────────────────────

class ScientistHub:
    """Main orchestrator for scientist collaboration."""
    
    def __init__(self, db_path: str = 'oceanverse.db'):
        self.auth = AuthenticationManager(db_path)
        self.review_manager = PredictionReviewManager(db_path)
        self.observation_manager = ObservationManager(db_path)
        self.collaboration_manager = CollaborationManager(db_path)
        self.logger = logger
    
    def scientist_login(self, email: str, password: str, ip_address: str = "0.0.0.0"):
        """Authenticate scientist."""
        return self.auth.login(email, password, ip_address)
    
    def scientist_logout(self, session_id: str):
        """End scientist session."""
        self.auth.logout(session_id)
    
    def get_dashboard(self, scientist_id: str) -> Dashboard:
        """Get personalized dashboard."""
        
        pending = self.review_manager.get_pending_reviews(scientist_id)
        observations = self.observation_manager.get_observations(scientist_id=scientist_id)
        
        return Dashboard(
            scientist_id=scientist_id,
            pending_reviews=pending,
            recent_approvals=[],
            recent_observations=[asdict(obs) for obs in observations[-5:]],
            collaboration_activity=[],
            model_performance=[],
            upcoming_deadlines=[],
            research_publications=[],
        )
    
    def review_prediction(self,
                         prediction_id: str,
                         scientist_id: str,
                         confidence: float,
                         comments: str) -> Dict:
        """Submit review of prediction."""
        
        review_id = self.review_manager.submit_review(
            prediction_id=prediction_id,
            scientist_id=scientist_id,
            status=ReviewStatus.PENDING,
            confidence=confidence,
            comments=comments,
        )
        
        return {
            'review_id': review_id,
            'status': 'pending',
            'prediction_id': prediction_id,
        }
    
    def approve_prediction(self,
                          prediction_id: str,
                          scientist_id: str) -> str:
        """Approve prediction for deployment."""
        return self.review_manager.approve_prediction(prediction_id, scientist_id)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    hub = ScientistHub()
    # Example usage
