"""
ECHO Module
Event Memory & Embedding Store for Continuous Learning
OCEANVERSE Phase 4
"""

import json
import logging
import sqlite3
import numpy as np
import pandas as pd
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from pathlib import Path
import hashlib

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# DATA STRUCTURES
# ─────────────────────────────────────────────────────────────

@dataclass
class Event:
    """Record of a marine event."""
    id: str
    event_type: str  # heatwave, cyclone, anomaly, etc.
    timestamp: datetime
    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float
    depth_min: float
    depth_max: float
    intensity: float  # 0-10 scale
    duration_days: int
    description: str = ""
    metadata: Dict = field(default_factory=dict)
    embedding: Optional[np.ndarray] = None


@dataclass
class SeasonalPattern:
    """Recurring seasonal pattern at a location."""
    location: Tuple[float, float]  # (lat, lon)
    variable: str
    month_ranges: List[Tuple[int, int]]  # [(start_month, end_month)]
    typical_value: float
    typical_std: float
    peak_value: float
    occurrence_probability: float


@dataclass
class Pattern:
    """Extracted pattern from event sequences."""
    id: str
    name: str
    event_types: List[str]
    typical_duration_days: int
    spatial_scale_km: float
    frequency_per_year: float
    precursors: List[str]  # Warning signs before pattern
    typical_impact: str


# ─────────────────────────────────────────────────────────────
# EVENT MEMORY
# ─────────────────────────────────────────────────────────────

class EventMemory:
    """Store and retrieve marine events with embeddings."""
    
    EVENT_TYPES = [
        'marine_heatwave',
        'cyclone',
        'sargasso_intrusion',
        'monsoon_shift',
        'oxygen_minimum_zone',
        'upwelling_event',
        'mjo_influence',
        'ioi_positive',
        'ioi_negative',
        'anomalous_sla',
        'current_reversal',
        'eddy_formation',
        'cold_core',
    ]
    
    def __init__(self, db_path: str = 'events.db'):
        self.db_path = db_path
        self.logger = logger
        self._init_db()
    
    def _init_db(self):
        """Initialize SQLite database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS events (
                id TEXT PRIMARY KEY,
                event_type TEXT NOT NULL,
                timestamp DATETIME NOT NULL,
                lat_min REAL, lat_max REAL,
                lon_min REAL, lon_max REAL,
                depth_min REAL, depth_max REAL,
                intensity REAL,
                duration_days INTEGER,
                description TEXT,
                metadata TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_event_type ON events(event_type)
        ''')
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_timestamp ON events(timestamp)
        ''')
        
        conn.commit()
        conn.close()
    
    def record_event(self,
                     event_type: str,
                     timestamp: datetime,
                     lat_min: float, lat_max: float,
                     lon_min: float, lon_max: float,
                     depth_min: float, depth_max: float,
                     intensity: float,
                     duration_days: int,
                     description: str = "",
                     metadata: Dict = None) -> Event:
        """Record a marine event."""
        
        if event_type not in self.EVENT_TYPES:
            self.logger.warning(f"Unknown event type: {event_type}")
        
        # Generate unique ID
        key = f"{event_type}_{timestamp.timestamp()}_{lat_min}_{lon_min}".encode()
        event_id = hashlib.md5(key).hexdigest()[:12]
        
        metadata = metadata or {}
        
        event = Event(
            id=event_id,
            event_type=event_type,
            timestamp=timestamp,
            lat_min=lat_min,
            lat_max=lat_max,
            lon_min=lon_min,
            lon_max=lon_max,
            depth_min=depth_min,
            depth_max=depth_max,
            intensity=intensity,
            duration_days=duration_days,
            description=description,
            metadata=metadata,
        )
        
        # Store in database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT OR REPLACE INTO events
            (id, event_type, timestamp, lat_min, lat_max, lon_min, lon_max,
             depth_min, depth_max, intensity, duration_days, description, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            event.id,
            event.event_type,
            event.timestamp.isoformat(),
            event.lat_min,
            event.lat_max,
            event.lon_min,
            event.lon_max,
            event.depth_min,
            event.depth_max,
            event.intensity,
            event.duration_days,
            event.description,
            json.dumps(event.metadata),
        ))
        conn.commit()
        conn.close()
        
        self.logger.info(f"Recorded event: {event_id} ({event_type})")
        return event
    
    def get_event_embedding(self, event: Event) -> np.ndarray:
        """
        Generate 128-D embedding for event.
        Captures: type, location, depth, intensity, duration.
        """
        # One-hot encode event type
        type_vec = np.zeros(len(self.EVENT_TYPES))
        type_idx = self.EVENT_TYPES.index(event.event_type) if event.event_type in self.EVENT_TYPES else 0
        type_vec[type_idx] = 1.0
        
        # Spatial features (normalized)
        lat_center = (event.lat_min + event.lat_max) / 2.0
        lon_center = (event.lon_min + event.lon_max) / 2.0
        lat_extent = (event.lat_max - event.lat_min) / 30.0  # Normalize to typical Indian Ocean extent
        lon_extent = (event.lon_max - event.lon_min) / 60.0
        
        spatial_vec = np.array([
            lat_center / 30.0,      # Normalize to [0, 1]
            lon_center / 105.0,
            lat_extent,
            lon_extent,
        ])
        
        # Depth features
        depth_vec = np.array([
            event.depth_min / 6000.0,
            event.depth_max / 6000.0,
            (event.depth_max - event.depth_min) / 6000.0,
        ])
        
        # Intensity and duration
        temporal_vec = np.array([
            event.intensity / 10.0,
            np.log(event.duration_days + 1) / 4.0,
        ])
        
        # Temporal features (month of year cyclicity)
        month = event.timestamp.month
        temporal_cyclicity = np.array([
            np.sin(2 * np.pi * month / 12),
            np.cos(2 * np.pi * month / 12),
        ])
        
        # Concatenate all features
        embedding = np.concatenate([
            type_vec,          # 13 dims
            spatial_vec,       # 4 dims
            depth_vec,         # 3 dims
            temporal_vec,      # 2 dims
            temporal_cyclicity,# 2 dims
            np.zeros(102),     # Padding to 128 dims
        ])[:128]
        
        event.embedding = embedding
        return embedding
    
    def query_similar_events(self,
                            event: Event,
                            k: int = 10) -> List[Event]:
        """Find k most similar events by embedding similarity."""
        
        query_emb = self.get_event_embedding(event)
        
        # Retrieve all events from database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM events ORDER BY timestamp DESC')
        rows = cursor.fetchall()
        conn.close()
        
        # Compute similarity to each event
        similarities = []
        for row in rows:
            event_id = row[0]
            if event_id == event.id:
                continue
            
            # Reconstruct event from row
            similar_event = Event(
                id=event_id,
                event_type=row[1],
                timestamp=datetime.fromisoformat(row[2]),
                lat_min=row[3], lat_max=row[4],
                lon_min=row[5], lon_max=row[6],
                depth_min=row[7], depth_max=row[8],
                intensity=row[9],
                duration_days=row[10],
                description=row[11],
                metadata=json.loads(row[12]) if row[12] else {},
            )
            
            similar_emb = self.get_event_embedding(similar_event)
            
            # Cosine similarity
            cosine_sim = np.dot(query_emb, similar_emb) / (
                np.linalg.norm(query_emb) * np.linalg.norm(similar_emb) + 1e-8
            )
            similarities.append((cosine_sim, similar_event))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[0], reverse=True)
        
        return [evt for _, evt in similarities[:k]]
    
    def get_seasonal_patterns(self,
                             lat: float,
                             lon: float,
                             variable: str = 'temperature') -> List[SeasonalPattern]:
        """Extract typical seasonal patterns for a location."""
        
        # Retrieve events near this location
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            SELECT * FROM events
            WHERE lat_min <= ? AND lat_max >= ?
              AND lon_min <= ? AND lon_max >= ?
            ORDER BY timestamp
        ''', (lat, lat, lon, lon))
        
        rows = cursor.fetchall()
        conn.close()
        
        # Group by month
        events_by_month = {i: [] for i in range(1, 13)}
        for row in rows:
            event_type = row[1]
            timestamp = datetime.fromisoformat(row[2])
            intensity = row[9]
            
            events_by_month[timestamp.month].append({
                'type': event_type,
                'intensity': intensity,
            })
        
        # Extract patterns
        patterns = []
        for month, events in events_by_month.items():
            if events:
                intensities = [e['intensity'] for e in events]
                pattern = SeasonalPattern(
                    location=(lat, lon),
                    variable=variable,
                    month_ranges=[(month, month)],
                    typical_value=np.mean(intensities),
                    typical_std=np.std(intensities),
                    peak_value=np.max(intensities),
                    occurrence_probability=len(events) / 5.0,  # Simplified
                )
                patterns.append(pattern)
        
        return patterns


# ─────────────────────────────────────────────────────────────
# EMBEDDING MEMORY (FAISS)
# ─────────────────────────────────────────────────────────────

class EmbeddingMemory:
    """Dense vector storage with FAISS-like indexing."""
    
    def __init__(self, dimension: int = 128, index_path: str = 'embeddings/'):
        self.dimension = dimension
        self.index_path = Path(index_path)
        self.index_path.mkdir(parents=True, exist_ok=True)
        self.embeddings_file = self.index_path / 'embeddings.npy'
        self.metadata_file = self.index_path / 'metadata.json'
        self.embeddings = np.empty((0, dimension), dtype=np.float32)
        self.metadata = {}
        self.logger = logger
        self._load()
    
    def _load(self):
        """Load embeddings from disk."""
        if self.embeddings_file.exists():
            self.embeddings = np.load(self.embeddings_file)
        if self.metadata_file.exists():
            with open(self.metadata_file) as f:
                self.metadata = json.load(f)
    
    def add_embedding(self, item_id: str, embedding: np.ndarray):
        """Add embedding to index."""
        if len(embedding) != self.dimension:
            raise ValueError(f"Expected {self.dimension}D, got {len(embedding)}D")
        
        # Append to embeddings array
        self.embeddings = np.vstack([self.embeddings, embedding.reshape(1, -1)])
        
        # Store metadata
        self.metadata[item_id] = {
            'index': len(self.embeddings) - 1,
            'added_at': datetime.now().isoformat(),
        }
        
        self._save()
    
    def search(self, query_embedding: np.ndarray, k: int = 10) -> List[Tuple[str, float]]:
        """Search for k nearest neighbors."""
        
        if len(self.embeddings) == 0:
            return []
        
        # Compute cosine similarities
        query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
        embeddings_norm = self.embeddings / (np.linalg.norm(self.embeddings, axis=1, keepdims=True) + 1e-8)
        
        similarities = np.dot(embeddings_norm, query_norm)
        
        # Get top-k
        top_indices = np.argsort(similarities)[-k:][::-1]
        
        # Map back to item IDs
        results = []
        index_to_id = {v['index']: k for k, v in self.metadata.items()}
        
        for idx in top_indices:
            if idx in index_to_id:
                item_id = index_to_id[idx]
                similarity = float(similarities[idx])
                results.append((item_id, similarity))
        
        return results
    
    def _save(self):
        """Save embeddings to disk."""
        np.save(self.embeddings_file, self.embeddings)
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)


# ─────────────────────────────────────────────────────────────
# PATTERN LEARNER
# ─────────────────────────────────────────────────────────────

class PatternLearner:
    """Extract recurring patterns from event sequences."""
    
    def __init__(self, event_memory: EventMemory):
        self.event_memory = event_memory
        self.logger = logger
        self.patterns: Dict[str, Pattern] = {}
    
    def extract_patterns(self, event_window: List[Event]) -> List[Pattern]:
        """Detect patterns in event sequence."""
        
        if len(event_window) < 3:
            return []
        
        # Group consecutive events of same type
        patterns = []
        i = 0
        while i < len(event_window):
            event_type = event_window[i].event_type
            group = [event_window[i]]
            
            j = i + 1
            while j < len(event_window) and event_window[j].event_type == event_type:
                group.append(event_window[j])
                j += 1
            
            # Analyze group
            if len(group) >= 2:
                pattern = self._analyze_group(event_type, group)
                patterns.append(pattern)
            
            i = j
        
        return patterns
    
    def _analyze_group(self, event_type: str, events: List[Event]) -> Pattern:
        """Analyze group of same-type events."""
        
        pattern_id = f"pattern_{event_type}_{len(self.patterns)}"
        
        # Duration stats
        durations = [e.duration_days for e in events]
        avg_duration = np.mean(durations)
        
        # Spatial scale (approximate from area)
        spatial_scales = []
        for e in events:
            lat_extent = abs(e.lat_max - e.lat_min) * 111  # km per degree
            lon_extent = abs(e.lon_max - e.lon_min) * 111 * np.cos(np.radians((e.lat_min + e.lat_max) / 2))
            scale = np.sqrt(lat_extent**2 + lon_extent**2)
            spatial_scales.append(scale)
        avg_spatial = np.mean(spatial_scales)
        
        pattern = Pattern(
            id=pattern_id,
            name=f"{event_type}_pattern",
            event_types=[event_type],
            typical_duration_days=int(avg_duration),
            spatial_scale_km=avg_spatial,
            frequency_per_year=len(events) / 1.0,  # Simplified
            precursors=[],
            typical_impact=f"Marine {event_type}",
        )
        
        self.patterns[pattern_id] = pattern
        return pattern
    
    def forecast_pattern_occurrence(self,
                                   pattern: Pattern,
                                   current_state: Dict) -> Dict:
        """Estimate probability of pattern occurring given current state."""
        
        # Simplified: always return 0.5
        return {
            'pattern_id': pattern.id,
            'probability': 0.5,
            'expected_days_to_occurrence': 7,
            'confidence': 0.6,
        }


# ─────────────────────────────────────────────────────────────
# ECHO ORCHESTRATOR
# ─────────────────────────────────────────────────────────────

class ECHO:
    """Main ECHO module orchestrator."""
    
    def __init__(self, db_path: str = 'events.db'):
        self.event_memory = EventMemory(db_path)
        self.embedding_memory = EmbeddingMemory()
        self.pattern_learner = PatternLearner(self.event_memory)
        self.logger = logger
    
    def record_and_learn(self,
                        event_type: str,
                        timestamp: datetime,
                        lat_min: float, lat_max: float,
                        lon_min: float, lon_max: float,
                        depth_min: float, depth_max: float,
                        intensity: float,
                        duration_days: int) -> str:
        """Record event and update embeddings."""
        
        event = self.event_memory.record_event(
            event_type=event_type,
            timestamp=timestamp,
            lat_min=lat_min, lat_max=lat_max,
            lon_min=lon_min, lon_max=lon_max,
            depth_min=depth_min, depth_max=depth_max,
            intensity=intensity,
            duration_days=duration_days,
        )
        
        # Generate and store embedding
        embedding = self.event_memory.get_event_embedding(event)
        self.embedding_memory.add_embedding(event.id, embedding)
        
        self.logger.info(f"ECHO: Recorded and embedded event {event.id}")
        return event.id
    
    def find_similar_events(self, event: Event, k: int = 5) -> List[Event]:
        """Find similar past events."""
        return self.event_memory.query_similar_events(event, k=k)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Example usage
    echo = ECHO()
    # event_id = echo.record_and_learn(
    #     event_type='marine_heatwave',
    #     timestamp=datetime.now(),
    #     lat_min=10, lat_max=15,
    #     lon_min=80, lon_max=85,
    #     depth_min=0, depth_max=100,
    #     intensity=7.5,
    #     duration_days=30,
    # )
    # print(f"Event ID: {event_id}")
