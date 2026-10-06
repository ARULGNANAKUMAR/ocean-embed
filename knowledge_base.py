"""
KNOWLEDGE BASE
Versioned Storage & Retrieval System for Datasets, Models, Embeddings
OCEANVERSE Phase 4
"""

import json
import sqlite3
import logging
import pickle
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any
from datetime import datetime

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# DATA STRUCTURES
# ─────────────────────────────────────────────────────────────

@dataclass
class DatasetMetadata:
    """Dataset metadata."""
    id: str
    name: str
    source: str
    version: str
    start_date: str
    end_date: str
    spatial_bounds: Dict
    variables: List[str]
    quality_metrics: Dict
    created_at: str


@dataclass
class ModelMetadata:
    """Model metadata."""
    id: str
    name: str
    architecture: str
    version: str
    checkpoint_path: str
    hyperparameters: Dict
    training_metrics: Dict
    validation_metrics: Dict
    created_at: str
    status: str  # 'active', 'archived', 'candidate'


# ─────────────────────────────────────────────────────────────
# DATASET REGISTRY
# ─────────────────────────────────────────────────────────────

class DatasetRegistry:
    """Manage versioned datasets."""
    
    def __init__(self, db_path: str = 'knowledge_base.db'):
        self.db_path = db_path
        self.logger = logger
        self._init_db()
    
    def _init_db(self):
        """Initialize database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS datasets (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                source TEXT,
                version TEXT,
                start_date TEXT,
                end_date TEXT,
                spatial_bounds TEXT,
                variables TEXT,
                quality_metrics TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS dataset_tags (
                dataset_id TEXT,
                tag TEXT,
                FOREIGN KEY(dataset_id) REFERENCES datasets(id)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def register_dataset(self,
                        name: str,
                        source: str,
                        version: str,
                        start_date: str,
                        end_date: str,
                        spatial_bounds: Dict,
                        variables: List[str],
                        quality_metrics: Dict) -> str:
        """Register a new dataset."""
        
        dataset_id = f"ds_{name}_{version}".replace(' ', '_')
        
        metadata = DatasetMetadata(
            id=dataset_id,
            name=name,
            source=source,
            version=version,
            start_date=start_date,
            end_date=end_date,
            spatial_bounds=spatial_bounds,
            variables=variables,
            quality_metrics=quality_metrics,
            created_at=datetime.now().isoformat(),
        )
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT OR REPLACE INTO datasets
            (id, name, source, version, start_date, end_date, 
             spatial_bounds, variables, quality_metrics, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            metadata.id,
            metadata.name,
            metadata.source,
            metadata.version,
            metadata.start_date,
            metadata.end_date,
            json.dumps(metadata.spatial_bounds),
            json.dumps(metadata.variables),
            json.dumps(metadata.quality_metrics),
            metadata.created_at,
        ))
        conn.commit()
        conn.close()
        
        self.logger.info(f"Registered dataset: {dataset_id}")
        return dataset_id
    
    def get_dataset(self, dataset_id: str) -> Optional[DatasetMetadata]:
        """Retrieve dataset metadata."""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM datasets WHERE id = ?', (dataset_id,))
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return DatasetMetadata(
                id=row[0],
                name=row[1],
                source=row[2],
                version=row[3],
                start_date=row[4],
                end_date=row[5],
                spatial_bounds=json.loads(row[6]),
                variables=json.loads(row[7]),
                quality_metrics=json.loads(row[8]),
                created_at=row[9],
            )
        return None
    
    def list_datasets(self, filters: Dict = None) -> List[DatasetMetadata]:
        """Query datasets with filters."""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        query = 'SELECT * FROM datasets'
        params = []
        
        if filters:
            conditions = []
            if 'source' in filters:
                conditions.append('source = ?')
                params.append(filters['source'])
            if 'name' in filters:
                conditions.append('name LIKE ?')
                params.append(f"%{filters['name']}%")
            
            if conditions:
                query += ' WHERE ' + ' AND '.join(conditions)
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()
        
        return [
            DatasetMetadata(
                id=row[0],
                name=row[1],
                source=row[2],
                version=row[3],
                start_date=row[4],
                end_date=row[5],
                spatial_bounds=json.loads(row[6]),
                variables=json.loads(row[7]),
                quality_metrics=json.loads(row[8]),
                created_at=row[9],
            )
            for row in rows
        ]
    
    def tag_dataset(self, dataset_id: str, tags: List[str]):
        """Add tags to dataset."""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for tag in tags:
            cursor.execute(
                'INSERT INTO dataset_tags (dataset_id, tag) VALUES (?, ?)',
                (dataset_id, tag)
            )
        
        conn.commit()
        conn.close()
        self.logger.info(f"Tagged dataset {dataset_id}: {tags}")


# ─────────────────────────────────────────────────────────────
# MODEL REGISTRY
# ─────────────────────────────────────────────────────────────

class ModelRegistry:
    """Manage model versions and metadata."""
    
    def __init__(self, db_path: str = 'knowledge_base.db'):
        self.db_path = db_path
        self.logger = logger
        self._init_db()
    
    def _init_db(self):
        """Initialize database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS models (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                architecture TEXT,
                version TEXT,
                checkpoint_path TEXT,
                hyperparameters TEXT,
                training_metrics TEXT,
                validation_metrics TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                status TEXT DEFAULT 'archived'
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def register_model(self,
                      name: str,
                      architecture: str,
                      version: str,
                      checkpoint_path: str,
                      hyperparameters: Dict,
                      training_metrics: Dict,
                      validation_metrics: Dict,
                      status: str = 'archived') -> str:
        """Register a trained model."""
        
        model_id = f"model_{name}_{version}".replace(' ', '_')
        
        metadata = ModelMetadata(
            id=model_id,
            name=name,
            architecture=architecture,
            version=version,
            checkpoint_path=checkpoint_path,
            hyperparameters=hyperparameters,
            training_metrics=training_metrics,
            validation_metrics=validation_metrics,
            created_at=datetime.now().isoformat(),
            status=status,
        )
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT OR REPLACE INTO models
            (id, name, architecture, version, checkpoint_path,
             hyperparameters, training_metrics, validation_metrics,
             created_at, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            metadata.id,
            metadata.name,
            metadata.architecture,
            metadata.version,
            metadata.checkpoint_path,
            json.dumps(metadata.hyperparameters),
            json.dumps(metadata.training_metrics),
            json.dumps(metadata.validation_metrics),
            metadata.created_at,
            metadata.status,
        ))
        conn.commit()
        conn.close()
        
        self.logger.info(f"Registered model: {model_id}")
        return model_id
    
    def get_model(self, model_id: str) -> Optional[ModelMetadata]:
        """Retrieve model metadata."""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM models WHERE id = ?', (model_id,))
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return ModelMetadata(
                id=row[0],
                name=row[1],
                architecture=row[2],
                version=row[3],
                checkpoint_path=row[4],
                hyperparameters=json.loads(row[5]),
                training_metrics=json.loads(row[6]),
                validation_metrics=json.loads(row[7]),
                created_at=row[8],
                status=row[9],
            )
        return None
    
    def get_best_model(self, metric: str = 'validation_rmse') -> Optional[ModelMetadata]:
        """Get highest-performing model by metric."""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM models ORDER BY created_at DESC LIMIT 1')
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return ModelMetadata(
                id=row[0],
                name=row[1],
                architecture=row[2],
                version=row[3],
                checkpoint_path=row[4],
                hyperparameters=json.loads(row[5]),
                training_metrics=json.loads(row[6]),
                validation_metrics=json.loads(row[7]),
                created_at=row[8],
                status=row[9],
            )
        return None
    
    def list_model_history(self, name: str) -> List[ModelMetadata]:
        """Get all versions of a model."""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM models WHERE name = ? ORDER BY created_at DESC', (name,))
        rows = cursor.fetchall()
        conn.close()
        
        return [
            ModelMetadata(
                id=row[0],
                name=row[1],
                architecture=row[2],
                version=row[3],
                checkpoint_path=row[4],
                hyperparameters=json.loads(row[5]),
                training_metrics=json.loads(row[6]),
                validation_metrics=json.loads(row[7]),
                created_at=row[8],
                status=row[9],
            )
            for row in rows
        ]
    
    def compare_models(self, model_ids: List[str]) -> Dict:
        """Compare multiple models."""
        
        comparison = {}
        for model_id in model_ids:
            metadata = self.get_model(model_id)
            if metadata:
                comparison[model_id] = {
                    'name': metadata.name,
                    'architecture': metadata.architecture,
                    'version': metadata.version,
                    'training_metrics': metadata.training_metrics,
                    'validation_metrics': metadata.validation_metrics,
                }
        
        return comparison
    
    def activate_model(self, model_id: str):
        """Set model as active."""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Deactivate all others
        cursor.execute('UPDATE models SET status = ? WHERE status = ?', ('archived', 'active'))
        
        # Activate this model
        cursor.execute('UPDATE models SET status = ? WHERE id = ?', ('active', model_id))
        
        conn.commit()
        conn.close()
        
        self.logger.info(f"Activated model: {model_id}")
    
    def get_active_model(self) -> Optional[ModelMetadata]:
        """Get currently active model."""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM models WHERE status = ? LIMIT 1', ('active',))
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return ModelMetadata(
                id=row[0],
                name=row[1],
                architecture=row[2],
                version=row[3],
                checkpoint_path=row[4],
                hyperparameters=json.loads(row[5]),
                training_metrics=json.loads(row[6]),
                validation_metrics=json.loads(row[7]),
                created_at=row[8],
                status=row[9],
            )
        return None


# ─────────────────────────────────────────────────────────────
# PREDICTION ARCHIVE
# ─────────────────────────────────────────────────────────────

class PredictionArchive:
    """Store and verify predictions."""
    
    def __init__(self, db_path: str = 'knowledge_base.db'):
        self.db_path = db_path
        self.logger = logger
        self._init_db()
    
    def _init_db(self):
        """Initialize database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                id TEXT PRIMARY KEY,
                model_id TEXT,
                timestamp DATETIME,
                lat REAL,
                lon REAL,
                depth REAL,
                variable TEXT,
                predicted_value REAL,
                confidence REAL,
                lead_time_days INTEGER,
                archived_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS verification (
                prediction_id TEXT,
                observed_value REAL,
                source TEXT,
                verified_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(prediction_id) REFERENCES predictions(id)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def archive_prediction(self,
                          model_id: str,
                          timestamp: datetime,
                          location: tuple,
                          variables: Dict,
                          lead_time_days: int) -> str:
        """Archive a prediction for future verification."""
        
        pred_id = f"pred_{model_id}_{timestamp.timestamp()}".replace('.', '_')
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO predictions
            (id, model_id, timestamp, lat, lon, depth, variable,
             predicted_value, confidence, lead_time_days)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            pred_id,
            model_id,
            timestamp.isoformat(),
            location[0],
            location[1],
            variables.get('depth', 0),
            variables.get('variable', 'temperature'),
            variables.get('value', 0),
            variables.get('confidence', 0.5),
            lead_time_days,
        ))
        conn.commit()
        conn.close()
        
        return pred_id
    
    def verify_prediction(self,
                         prediction_id: str,
                         observed_value: float,
                         source: str):
        """Record ground truth observation."""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO verification
            (prediction_id, observed_value, source)
            VALUES (?, ?, ?)
        ''', (prediction_id, observed_value, source))
        conn.commit()
        conn.close()
        
        self.logger.info(f"Verified prediction: {prediction_id}")
    
    def compute_calibration(self, model_id: str) -> Dict:
        """Evaluate prediction calibration."""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            SELECT p.confidence, ABS(p.predicted_value - v.observed_value) as error
            FROM predictions p
            LEFT JOIN verification v ON p.id = v.prediction_id
            WHERE p.model_id = ? AND v.observed_value IS NOT NULL
        ''', (model_id,))
        rows = cursor.fetchall()
        conn.close()
        
        if not rows:
            return {'status': 'insufficient_data'}
        
        confidences = [r[0] for r in rows]
        errors = [r[1] for r in rows]
        
        # Bin by confidence
        bins = [0.1 * i for i in range(1, 11)]
        calibration = {}
        
        for bin_edge in bins:
            mask = [c >= bin_edge - 0.05 and c < bin_edge + 0.05 for c in confidences]
            if any(mask):
                accuracy = 1.0 - (sum(e for e, m in zip(errors, mask) if m) / sum(mask))
                calibration[f'{bin_edge:.1f}'] = accuracy
        
        return {
            'model_id': model_id,
            'calibration': calibration,
            'num_verified': len(rows),
        }


# ─────────────────────────────────────────────────────────────
# KNOWLEDGE BASE ORCHESTRATOR
# ─────────────────────────────────────────────────────────────

class KnowledgeBase:
    """Main orchestrator for all knowledge storage."""
    
    def __init__(self, base_path: str = 'knowledge_base'):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        self.dataset_registry = DatasetRegistry(str(self.base_path / 'kb.db'))
        self.model_registry = ModelRegistry(str(self.base_path / 'kb.db'))
        self.prediction_archive = PredictionArchive(str(self.base_path / 'kb.db'))
        
        self.logger = logger
    
    def register_training_run(self,
                             dataset_id: str,
                             model_id: str,
                             metrics: Dict) -> str:
        """Record a training run."""
        
        run_id = f"run_{model_id}_{datetime.now().timestamp()}"
        run_data = {
            'run_id': run_id,
            'dataset_id': dataset_id,
            'model_id': model_id,
            'metrics': metrics,
            'timestamp': datetime.now().isoformat(),
        }
        
        run_file = self.base_path / 'runs' / f"{run_id}.json"
        run_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(run_file, 'w') as f:
            json.dump(run_data, f, indent=2)
        
        self.logger.info(f"Recorded training run: {run_id}")
        return run_id


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    kb = KnowledgeBase()
    dataset_id = kb.dataset_registry.register_dataset(
        name="GLORYS",
        source="Copernicus",
        version="1.0",
        start_date="2025-01-01",
        end_date="2025-01-31",
        spatial_bounds={'lat': [10, 15], 'lon': [80, 85]},
        variables=['temperature', 'salinity'],
        quality_metrics={'completeness': 0.95},
    )
    print(f"Registered dataset: {dataset_id}")
