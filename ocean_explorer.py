"""
OceanVerse Ocean Explorer Module
Handles ROV/AUV planning, dive visualization, and mission timeline management.
"""

import json
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import numpy as np

logger = logging.getLogger(__name__)


class OceanExplorer:
    """Ocean exploration planning and visualization engine."""
    
    def __init__(self, db_connection=None):
        """Initialize Ocean Explorer."""
        self.db = db_connection
        self.missions = {}
        self.rov_fleet = {}
        self.auv_fleet = {}
        self.active_dives = []
        
    def plan_rov_mission(self, 
                        mission_name: str,
                        target_depth: float,
                        coordinates: Tuple[float, float],
                        duration_hours: float,
                        waypoints: List[Tuple[float, float, float]],
                        objectives: List[str]) -> Dict:
        """
        Plan ROV (Remotely Operated Vehicle) mission.
        
        Args:
            mission_name: Name of the mission
            target_depth: Target depth in meters
            coordinates: Starting coordinates (lat, lon)
            duration_hours: Expected mission duration
            waypoints: List of (lat, lon, depth) waypoints
            objectives: List of mission objectives
            
        Returns:
            Mission plan dictionary
        """
        mission_id = f"rov_{datetime.now().timestamp()}"
        
        mission = {
            'id': mission_id,
            'type': 'ROV',
            'name': mission_name,
            'start_coordinates': coordinates,
            'target_depth': target_depth,
            'duration_hours': duration_hours,
            'waypoints': waypoints,
            'objectives': objectives,
            'created_at': datetime.now().isoformat(),
            'status': 'planned',
            'route_distance_km': self._calculate_route_distance(waypoints),
            'estimated_battery_usage': self._estimate_battery(duration_hours),
            'data_collection_points': len(waypoints),
        }
        
        self.missions[mission_id] = mission
        logger.info(f"ROV mission '{mission_name}' planned: {mission_id}")
        return mission
    
    def plan_auv_mission(self,
                        mission_name: str,
                        grid_bounds: Dict[str, float],
                        survey_depth: float,
                        survey_type: str,
                        sampling_interval_meters: float) -> Dict:
        """
        Plan AUV (Autonomous Underwater Vehicle) mission.
        
        Args:
            mission_name: Name of the mission
            grid_bounds: Survey area bounds {north, south, east, west}
            survey_depth: Depth for survey
            survey_type: Type of survey (bathymetry, thermal, chemical, etc.)
            sampling_interval_meters: Distance between samples
            
        Returns:
            Mission plan dictionary
        """
        mission_id = f"auv_{datetime.now().timestamp()}"
        
        grid_size = self._calculate_grid_coverage(grid_bounds, sampling_interval_meters)
        
        mission = {
            'id': mission_id,
            'type': 'AUV',
            'name': mission_name,
            'grid_bounds': grid_bounds,
            'survey_depth': survey_depth,
            'survey_type': survey_type,
            'sampling_interval': sampling_interval_meters,
            'created_at': datetime.now().isoformat(),
            'status': 'planned',
            'grid_points': grid_size['points'],
            'estimated_duration_hours': grid_size['duration'],
            'coverage_km2': grid_size['coverage'],
        }
        
        self.missions[mission_id] = mission
        logger.info(f"AUV mission '{mission_name}' planned: {mission_id}")
        return mission
    
    def start_dive(self, mission_id: str, vehicle_id: str) -> Dict:
        """
        Start a dive mission.
        
        Args:
            mission_id: ID of the mission to start
            vehicle_id: ID of the vehicle executing the mission
            
        Returns:
            Dive session dictionary
        """
        if mission_id not in self.missions:
            raise ValueError(f"Mission {mission_id} not found")
        
        dive_id = f"dive_{datetime.now().timestamp()}"
        
        dive = {
            'id': dive_id,
            'mission_id': mission_id,
            'vehicle_id': vehicle_id,
            'start_time': datetime.now().isoformat(),
            'status': 'in_progress',
            'current_depth': 0,
            'current_location': None,
            'waypoint_index': 0,
            'data_collected': 0,
            'battery_level': 100,
        }
        
        self.active_dives.append(dive)
        logger.info(f"Dive started: {dive_id} using {vehicle_id}")
        return dive
    
    def update_dive_status(self, dive_id: str, 
                          current_depth: float,
                          current_location: Tuple[float, float],
                          battery_level: float,
                          data_collected: int) -> Dict:
        """Update active dive status."""
        for dive in self.active_dives:
            if dive['id'] == dive_id:
                dive['current_depth'] = current_depth
                dive['current_location'] = current_location
                dive['battery_level'] = battery_level
                dive['data_collected'] = data_collected
                dive['last_update'] = datetime.now().isoformat()
                return dive
        
        raise ValueError(f"Dive {dive_id} not found")
    
    def end_dive(self, dive_id: str) -> Dict:
        """End a dive mission."""
        for i, dive in enumerate(self.active_dives):
            if dive['id'] == dive_id:
                dive['status'] = 'completed'
                dive['end_time'] = datetime.now().isoformat()
                completed_dive = self.active_dives.pop(i)
                logger.info(f"Dive completed: {dive_id}")
                return completed_dive
        
        raise ValueError(f"Dive {dive_id} not found")
    
    def get_mission_timeline(self, mission_id: str) -> Dict:
        """Get detailed timeline for a mission."""
        if mission_id not in self.missions:
            raise ValueError(f"Mission {mission_id} not found")
        
        mission = self.missions[mission_id]
        timeline = {
            'mission_id': mission_id,
            'name': mission['name'],
            'type': mission['type'],
            'created_at': mission['created_at'],
            'status': mission['status'],
            'waypoints': mission.get('waypoints', []),
            'objectives': mission.get('objectives', []),
            'estimated_duration': mission.get('duration_hours', mission.get('estimated_duration_hours')),
            'depth_profile': self._generate_depth_profile(mission),
            'milestones': [
                {'time_hours': 0, 'event': 'Departure'},
                {'time_hours': mission.get('duration_hours', mission.get('estimated_duration_hours')) / 2, 'event': 'Midpoint'},
                {'time_hours': mission.get('duration_hours', mission.get('estimated_duration_hours')), 'event': 'Return'},
            ]
        }
        
        return timeline
    
    def visualize_dive(self, dive_id: str) -> Dict:
        """Generate visualization data for a dive."""
        dive = None
        for d in self.active_dives:
            if d['id'] == dive_id:
                dive = d
                break
        
        if not dive:
            raise ValueError(f"Active dive {dive_id} not found")
        
        return {
            'dive_id': dive_id,
            'current_depth': dive['current_depth'],
            'current_location': dive['current_location'],
            'battery_level': dive['battery_level'],
            'data_collected': dive['data_collected'],
            'status': dive['status'],
            'trail': self._generate_dive_trail(dive),
        }
    
    # Helper methods
    
    def _calculate_route_distance(self, waypoints: List[Tuple[float, float, float]]) -> float:
        """Calculate total route distance in km."""
        if len(waypoints) < 2:
            return 0.0
        
        total_distance = 0.0
        for i in range(len(waypoints) - 1):
            lat1, lon1, _ = waypoints[i]
            lat2, lon2, _ = waypoints[i + 1]
            # Simplified distance calculation (should use proper geodetic distance)
            distance = np.sqrt((lat2 - lat1) ** 2 + (lon2 - lon1) ** 2) * 111  # rough km conversion
            total_distance += distance
        
        return round(total_distance, 2)
    
    def _estimate_battery(self, duration_hours: float) -> float:
        """Estimate battery usage percentage."""
        # Simplified: assume 95% battery usage per hour
        return min(100.0, duration_hours * 0.95)
    
    def _calculate_grid_coverage(self, grid_bounds: Dict[str, float], 
                                sampling_interval: float) -> Dict:
        """Calculate grid coverage statistics."""
        north = grid_bounds.get('north', 0)
        south = grid_bounds.get('south', 0)
        east = grid_bounds.get('east', 0)
        west = grid_bounds.get('west', 0)
        
        lat_distance = (north - south) * 111  # km
        lon_distance = (east - west) * 111 * np.cos(np.radians((north + south) / 2))
        
        points_lat = int(lat_distance * 1000 / sampling_interval)
        points_lon = int(lon_distance * 1000 / sampling_interval)
        total_points = points_lat * points_lon
        
        coverage_km2 = lat_distance * lon_distance
        estimated_hours = (total_points * sampling_interval) / 1000  # rough estimate
        
        return {
            'points': total_points,
            'coverage': round(coverage_km2, 2),
            'duration': round(estimated_hours, 2),
        }
    
    def _generate_depth_profile(self, mission: Dict) -> List[float]:
        """Generate depth profile for mission visualization."""
        if mission['type'] == 'ROV':
            waypoints = mission.get('waypoints', [])
            return [wp[2] for wp in waypoints]  # Extract depth from waypoints
        else:
            return [mission['survey_depth']] * 10  # Flat profile for AUV
    
    def _generate_dive_trail(self, dive: Dict) -> List[Dict]:
        """Generate dive trail for visualization."""
        if dive['current_location'] is None:
            return []
        
        # Return simple trail (in real implementation, would track full path)
        return [
            {
                'timestamp': dive['last_update'] if 'last_update' in dive else dive['start_time'],
                'latitude': dive['current_location'][0],
                'longitude': dive['current_location'][1],
                'depth': dive['current_depth'],
            }
        ]
