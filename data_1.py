from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import random
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
import logging
import copy
from fastapi import APIRouter
import asyncio
from fastapi import Request

app = FastAPI()

# Add CORS middleware for cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



# Base template for train data generation
TRAIN_DATA_TEMPLATE = {
    "type": "train_snapshot",
    "timestamp": "",
    "payload": []
}

# Station and section configurations - EXTENDED with branch lines
STATIONS = ["STN_A", "STN_B", "STN_C", "STN_D", "STN_E", "STN_F"]
SECTIONS = [
    {"id": "SEC_1", "start": "STN_A", "end": "STN_B", "length_km": 8.5, "capacity": 2, "max_speed_kmh": 120, "track_type": "double"},
    {"id": "SEC_2", "start": "STN_B", "end": "STN_C", "length_km": 6.2, "capacity": 1, "max_speed_kmh": 100, "track_type": "single"},
    {"id": "SEC_3", "start": "STN_C", "end": "STN_D", "length_km": 7.8, "capacity": 2, "max_speed_kmh": 140, "track_type": "double"},
    {"id": "SEC_4", "start": "STN_D", "end": "STN_E", "length_km": 5.3, "capacity": 1, "max_speed_kmh": 110, "track_type": "single"},
    {"id": "SEC_5", "start": "STN_E", "end": "STN_F", "length_km": 9.1, "capacity": 3, "max_speed_kmh": 160, "track_type": "double"},
    # Branch line for alternate routing
    {"id": "SEC_6", "start": "STN_B", "end": "STN_E", "length_km": 12.0, "capacity": 1, "max_speed_kmh": 90, "track_type": "single"}
]

TRAIN_TYPES = [
    {"name": "Express", "priority": 5, "speed_range": (140, 160)},
    {"name": "Freight", "priority": 2, "speed_range": (80, 100)},
    {"name": "Local", "priority": 3, "speed_range": (100, 120)},
    {"name": "High-Speed", "priority": 5, "speed_range": (160, 180)}
]

# Global train state management
class TrainState:
    def __init__(self):
        self.trains: Dict[str, Dict[str, Any]] = {}
        self.section_disruptions: Dict[str, Dict[str, Any]] = {}
        self.occupied_sections: Dict[str, List[str]] = {}  # section_id -> [train_ids]
        self.initialized = False
        self.start_time = datetime.now()
        self.last_update = datetime.now()

train_state = TrainState()

# Pydantic models for request/response validation
class TrainSnapshotResponse(BaseModel):
    type: str
    timestamp: str
    payload: List[Dict[str, Any]]

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    total_trains: int
    active_trains: int
    active_disruptions: int
    disrupted_sections: List[str]

def get_section_by_id(section_id: str) -> Optional[Dict[str, Any]]:
    """Get section data by ID"""
    for section in SECTIONS:
        if section["id"] == section_id:
            return section
    return None

def get_station_connections(station: str) -> List[str]:
    """Get all sections connected to a station"""
    connections = []
    for section in SECTIONS:
        if section["start"] == station:
            connections.append(section["id"])
        elif section["end"] == station:
            connections.append(section["id"])
    return connections

def find_route_to_destination(current_section_id: str, target_station: str, direction: str) -> List[str]:
    """Find route from current section to target station"""
    current_section = get_section_by_id(current_section_id)
    if not current_section:
        return []
    
    # Simple routing logic - can be enhanced with pathfinding
    route = []
    current_station = current_section["end"] if direction == "forward" else current_section["start"]
    
    # If we're at the target station, we're done
    if current_station == target_station:
        return []
    
    # Find next sections based on direction and destination
    for section in SECTIONS:
        if direction == "forward":
            if section["start"] == current_station:
                route.append(section["id"])
                break
        else:
            if section["end"] == current_station:
                route.append(section["id"])
                break
    
    return route

def get_next_section_towards_destination(current_section_id: str, target_station: str, direction: str) -> Optional[str]:
    """Get the next section towards destination, considering disruptions"""
    current_section = get_section_by_id(current_section_id)
    if not current_section:
        return None
    
    current_station = current_section["end"] if direction == "forward" else current_section["start"]
    
    # If we're at destination, no next section
    if current_station == target_station:
        return None
    
    # Find potential next sections
    candidates = []
    for section in SECTIONS:
        if direction == "forward" and section["start"] == current_station:
            candidates.append(section["id"])
        elif direction == "backward" and section["end"] == current_station:
            candidates.append(section["id"])
    
    # Filter out disrupted sections and choose best available
    available_sections = []
    for section_id in candidates:
        if not is_section_disrupted(section_id) and not is_section_at_capacity(section_id):
            available_sections.append(section_id)
    
    if available_sections:
        # Prefer main route, fall back to alternatives
        main_route_sections = ["SEC_1", "SEC_2", "SEC_3", "SEC_4", "SEC_5"]
        for section_id in available_sections:
            if section_id in main_route_sections:
                return section_id
        return available_sections[0]  # Use any available alternative
    
    return candidates[0] if candidates else None  # Return even if disrupted (train will wait)

def is_section_disrupted(section_id: str) -> bool:
    """Check if section has active disruptions"""
    disruption = train_state.section_disruptions.get(section_id)
    if not disruption:
        return False
    
    # Check if disruption is still active
    end_time = datetime.fromisoformat(disruption["end_time"])
    return datetime.now() < end_time

def is_section_at_capacity(section_id: str) -> bool:
    """Check if section is at capacity"""
    section = get_section_by_id(section_id)
    if not section:
        return True
    
    occupied_trains = train_state.occupied_sections.get(section_id, [])
    return len(occupied_trains) >= section["capacity"]

def update_section_occupancy():
    """Update which trains occupy which sections"""
    train_state.occupied_sections.clear()
    
    for train_id, train_data in train_state.trains.items():
        if train_data["status"] not in ["Arrived", "Cancelled"]:
            section_id = train_data["current_location"]["section_id"]
            if section_id not in train_state.occupied_sections:
                train_state.occupied_sections[section_id] = []
            train_state.occupied_sections[section_id].append(train_id)

def generate_section_disruptions():
    """Randomly generate section disruptions"""
    # 15% chance of new disruption per update
    if random.random() < 0.15:
        available_sections = [s["id"] for s in SECTIONS if s["id"] not in train_state.section_disruptions]
        if available_sections:
            section_id = random.choice(available_sections)
            disruption_types = ["maintenance", "signal_failure", "track_work", "emergency"]
            disruption_type = random.choice(disruption_types)
            
            duration_minutes = random.randint(10, 60)
            start_time = datetime.now()
            end_time = start_time + timedelta(minutes=duration_minutes)
            
            train_state.section_disruptions[section_id] = {
                "type": disruption_type,
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "duration_minutes": duration_minutes,
                "severity": random.choice(["low", "medium", "high"])
            }
            
            logger.info(f"New disruption in {section_id}: {disruption_type} for {duration_minutes} minutes")
    
    # Remove expired disruptions
    expired_sections = []
    for section_id, disruption in train_state.section_disruptions.items():
        end_time = datetime.fromisoformat(disruption["end_time"])
        if datetime.now() >= end_time:
            expired_sections.append(section_id)
    
    for section_id in expired_sections:
        del train_state.section_disruptions[section_id]
        logger.info(f"Disruption cleared in {section_id}")

def calculate_position_progress(train_data: Dict[str, Any], time_elapsed_min: float) -> Dict[str, Any]:
    """Calculate train position based on time elapsed and speed"""
    current_section = get_section_by_id(train_data["current_location"]["section_id"])
    if not current_section:
        return train_data
    
    # Check for stochastic delays (breakdowns, signal stops)
    if random.random() < 0.05:  # 5% chance per update
        train_data["breakdown_until"] = (datetime.now() + timedelta(minutes=random.randint(5, 20))).isoformat()
        logger.info(f"Train {train_data['train_id']} breakdown for emergency stop")
    
    # Check if train is in breakdown
    if train_data.get("breakdown_until"):
        breakdown_end = datetime.fromisoformat(train_data["breakdown_until"])
        if datetime.now() < breakdown_end:
            return train_data  # No movement during breakdown
        else:
            del train_data["breakdown_until"]  # Clear breakdown
    
    # Check if section is disrupted - trains must wait
    if is_section_disrupted(current_section["id"]):
        train_data["status"] = "Waiting - Section Disrupted"
        return train_data
    
    # Check capacity constraints for single-track sections
    if current_section["capacity"] == 1:
        occupied_trains = train_state.occupied_sections.get(current_section["id"], [])
        if len(occupied_trains) > 1 and train_data["train_id"] not in occupied_trains[:1]:
            train_data["status"] = "Waiting - Traffic"
            return train_data
    
    # Base speed calculation with priority considerations
    train_type_data = next((t for t in TRAIN_TYPES if t["name"] == train_data["type"]), TRAIN_TYPES[0])
    base_speed = train_data["max_speed_kmh"]
    
    # Apply speed modifiers
    speed_modifier = 1.0
    if train_data["status"] in ["Delayed", "Waiting - Section Disrupted", "Waiting - Traffic"]:
        speed_modifier = 0.7
    elif train_data["status"] == "Cancelled":
        speed_modifier = 0.0
    
    # Section-specific speed restrictions
    section_disruption = train_state.section_disruptions.get(current_section["id"])
    if section_disruption and section_disruption["severity"] == "high":
        speed_modifier *= 0.4
    elif train_data.get("restricted_speed", False):
        speed_modifier *= 0.6
    
    effective_speed_kmh = min(base_speed * speed_modifier, current_section["max_speed_kmh"])
    
    # Calculate movement
    distance_traveled_km = (effective_speed_kmh / 60) * time_elapsed_min
    distance_traveled_m = distance_traveled_km * 1000
    
    current_position_m = train_data["current_location"]["position_m"]
    section_length_m = current_section["length_km"] * 1000
    new_position_m = current_position_m + distance_traveled_m
    
    # Check if train completed current section
    if new_position_m >= section_length_m:
        next_section_id = get_next_section_towards_destination(
            current_section["id"], 
            train_data["destination_station"],
            train_data["direction"]
        )
        
        if next_section_id:
            train_data["current_location"]["section_id"] = next_section_id
            train_data["current_location"]["position_m"] = new_position_m - section_length_m
            train_data["status"] = "On time"
        else:
            # Check if we're at destination
            current_station = current_section["end"] if train_data["direction"] == "forward" else current_section["start"]
            if current_station == train_data["destination_station"]:
                train_data["current_location"]["position_m"] = section_length_m
                train_data["status"] = "Arrived"
                train_data["actual_arrival"] = datetime.now().isoformat()
                logger.info(f"Train {train_data['train_id']} arrived at {train_data['destination_station']} (journey #{train_data.get('journey_count', 1)})")
            else:
                # No route available - train must wait
                train_data["current_location"]["position_m"] = section_length_m
                train_data["status"] = "Waiting - No Route"
    else:
        train_data["current_location"]["position_m"] = new_position_m
        if train_data["status"] in ["Waiting - Traffic", "Waiting - Section Disrupted"]:
            train_data["status"] = "On time"
    
    return train_data

def initialize_train(train_id: str) -> Dict[str, Any]:
    """Initialize a new train with random destination and direction"""
    now = datetime.now()
    train_type_data = random.choice(TRAIN_TYPES)
    
    # Random destination (not origin station)
    all_destinations = STATIONS[1:]  # Exclude STN_A as destination
    destination = random.choice(all_destinations)
    
    # Random direction
    direction = random.choice(["forward", "backward"])
    
    # Set starting position based on direction
    if direction == "forward":
        starting_section = "SEC_1"
        starting_position = random.randint(0, 500)
    else:
        starting_section = "SEC_5"  # Start from end for backward trains
        starting_position = random.randint(8600, 9100)  # Near end of section
        destination = random.choice(["STN_A", "STN_B", "STN_C"])  # Backward destinations
    
    train_data = {
        "train_id": train_id,
        "type": train_type_data["name"],
        "priority": train_type_data["priority"],
        "max_speed_kmh": random.randint(*train_type_data["speed_range"]),
        "length_m": random.randint(150, 400),
        "direction": direction,
        "destination_station": destination,
        "current_location": {
            "section_id": starting_section,
            "position_m": starting_position
        },
        "status": "On time",
        "actual_departure": (now - timedelta(minutes=random.randint(15, 45))).isoformat(),
        "actual_arrival": None,
        "restricted_speed": random.choice([True, False]) if random.random() < 0.2 else False,
        "journey_count": 1
    }
    
    return train_data

def reset_train_for_new_journey(train_data: Dict[str, Any]) -> Dict[str, Any]:
    """Reset an arrived train for a new journey"""
    now = datetime.now()
    train_type_data = random.choice(TRAIN_TYPES)
    
    # Random new destination and direction
    direction = random.choice(["forward", "backward"])
    
    if direction == "forward":
        starting_section = "SEC_1"
        starting_position = 0.0
        destinations = STATIONS[1:]  # B, C, D, E, F
    else:
        starting_section = "SEC_5"
        starting_position = 0.0
        destinations = ["STN_A", "STN_B", "STN_C"]  # Backward destinations
    
    destination = random.choice(destinations)
    
    logger.info(f"Train {train_data['train_id']} starting new {direction} journey to {destination}")
    
    return {
        "train_id": train_data["train_id"],
        "type": train_type_data["name"],
        "priority": train_type_data["priority"],
        "max_speed_kmh": random.randint(*train_type_data["speed_range"]),
        "length_m": random.randint(150, 400),
        "direction": direction,
        "destination_station": destination,
        "current_location": {
            "section_id": starting_section,
            "position_m": starting_position
        },
        "status": "On time",
        "actual_departure": now.isoformat(),
        "actual_arrival": None,
        "restricted_speed": random.choice([True, False]) if random.random() < 0.2 else False,
        "journey_count": train_data.get("journey_count", 0) + 1
    }

def update_train_data(train_data: Dict[str, Any], time_elapsed_min: float) -> Dict[str, Any]:
    """Update train data for the current time step"""
    if train_data["status"] == "Arrived":
        return reset_train_for_new_journey(train_data)
    
    train_data = calculate_position_progress(train_data, time_elapsed_min)
    
    # Dynamic status updates based on priority conflicts
    if random.random() < 0.2:
        current_section = get_section_by_id(train_data["current_location"]["section_id"])
        if current_section and current_section["capacity"] == 1:
            # Check for priority conflicts
            occupied_trains = train_state.occupied_sections.get(current_section["id"], [])
            if len(occupied_trains) > 1:
                train_priorities = [(tid, train_state.trains[tid]["priority"]) for tid in occupied_trains if tid in train_state.trains]
                train_priorities.sort(key=lambda x: x[1], reverse=True)  # Highest priority first
                
                # Lower priority trains get delayed
                for i, (tid, priority) in enumerate(train_priorities[1:], 1):
                    if tid == train_data["train_id"]:
                        train_data["status"] = "Delayed"
    
    # Random speed restrictions
    if random.random() < 0.1:
        train_data["restricted_speed"] = not train_data.get("restricted_speed", False)
    
    return train_data

def generate_train_bundle(train_data: Dict[str, Any]) -> Dict[str, Any]:
    """Generate enhanced train data bundle with disruption info"""
    current_section = get_section_by_id(train_data["current_location"]["section_id"])
    
    section_data = {
        "section_id": current_section["id"],
        "start_station": current_section["start"],
        "end_station": current_section["end"],
        "length_km": current_section["length_km"],
        "capacity": current_section["capacity"],
        "max_speed_kmh": current_section["max_speed_kmh"],
        "track_type": current_section["track_type"],
        "is_disrupted": is_section_disrupted(current_section["id"]),
        "occupancy_count": len(train_state.occupied_sections.get(current_section["id"], []))
    }
    
    # Enhanced signal data with priority handling
    occupied_trains = train_state.occupied_sections.get(current_section["id"], [])
    signal_data = {
        "block_id": f"BLK_{current_section['id'][4:]}01",
        "section_id": current_section["id"],
        "occupancy_status": "occupied" if occupied_trains else "free",
        "occupying_trains": len(occupied_trains),
        "signal_type": "automatic" if current_section["capacity"] > 1 else "manual",
        "headway_time_s": random.randint(60, 300),
        "priority_override": any(train_state.trains.get(tid, {}).get("priority", 1) >= 5 for tid in occupied_trains)
    }
    
    # Enhanced event data
    disruption = train_state.section_disruptions.get(current_section["id"])
    has_disruption = disruption is not None
    has_breakdown = train_data.get("breakdown_until") is not None
    
    event_data = {
        "event_type": "Section Disruption" if has_disruption else "Breakdown" if has_breakdown else 
                     "Delay" if train_data["status"] == "Delayed" else "Restriction" if train_data.get("restricted_speed") else "None",
        "train_id": train_data["train_id"],
        "section_id": current_section["id"],
        "timestamp": datetime.now().isoformat() if (has_disruption or has_breakdown or 
                    train_data["status"] == "Delayed" or train_data.get("restricted_speed")) else None,
        "disruption_details": disruption if has_disruption else None,
        "delay_duration_min": random.randint(5, 60) if train_data["status"] == "Delayed" else 0
    }
    
    return {
        "train": train_data,
        "section": section_data,
        "signal": signal_data,
        "event": event_data
    }

def update_train_state():
    """Update the global train state based on elapsed time"""
    global train_state
    
    current_time = datetime.now()
    time_since_last_update = (current_time - train_state.last_update).total_seconds() / 60
    
    if not train_state.initialized:
        logger.info("Initializing 10 trains (TR001-TR010) with random destinations and directions")
        for i in range(1, 11):
            train_id = f"TR{str(i).zfill(3)}"
            train_data = initialize_train(train_id)
            train_state.trains[train_id] = train_data
        train_state.initialized = True
        train_state.last_update = current_time
    else:
        # Generate section disruptions
        generate_section_disruptions()
        
        # Update section occupancy
        update_section_occupancy()
        
        # Update all trains
        for train_id, train_data in train_state.trains.items():
            train_state.trains[train_id] = update_train_data(train_data, time_since_last_update)
        
        train_state.last_update = current_time

def generate_train_snapshot():
    """Generate complete train snapshot with enhanced traffic management"""
    update_train_state()
    
    current_time = datetime.now()
    snapshot = copy.deepcopy(TRAIN_DATA_TEMPLATE)
    snapshot["timestamp"] = current_time.isoformat()
    snapshot["payload"] = []
    
    # Add system status
    snapshot["system_status"] = {
        "active_disruptions": len(train_state.section_disruptions),
        "disrupted_sections": list(train_state.section_disruptions.keys()),
        "section_occupancy": {k: len(v) for k, v in train_state.occupied_sections.items()}
    }
    
    # Generate bundles for all trains
    for train_id in sorted(train_state.trains.keys()):
        train_data = train_state.trains[train_id]
        bundle = generate_train_bundle(train_data)
        snapshot["payload"].append(bundle)
    
    return snapshot

# API Endpoints

@app.get("/")
async def get():
    return {
        "message": "Enhanced Traffic Management API",
        "endpoints": {
            "train_data": "/api/train-data",
            "health": "/health",
            "trains": "/trains",
            "reset": "/reset",
            "summary": "/api/train-data/summary",
            "disruptions": "/api/disruptions",
            "schedule": "/api/schedule",
            "optimization_results": "/api/optimization/results"
        },
        "features": [
            "Random destinations and bidirectional traffic",
            "Real-time section disruptions and maintenance",
            "Priority-based routing and conflict resolution",
            "Stochastic delays and breakdowns",
            "Dynamic rerouting with alternate paths",
            "Section capacity management"
        ]
    }

@app.get("/api/train-data", response_model=TrainSnapshotResponse)
async def get_train_data():
    """Get current train data snapshot - enhanced with traffic management"""
    try:
        train_data = generate_train_snapshot()
        return train_data
    except Exception as e:
        logger.error(f"Error generating train data: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate train data")

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint with enhanced statistics"""
    update_train_state()
    
    active_trains = sum(1 for train in train_state.trains.values() 
                       if train["status"] not in ["Arrived", "Cancelled"])
    
    return {
        "status": "healthy", 
        "timestamp": datetime.now().isoformat(),
        "total_trains": len(train_state.trains),
        "active_trains": active_trains,
        "active_disruptions": len(train_state.section_disruptions),
        "disrupted_sections": list(train_state.section_disruptions.keys())
    }
    



@app.get("/trains")
async def get_current_trains():
    """Get current state of all trains with routing info"""
    update_train_state()
    return {
        "trains": train_state.trains, 
        "section_occupancy": train_state.occupied_sections,
        "disruptions": train_state.section_disruptions,
        "initialized": train_state.initialized,
        "last_update": train_state.last_update.isoformat()
    }

@app.get("/api/disruptions")
async def get_disruptions():
    """Get current section disruptions"""
    update_train_state()
    return {
        "active_disruptions": train_state.section_disruptions,
        "affected_sections": len(train_state.section_disruptions),
        "timestamp": datetime.now().isoformat()
    }

@app.post("/reset")
async def reset_simulation():
    """Reset the simulation"""
    global train_state
    train_state = TrainState()
    return {"message": "Enhanced simulation reset", "timestamp": datetime.now().isoformat()}

latest_optimization_result = {}

@app.get("/api/optimization/results")
def get_optimization_results():
    """Get the latest optimization results"""
    if not latest_optimization_result:
        return {"message": "No optimization results available yet", "data": {}}
    
    return {
        "message": "Latest optimization results",
        "timestamp": latest_optimization_result.get("timestamp", ""),
        "data": latest_optimization_result
    }

@app.post("/api/optimization/results")
def update_optimization_results(data: dict):
    """Update optimization results from external optimizer"""
    global latest_optimization_result
    latest_optimization_result = {
        **data,
        "timestamp": datetime.now().isoformat(),
        "received_at": datetime.now().isoformat()
    }
    return {"message": "Optimization results updated successfully"}


@app.get("/api/train-data/summary")
async def get_train_summary():
    """Get enhanced summary of train operations"""
    update_train_state()
    
    status_counts = {}
    train_types = {}
    direction_counts = {"forward": 0, "backward": 0}
    destination_counts = {}
    
    for train in train_state.trains.values():
        # Count by status
        status = train["status"]
        status_counts[status] = status_counts.get(status, 0) + 1
        
        # Count by type
        train_type = train["type"]
        train_types[train_type] = train_types.get(train_type, 0) + 1
        
        # Count by direction
        direction = train.get("direction", "forward")
        direction_counts[direction] = direction_counts.get(direction, 0) + 1
        
        # Count by destination
        destination = train.get("destination_station", "Unknown")
        destination_counts[destination] = destination_counts.get(destination, 0) + 1
    
    return {
        "total_trains": len(train_state.trains),
        "status_breakdown": status_counts,
        "type_breakdown": train_types,
        "direction_breakdown": direction_counts,
        "destination_breakdown": destination_counts,
        "active_disruptions": len(train_state.section_disruptions),
        "section_occupancy": {k: len(v) for k, v in train_state.occupied_sections.items()},
        "timestamp": datetime.now().isoformat(),
        "system_uptime_minutes": (datetime.now() - train_state.start_time).total_seconds() / 60
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)