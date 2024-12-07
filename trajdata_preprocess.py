import numpy as np
import pandas as pd
import os
import torch
import pickle
from pathlib import Path
from typing import Dict, Any, List
from tqdm import tqdm
from typing import Optional

# Add these constants at the top of the file after imports
_agent_types = ["vehicle", "pedestrian", "cyclist", "background"]
_polygon_types = ["VEHICLE", "BIKE", "BUS", "PEDESTRIAN"]
_polygon_light_type = [
    "LANE_STATE_STOP",
    "LANE_STATE_GO",
    "LANE_STATE_CAUTION",
    "LANE_STATE_UNKNOWN",
]
_point_types = [
    "DASH_SOLID_YELLOW",
    "DASH_SOLID_WHITE",
    "DASHED_WHITE",
    "DASHED_YELLOW",
    "DOUBLE_SOLID_YELLOW",
    "DOUBLE_SOLID_WHITE",
    "DOUBLE_DASH_YELLOW",
    "DOUBLE_DASH_WHITE",
    "SOLID_YELLOW",
    "SOLID_WHITE",
    "SOLID_DASH_WHITE",
    "SOLID_DASH_YELLOW",
    "EDGE",
    "NONE",
    "UNKNOWN",
    "CROSSWALK",
    "CENTERLINE",
]
_point_sides = ["LEFT", "RIGHT", "CENTER"]


# Add helper function
def safe_list_index(ls: List[Any], elem: Any) -> Optional[int]:
    """Safely get index of element in list, return None if not found"""
    try:
        return ls.index(elem)
    except ValueError:
        return None


def load_trajdata_cache(
    cache_path: Path, scene_name: str, dt: float = 0.1
) -> Dict[str, Any]:
    """Load cached data from trajdata format"""
    scene_cache_dir = cache_path / scene_name

    # Load agent data
    agent_data = pd.read_feather(scene_cache_dir / f"agent_data_dt{dt:.2f}.feather")
    
    # Load scene index to get location info
    with open(scene_cache_dir / f"scene_index_dt{dt:.2f}.pkl", "rb") as f:
        scene_index = pickle.load(f)
    
    # Get map name from scene index
    map_name = scene_index.get("map_name", None)
    if map_name is None:
        print(f"Warning: No map name found for scene {scene_name}")
        return {
            "agent_data": agent_data,
            "map_name": None,
            "traffic_light_data": pd.DataFrame(),
            "map_data": None
        }

    # Load traffic light data
    try:
        traffic_light_data = pd.read_feather(
            scene_cache_dir / f"tls_data_dt{dt:.2f}.feather"
        )
    except FileNotFoundError:
        traffic_light_data = pd.DataFrame()

    # Load map data for the corresponding city
    map_path = cache_path / "maps" / f"{map_name}.pb"
    if not map_path.exists():
        print(f"Warning: Map file not found: {map_path}")
        return {
            "agent_data": agent_data,
            "map_name": map_name,
            "traffic_light_data": traffic_light_data,
            "map_data": None
        }

    with open(map_path, "rb") as f:
        map_data = f.read()

    return {
        "agent_data": agent_data,
        "map_name": map_name,
        "traffic_light_data": traffic_light_data,
        "map_data": map_data,
    }


def process_agent_data(agent_data: pd.DataFrame, av_id: str) -> Dict[str, Any]:
    """Process agent data using existing get_agent_features function"""
    return get_agent_features(
        df=agent_data, av_id=av_id, num_historical_steps=10, dim=3, num_steps=91
    )


def process_map_data(
    map_data: bytes,
    traffic_light_data: pd.DataFrame,
    center_pos: np.ndarray,
    patch_size: float = 150.0,  # 局部地图范围大小(米)
    resolution: float = 0.5,    # 栅格化分辨率(米/像素)
) -> Dict[str, Any]:
    """Process map data with local region cropping and rasterization
    
    Args:
        map_data: Raw protobuf map data
        traffic_light_data: Traffic light states
        center_pos: Center position [x, y] in world coordinates
        patch_size: Size of local map patch in meters
        resolution: Rasterization resolution in meters/pixel
    """
    try:
        from trajdata.proto.vectorized_map_pb2 import VectorizedMap
        from trajdata.maps.vec_map import VectorMap
        
        # Parse protobuf map
        vector_map = VectorizedMap()
        vector_map.ParseFromString(map_data)
        
        # Convert to VectorMap object
        vec_map = VectorMap.from_proto(vector_map)
        
        # Get rasterized local map patch
        map_img, raster_from_world = vec_map.rasterize(
            resolution=resolution,
            return_tf_mat=True,
            incl_centerlines=True,
            area_color=(255, 255, 255),
            edge_color=(0, 0, 0),
            center_color=(128, 128, 128)
        )
        
        # Get map elements within patch range
        nearby_lanes = vec_map.get_lanes_within(
            np.array([center_pos[0], center_pos[1], 0]), 
            dist=patch_size/2
        )
        
        nearby_areas = vec_map.get_areas_within(
            np.array([center_pos[0], center_pos[1], 0]),
            dist=patch_size/2
        )
        
        # Process map features
        map_features = {
            "raster": map_img,                    # 栅格化地图
            "raster_from_world": raster_from_world,# 世界坐标到栅格坐标转换矩阵
            "lanes": [],                          # 车道信息
            "crosswalks": [],                     # 人行横道信息
            "traffic_lights": []                  # 交通灯信息
        }
        
        # Process lanes
        for lane in nearby_lanes:
            lane_feature = {
                "id": lane.id,
                "centerline": lane.center.points,
                "left_boundary": lane.left_edge.points if lane.left_edge else None,
                "right_boundary": lane.right_edge.points if lane.right_edge else None,
            }
            map_features["lanes"].append(lane_feature)
            
        # Process areas (crosswalks etc)
        for area in nearby_areas:
            if area.elem_type == "PED_CROSSWALK":
                crosswalk = {
                    "polygon": area.polygon.xy,
                }
                map_features["crosswalks"].append(crosswalk)
                
        # Add traffic light states if available
        if not traffic_light_data.empty:
            for _, tl in traffic_light_data.iterrows():
                traffic_light = {
                    "id": tl["id"],
                    "state": tl["state"],
                    "position": np.array([tl["x"], tl["y"], tl["z"]])
                }
                map_features["traffic_lights"].append(traffic_light)
                
        return map_features
        
    except Exception as e:
        print(f"Error processing map data: {e}")
        return {}


def get_agent_features(
    df: pd.DataFrame, av_id, num_historical_steps=10, dim=3, num_steps=91
) -> Dict[str, Any]:
    # Filtering agents based on history steps
    historical_df = df[df["timestep"] == num_historical_steps - 1]
    agent_ids = list(historical_df["track_id"].unique())
    df = df[df["track_id"].isin(agent_ids)]

    num_agents = len(agent_ids)
    # Initialization
    valid_mask = torch.zeros(num_agents, num_steps, dtype=torch.bool)
    current_valid_mask = torch.zeros(num_agents, dtype=torch.bool)
    predict_mask = torch.zeros(num_agents, num_steps, dtype=torch.bool)
    agent_id: List[Optional[str]] = [None] * num_agents
    agent_type = torch.zeros(num_agents, dtype=torch.uint8)
    agent_category = torch.zeros(num_agents, dtype=torch.uint8)
    position = torch.zeros(num_agents, num_steps, dim, dtype=torch.float)
    heading = torch.zeros(num_agents, num_steps, dtype=torch.float)
    velocity = torch.zeros(num_agents, num_steps, dim, dtype=torch.float)
    shape = torch.zeros(num_agents, num_steps, dim, dtype=torch.float)

    for track_id, track_df in df.groupby("track_id"):
        agent_idx = agent_ids.index(track_id)
        agent_steps = track_df["timestep"].values

        valid_mask[agent_idx, agent_steps] = True
        current_valid_mask[agent_idx] = valid_mask[agent_idx, num_historical_steps - 1]
        predict_mask[agent_idx, agent_steps] = True
        valid_mask[agent_idx, 1:num_historical_steps] = (
            valid_mask[agent_idx, : num_historical_steps - 1]
            & valid_mask[agent_idx, 1:num_historical_steps]
        )
        valid_mask[agent_idx, 0] = False
        predict_mask[agent_idx, :num_historical_steps] = False
        if not current_valid_mask[agent_idx]:
            predict_mask[agent_idx, num_historical_steps:] = False

        agent_id[agent_idx] = track_id
        agent_type[agent_idx] = _agent_types.index(track_df["object_type"].values[0])
        agent_category[agent_idx] = track_df["object_category"].values[0]
        position[agent_idx, agent_steps, :3] = torch.from_numpy(
            np.stack(
                [
                    track_df["position_x"].values,
                    track_df["position_y"].values,
                    track_df["position_z"].values,
                ],
                axis=-1,
            )
        ).float()
        heading[agent_idx, agent_steps] = torch.from_numpy(
            track_df["heading"].values
        ).float()
        velocity[agent_idx, agent_steps, :2] = torch.from_numpy(
            np.stack(
                [track_df["velocity_x"].values, track_df["velocity_y"].values], axis=-1
            )
        ).float()
        shape[agent_idx, agent_steps, :3] = torch.from_numpy(
            np.stack(
                [
                    track_df["length"].values,
                    track_df["width"].values,
                    track_df["height"].values,
                ],
                axis=-1,
            )
        ).float()
    av_idx = agent_id.index(av_id)

    return {
        "num_nodes": num_agents,
        "av_index": av_idx,
        "valid_mask": valid_mask,
        "predict_mask": predict_mask,
        "id": agent_id,
        "type": agent_type,
        "category": agent_category,
        "position": position,
        "heading": heading,
        "velocity": velocity,
        "shape": shape,
    }


def convert_trajdata_to_final(cache_path: Path, output_dir: Path):
    """Batch convert cached trajdata to final format"""
    os.makedirs(output_dir, exist_ok=True)

    # Get all scene directories
    scene_dirs = [d for d in cache_path.iterdir() if d.is_dir() and d.name != "maps"]
    
    # Load maps once and cache them
    map_cache = {}
    maps_dir = cache_path / "maps"
    for map_file in maps_dir.glob("*.pb"):
        with open(map_file, "rb") as f:
            map_cache[map_file.stem] = f.read()

    for scene_dir in tqdm(scene_dirs):
        # Load cached data
        cache_data = load_trajdata_cache(cache_path, scene_dir.name)

        # Get ego vehicle position
        agent_data = cache_data["agent_data"]
        ego_df = agent_data[agent_data["is_sdc"]]
        center_pos = np.array([
            ego_df.iloc[0]["position_x"],
            ego_df.iloc[0]["position_y"]
        ])
        
        # Process map data with local region
        map_features = {}
        if cache_data["map_data"] is not None:
            map_features = process_map_data(
                cache_data["map_data"],
                cache_data["traffic_light_data"],
                center_pos
            )

        # Process agent data
        agent_features = process_agent_data(agent_data, ego_df.iloc[0]["track_id"])

        # Construct final data format
        final_data = {
            "scenario_id": scene_dir.name,
            "city": cache_data["map_name"],  # Now we have the city/map name
            "agent": agent_features,
            "map_data": map_features,
        }

        # Save processed data
        with open(output_dir / f"{scene_dir.name}.pkl", "wb") as f:
            pickle.dump(final_data, f)


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="/home/haoweis/.unified_data_cache/nuplan_mini/",
    )
    parser.add_argument("--output_dir", type=str, default="data/trajdata/processed")
    args = parser.parse_args()

    convert_trajdata_to_final(
        cache_path=Path(args.cache_dir), output_dir=Path(args.output_dir)
    )
