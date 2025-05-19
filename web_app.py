import streamlit as st
import osmnx as ox
import networkx as nx
import random
from pyproj import Transformer
import srtm
from shapely.geometry import LineString, box, MultiLineString, Point
from shapely.strtree import STRtree
from osmnx.features import features_from_polygon, features_from_place
import matplotlib.pyplot as plt
import pandas as pd
from pyproj import CRS
import math
from geopandas import GeoDataFrame
from shapely.ops import transform
import pyproj
import folium
from streamlit_folium import folium_static, st_folium
import numpy as np
from collections import defaultdict, Counter
import contextily as cx
import geopandas as gpd
import traceback
import time

# Import the POI categories
from poi_categories import POI_CATEGORIES

# Road type factors (base multipliers)
ROAD_TYPE_FACTORS = {
    "motorway": 5.0,
    "trunk": 4.0,
    "primary": 3.0,
    "secondary": 1.5,
    "tertiary": 1.2,
    "unclassified": 1.0,
    "residential": 1.0,
    "living_street": 0.9,
    "service": 0.8,
    "track": 1.1,
    "path": 0.7,
    "footway": 0.6,
    "cycleway": 0.5,
}


# Helper function for product of values
def product(values):
    """Calculate product of a list of values"""
    result = 1.0
    for val in values:
        result *= val
    return result


@st.cache_resource
def load_graph(place_name, network_type="walk"):
    """Loads, projects, and annotates the graph."""
    try:
        st.write(f"Loading graph for '{place_name}'...")
        G_ll = ox.graph_from_place(place_name, network_type=network_type)

        # Store the original unprojected graph for coordinate reference
        st.session_state["unprojected_graph"] = G_ll

        st.write("Graph loaded. Adding elevation data...")
        elev = srtm.get_data()
        for nid, data in G_ll.nodes(data=True):
            data["elevation"] = elev.get_elevation(data["y"], data["x"]) or 0.0

        st.write("Elevation added. Calculating edge grades...")
        for u, v, key, data in G_ll.edges(keys=True, data=True):
            z1, z2 = G_ll.nodes[u]["elevation"], G_ll.nodes[v]["elevation"]
            data["elev_gain"] = max(0, z2 - z1)
            # Ensure highway tag is a list
            hw = data.get("highway", "unclassified")
            data["highway"] = hw if isinstance(hw, list) else [hw]
            # Calculate grade
            length = data.get("length", 0)
            if length > 0:
                data["grade"] = (z2 - z1) / length
            else:
                data["grade"] = 0.0

        st.write("Projecting graph...")
        G_proj = ox.project_graph(G_ll)
        st.write("Graph processing complete.")
        return G_proj
    except Exception as e:
        st.error(f"Error loading graph for '{place_name}': {e}")
        st.error(traceback.format_exc())
        return None


def load_pois_bbox(G, place_name, selected_tags_dict):
    """
    Pull POIs inside the graph's bounding box.
    selected_tags_dict is like {'amenity': ['restaurant', 'cafe'], 'shop': ['supermarket']}
    """
    if not selected_tags_dict:
        return None

    # Flatten the tags for OSMnx query
    tags = {}
    for category, subtypes in selected_tags_dict.items():
        tags[category] = subtypes

    try:
        # Get graph bounds
        nodes_gdf = ox.graph_to_gdfs(G, nodes=True, edges=False)
        minx, miny, maxx, maxy = nodes_gdf.total_bounds
        bbox_poly = box(minx, miny, maxx, maxy)

        # Use the graph's CRS explicitly
        graph_crs = G.graph["crs"]

        with st.spinner("Fetching POIs..."):

            try:
                # First try with polygon in graph CRS
                pois = features_from_polygon(bbox_poly, tags=tags)
                if pois is not None and not pois.empty:
                    # Project to graph CRS
                    pois = pois.to_crs(graph_crs)
            except Exception as polygon_error:
                st.warning(f"Error with polygon approach: {polygon_error}")
                # Fallback to place name approach
                try:
                    pois = features_from_place(place_name, tags=tags)
                    if pois is not None and not pois.empty:
                        # Project to graph CRS
                        pois = pois.to_crs(graph_crs)
                        # Filter to bbox
                        bbox_poly_wgs84 = (
                            gpd.GeoDataFrame(geometry=[bbox_poly], crs=graph_crs)
                            .to_crs(4326)
                            .geometry[0]
                        )
                        pois = pois[pois.geometry.intersects(bbox_poly_wgs84)]
                except Exception as place_error:
                    st.error(f"Error with place approach: {place_error}")
                    return None

            # Ensure the geometry column exists and filter out invalid geometries
            if pois is None or "geometry" not in pois.columns:
                st.warning("No POIs found or no geometry column.")
                return None

            pois = pois[pois.geometry.notna()]
            if pois.empty:
                st.warning("No POIs found within the graph area for selected tags.")
                return None

            return pois

    except Exception as e:
        st.error(f"Error fetching POIs: {e}")
        st.error(traceback.format_exc())
        return None


def route_point_to_point(
    G,
    origin_coords,  # This should be (lat, lon)
    dest_coords,  # This should be (lat, lon)
    length_w=1.0,
    elev_w=1.0,
    factors=None,
    poi_tree=None,
    poi_buffer=50,
    poi_reward=0.5,
):
    """Generate a route from origin to destination."""
    st.sidebar.write("--- Inside route_point_to_point ---")  # DEBUG
    try:
        # Get graph's CRS and define WGS84 CRS
        graph_crs = G.graph.get("crs")
        if not graph_crs:
            raise ValueError("Graph CRS is not defined.")
        wgs84_crs = CRS("EPSG:4326")
        st.sidebar.write(f"Graph CRS: {graph_crs}")  # DEBUG

        # Create a transformer (ensure lon, lat order for input WGS84)
        transformer = Transformer.from_crs(wgs84_crs, graph_crs, always_xy=True)

        # Project origin coordinates (lon, lat) to graph CRS
        origin_lon, origin_lat = origin_coords[1], origin_coords[0]
        projected_origin_x, projected_origin_y = transformer.transform(
            origin_lon, origin_lat
        )
        st.sidebar.write(
            f"Origin (lat, lon): ({origin_lat}, {origin_lon}) -> Projected (X, Y): ({projected_origin_x:.2f}, {projected_origin_y:.2f})"
        )  # DEBUG

        # Project destination coordinates (lon, lat) to graph CRS
        dest_lon, dest_lat = dest_coords[1], dest_coords[0]
        projected_dest_x, projected_dest_y = transformer.transform(dest_lon, dest_lat)
        st.sidebar.write(
            f"Destination (lat, lon): ({dest_lat}, {dest_lon}) -> Projected (X, Y): ({projected_dest_x:.2f}, {projected_dest_y:.2f})"
        )  # DEBUG

        # Find nearest nodes using projected coordinates
        origin_node = ox.distance.nearest_nodes(
            G, projected_origin_x, projected_origin_y
        )
        dest_node = ox.distance.nearest_nodes(G, projected_dest_x, projected_dest_y)

        st.sidebar.write(f"Found origin node: {origin_node}")
        st.sidebar.write(f"Found destination node: {dest_node}")

        # Check if nodes are the same
        if origin_node == dest_node:
            st.warning("Origin and destination points map to the same network node.")
            # Return a path with just the single node to avoid shortest_path errors
            # Or potentially raise a specific error/warning here
            return [origin_node]  # Or return None / raise error

        # Define the weight function locally for point-to-point
        def p2p_weight_function(u, v, d):
            edge_data = d[0] if isinstance(d, (tuple, list)) else d
            length = max(float(edge_data.get("length", 1.0)), 0.1)
            factor = 1.0
            if factors:
                factor = product(
                    [value for key, value in factors.items() if key in edge_data]
                )  # Simplified factor check

            elev_penalty = 0
            if "grade" in edge_data and elev_w > 0:
                elev_penalty = abs(float(edge_data.get("grade", 0))) * elev_w * length

            base_cost = length * length_w * factor + elev_penalty

            poi_reward_value = 0
            if poi_tree is not None and u in G.nodes and v in G.nodes:
                # Check if nodes exist before accessing coordinates
                if G.nodes[u] and G.nodes[v]:
                    mid_point = Point(
                        (G.nodes[u]["x"] + G.nodes[v]["x"]) / 2,
                        (G.nodes[u]["y"] + G.nodes[v]["y"]) / 2,
                    )
                    # Query returns indices, check if list is non-empty
                    # Use STRtree query correctly - check if any geometry intersects
                    intersecting_indices = poi_tree.query(mid_point.buffer(poi_buffer))
                    if (
                        len(intersecting_indices) > 0
                    ):  # Check if the list of indices is non-empty
                        poi_reward_value = -length * 0.1 * poi_reward  # Apply reward

            return max(0.1, base_cost + poi_reward_value)

        st.sidebar.write("Calculating shortest path...")  # DEBUG
        path = nx.shortest_path(G, origin_node, dest_node, weight=p2p_weight_function)
        st.sidebar.write(f"Shortest path found with {len(path)} nodes.")  # DEBUG
        return path
    except nx.NetworkXNoPath:
        st.error("No path found between the specified points on the network.")
        raise ValueError("No path found between the specified points.")
    except Exception as e:
        st.error(f"Error finding point-to-point path: {e}")
        st.error(traceback.format_exc())  # Log detailed error
        raise ValueError(f"Error finding point-to-point path: {e}")


def get_midpoint_candidates(G, origin_node, target_m):
    """Get candidate nodes for midpoints at approximately the right distance."""
    # Target distance for the first half (slightly more than half)
    midpoint_target_dist = target_m * 0.60
    min_dist = midpoint_target_dist * 0.7  # Allow shorter first legs
    max_dist = midpoint_target_dist * 1.3  # Allow longer first legs

    candidates = []
    try:
        # Use Dijkstra to find nodes within a distance range
        lengths = nx.single_source_dijkstra_path_length(
            G, origin_node, cutoff=max_dist, weight="length"
        )
        candidates = [
            node
            for node, dist in lengths.items()
            if dist >= min_dist and node != origin_node
        ]

        # If not enough candidates, try a wider search (less efficient)
        if len(candidates) < 30:  # Need a decent pool
            st.sidebar.write("Widening midpoint search...")
            all_nodes = list(G.nodes())
            random.shuffle(all_nodes)
            count = 0
            for node in all_nodes:
                if node != origin_node and node not in candidates:
                    try:
                        # Check distance using a simple path length
                        dist = nx.shortest_path_length(
                            G, origin_node, node, weight="length"
                        )
                        if min_dist * 0.8 <= dist <= max_dist * 1.2:  # Wider tolerance
                            candidates.append(node)
                            count += 1
                            if count >= 50:  # Limit additional candidates
                                break
                    except (nx.NetworkXNoPath, KeyError):
                        continue  # Skip if no path or node error

    except Exception as e:
        st.sidebar.warning(f"Midpoint candidate search error: {e}. Using random nodes.")
        # Fallback: sample random nodes (less ideal)
        candidates = random.sample(
            list(G.nodes - {origin_node}), min(100, len(G.nodes) - 1)
        )

    # Return a random sample of the candidates found
    num_to_select = min(60, len(candidates))  # Try up to 60 candidates
    return random.sample(candidates, num_to_select) if candidates else []


# --- Main Circular Route Function ---


def route_circular(
    G,
    origin_coords,  # This should be (lat, lon)
    target_km,
    length_w=1.0,
    elev_w=1.0,
    factors=None,
    poi_tree=None,
    poi_buffer=50,
    poi_reward=0.5,
    tol_frac=0.1,
    seed=None,
    reuse_penalty=2.0,
    buffer_m=20,
    sector_guidance=True,
    sector_weight=0.2,
):
    """Generate a circular route starting and ending at the same point."""
    st.sidebar.write("--- Inside route_circular ---")  # DEBUG
    if seed is not None:
        np.random.seed(seed)

    # Get the unprojected graph for accurate coordinate comparison if needed later
    G_unprojected = st.session_state.get("unprojected_graph", None)

    try:
        # Get graph's CRS and define WGS84 CRS
        graph_crs = G.graph.get("crs")
        if not graph_crs:
            raise ValueError("Graph CRS is not defined.")
        wgs84_crs = CRS("EPSG:4326")
        st.sidebar.write(f"Graph CRS: {graph_crs}")  # DEBUG

        # Create a transformer (ensure lon, lat order for input WGS84)
        transformer = Transformer.from_crs(wgs84_crs, graph_crs, always_xy=True)

        # Project origin coordinates (lon, lat) to graph CRS
        origin_lon, origin_lat = origin_coords[1], origin_coords[0]
        projected_origin_x, projected_origin_y = transformer.transform(
            origin_lon, origin_lat
        )
        st.sidebar.write(
            f"Origin (lat, lon): ({origin_lat}, {origin_lon}) -> Projected (X, Y): ({projected_origin_x:.2f}, {projected_origin_y:.2f})"
        )  # DEBUG

        # Find nearest node using *projected* coordinates
        origin_node = ox.distance.nearest_nodes(
            G, projected_origin_x, projected_origin_y
        )
        st.sidebar.write(f"Found origin node: {origin_node}")

    except Exception as e:
        st.error(f"Error finding origin node: {e}")
        st.error(traceback.format_exc())
        raise ValueError(f"Could not find starting node on graph: {e}")

    # Verify the node location by getting its coordinates (optional debug/distance check)
    # Use the unprojected graph for this check if available
    display_node_graph = G_unprojected if G_unprojected else G
    if origin_node in display_node_graph.nodes:
        # Get coordinates from the appropriate graph (prefer unprojected for lat/lon)
        node_y = display_node_graph.nodes[origin_node]["y"]
        node_x = display_node_graph.nodes[origin_node]["x"]
        st.sidebar.write(
            f"Node coordinates ({'unprojected' if G_unprojected else 'projected'}): ({node_y}, {node_x})"
        )

        # If using unprojected graph, calculate Haversine distance
        if G_unprojected:

            def haversine_distance(lat1, lon1, lat2, lon2):
                lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
                dlon = lon2 - lon1
                dlat = lat2 - lat1
                a = (
                    math.sin(dlat / 2) ** 2
                    + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
                )
                c = 2 * math.asin(math.sqrt(a))
                r = 6371000
                return c * r

            dist_to_node = haversine_distance(
                origin_coords[0], origin_coords[1], node_y, node_x
            )
            st.sidebar.write(
                f"Distance from clicked point to nearest node: {dist_to_node:.1f} meters"
            )
            if dist_to_node > 500:
                st.warning(f"⚠️ Nearest node is {dist_to_node:.1f}m away from click.")

    target_m = target_km * 1000
    min_m = target_m * (1 - tol_frac)
    max_m = target_m * (1 + tol_frac)

    # --- Midpoint Candidate Search ---
    st.sidebar.write("Getting midpoint candidates...")  # DEBUG
    midpoint_candidates = get_midpoint_candidates(G, origin_node, target_m)
    if not midpoint_candidates:
        st.error("Could not find suitable midpoint candidates for the circular route.")
        raise ValueError("Failed to find midpoint candidates.")
    st.sidebar.write(f"Found {len(midpoint_candidates)} midpoint candidates.")  # DEBUG

    # --- Define Weight Functions ---
    def weight_fwd(u, v, d):
        edge_data = d[0] if isinstance(d, (tuple, list)) else d
        length = max(float(edge_data.get("length", 1.0)), 0.1)
        factor = 1.0
        if factors:
            factor = product(
                [value for key, value in factors.items() if key in edge_data]
            )

        elev_penalty = 0
        if "grade" in edge_data and elev_w > 0:
            elev_penalty = abs(float(edge_data.get("grade", 0))) * elev_w * length

        base_cost = length * length_w * factor + elev_penalty

        poi_reward_value = 0
        if poi_tree is not None and u in G.nodes and v in G.nodes:
            if G.nodes[u] and G.nodes[v]:
                mid_point = Point(
                    (G.nodes[u]["x"] + G.nodes[v]["x"]) / 2,
                    (G.nodes[u]["y"] + G.nodes[v]["y"]) / 2,
                )
                intersecting_indices = poi_tree.query(mid_point.buffer(poi_buffer))
                if len(intersecting_indices) > 0:
                    poi_reward_value = -length * 0.1 * poi_reward

        # Sector Guidance Penalty (applied on forward path)
        sector_penalty = 0
        if sector_guidance and u != origin_node:  # Don't penalize first edge
            # Get coordinates (use projected graph G for calculations)
            u_coords = (G.nodes[u]["x"], G.nodes[u]["y"])
            v_coords = (G.nodes[v]["x"], G.nodes[v]["y"])
            origin_coords_proj = (
                projected_origin_x,
                projected_origin_y,
            )  # Use projected origin

            # Calculate vectors
            vec_origin_u = (
                u_coords[0] - origin_coords_proj[0],
                u_coords[1] - origin_coords_proj[1],
            )
            vec_u_v = (v_coords[0] - u_coords[0], v_coords[1] - u_coords[1])

            # Calculate angle (dot product) - handle potential zero vectors
            len_ou = math.sqrt(vec_origin_u[0] ** 2 + vec_origin_u[1] ** 2)
            len_uv = math.sqrt(vec_u_v[0] ** 2 + vec_u_v[1] ** 2)

            if len_ou > 1e-6 and len_uv > 1e-6:  # Avoid division by zero
                dot_product = (
                    vec_origin_u[0] * vec_u_v[0] + vec_origin_u[1] * vec_u_v[1]
                )
                cos_theta = dot_product / (len_ou * len_uv)
                cos_theta = max(-1.0, min(1.0, cos_theta))  # Clamp to valid range
                angle_rad = math.acos(cos_theta)
                # Penalize turning back towards origin (angle close to pi or 180 deg)
                # Penalty increases as angle approaches pi
                sector_penalty = (
                    (angle_rad / math.pi) * length * sector_weight * -1
                )  # Reward moving away

        return max(0.1, base_cost + poi_reward_value + sector_penalty)

    def weight_bwd(u, v, d, path1_nodes):
        edge_data = d[0] if isinstance(d, (tuple, list)) else d
        length = max(float(edge_data.get("length", 1.0)), 0.1)
        factor = 1.0
        if factors:
            factor = product(
                [value for key, value in factors.items() if key in edge_data]
            )

        elev_penalty = 0
        if "grade" in edge_data and elev_w > 0:
            # Reverse grade for backward path
            grade = float(edge_data.get("grade", 0))
            elev_penalty = abs(-grade) * elev_w * length  # Use negative grade

        base_cost = length * length_w * factor + elev_penalty

        poi_reward_value = 0
        if poi_tree is not None and u in G.nodes and v in G.nodes:
            if G.nodes[u] and G.nodes[v]:
                mid_point = Point(
                    (G.nodes[u]["x"] + G.nodes[v]["x"]) / 2,
                    (G.nodes[u]["y"] + G.nodes[v]["y"]) / 2,
                )
                intersecting_indices = poi_tree.query(mid_point.buffer(poi_buffer))
                if len(intersecting_indices) > 0:
                    poi_reward_value = -length * 0.1 * poi_reward

        # Penalty for reusing nodes/edges from the first path
        reuse_cost = 0
        if v in path1_nodes:  # Penalize entering a node already used in path1
            reuse_cost = length * reuse_penalty

        return max(0.1, base_cost + poi_reward_value + reuse_cost)

    # --- Find Best Loop ---
    best_path = None
    best_len = float("inf")
    attempts = 0
    max_attempts = len(midpoint_candidates)  # Try all candidates

    st.sidebar.write(
        f"Attempting to find loop via {max_attempts} candidates..."
    )  # DEBUG

    for midpoint_node in midpoint_candidates:
        attempts += 1
        if attempts % 10 == 0:
            st.sidebar.write(f"Attempt {attempts}/{max_attempts}...")  # DEBUG

        try:
            # Path 1: Origin to Midpoint
            path1 = nx.shortest_path(
                G, source=origin_node, target=midpoint_node, weight=weight_fwd
            )
            len1 = sum(
                G.edges[u, v, 0]["length"] for u, v in zip(path1[:-1], path1[1:])
            )

            # Path 2: Midpoint to Origin (avoiding path1 nodes)
            path1_nodes_set = set(path1)  # Use set for faster lookup
            path2 = nx.shortest_path(
                G,
                source=midpoint_node,
                target=origin_node,
                weight=lambda u, v, d: weight_bwd(u, v, d, path1_nodes_set),
            )
            len2 = sum(
                G.edges[u, v, 0]["length"] for u, v in zip(path2[:-1], path2[1:])
            )

            total_len = len1 + len2

            # Check if length is within tolerance and better than current best
            if min_m <= total_len <= max_m:
                # Combine paths (avoid duplicating midpoint)
                combined_path = path1[:-1] + path2
                # Basic check for self-intersection (can be improved)
                if (
                    len(combined_path) > len(set(combined_path)) + 1
                ):  # Allow start/end overlap
                    st.sidebar.write(
                        f"Candidate {midpoint_node}: Path intersects too much, skipping."
                    )  # DEBUG
                    continue  # Skip paths that intersect too much

                # Check if this path is closer to the target length than the current best
                if abs(total_len - target_m) < abs(best_len - target_m):
                    best_len = total_len
                    best_path = combined_path
                    st.sidebar.write(
                        f"Found new best path via {midpoint_node}, length {best_len/1000:.2f} km."
                    )  # DEBUG
                    # Break early if a very good match is found
                    # if abs(best_len - target_m) < target_m * 0.01: # Within 1%
                    #    st.sidebar.write("Found excellent match, stopping search.")
                    #    break

        except nx.NetworkXNoPath:
            # st.sidebar.write(f"No path found via midpoint {midpoint_node}") # Can be noisy
            continue  # Try next candidate
        except Exception as loop_error:
            st.sidebar.warning(
                f"Error processing midpoint {midpoint_node}: {loop_error}"
            )
            continue  # Try next candidate

    if not best_path:
        st.error(
            f"Could not find a circular route within {tol_frac*100:.0f}% of {target_km} km after {attempts} attempts."
        )
        raise ValueError("Failed to find suitable circular route.")

    st.sidebar.write(f"Selected best path length: {best_len/1000:.2f} km")  # DEBUG
    return best_path


def _get_poi_type(poi_row):
    """Extract the primary type of a POI from its attributes."""
    # Prioritize common tags
    for key in ["amenity", "shop", "tourism", "leisure", "historic", "natural"]:
        if key in poi_row and pd.notna(poi_row[key]):
            # Return first non-null value found for these keys
            val = poi_row[key]
            # Handle cases where the value might be a list (though less common for these tags)
            if isinstance(val, list):
                return f"{key}: {val[0]}"  # Take the first item if it's a list
            return f"{key}: {val}"
    # Fallback if no common tags found
    for key, val in poi_row.items():
        # Check if it's a likely tag (not geometry, osmid, etc.) and has a value
        if key not in ["geometry", "osmid", "nodes", "element_type"] and pd.notna(val):
            if isinstance(val, list):
                return f"{key}: {val[0]}"
            return f"{key}: {val}"  # Return the first other tag found
    return "unknown"


# — Streamlit UI —
st.set_page_config(layout="wide")
st.title("🗺️ Route Planner")

# --- Initialize Session State ---
if "generated_map" not in st.session_state:
    st.session_state.generated_map = None
if "origin" not in st.session_state:
    st.session_state.origin = None
if "dest" not in st.session_state:
    st.session_state.dest = None
if "_last_click_processed" not in st.session_state:
    st.session_state._last_click_processed = None

# --- Sidebar Inputs ---
st.sidebar.header("Route Configuration")
place = st.sidebar.text_input("Enter Place Name", "Oxford, UK")
route_type = st.sidebar.radio("Route Type", ["Point-to-point", "Circular"])

st.sidebar.subheader("Weighting Factors")
length_w = st.sidebar.slider(
    "Distance Weight",
    0.1,
    5.0,
    1.0,
    0.1,
    help="Higher values prioritize shorter routes.",
)
elev_w = st.sidebar.slider(
    "Elevation Gain Weight",
    0.0,
    10.0,
    2.0,
    0.1,
    help="Higher values penalize uphill sections more.",
)

# --- POI Selection ---
st.sidebar.subheader("Points of Interest (POI)")
selected_poi_tags = {}
# Use columns for better layout if many categories
poi_cols = st.sidebar.columns(2)
col_idx = 0
# Dynamically create expanders for POI categories
with st.sidebar.expander("Select POI Categories & Types", expanded=False):
    for category, subtypes in POI_CATEGORIES.items():
        # Simple checkbox for the whole category first
        select_category = st.checkbox(f"{category.title()}", key=f"cat_{category}")
        if select_category:
            # If category selected, show multiselect for subtypes
            selected_subtypes = st.multiselect(
                f"Select {category} types",
                options=subtypes,
                default=[],
                key=f"sub_{category}",
            )
            if selected_subtypes:  # Only add if specific subtypes are chosen
                selected_poi_tags[category] = selected_subtypes
            else:  # If category checked but no subtypes, select all subtypes
                selected_poi_tags[category] = subtypes


poi_buffer = st.sidebar.slider(
    "POI Search Radius (m)",
    0,
    200,
    50,
    10,
    help="How close the route must pass to a POI to get a reward.",
)
poi_reward = st.sidebar.slider(
    "POI Reward Factor",
    0.1,
    1.0,
    0.5,
    0.1,
    help="How much to reward routes passing near POIs (lower value = stronger reward). Applied as cost multiplier.",
)


# --- Main Area Inputs ---
st.header("Define Start/End Points")

# Initialize session state for origin and destination if they don't exist
origin = st.session_state.origin
dest = st.session_state.dest

use_map_picker = st.checkbox("Use map to select points", value=True)

if use_map_picker:
    # Default center location
    try:
        geocode_result = ox.geocode(place if place else "Oxford, UK")
        center_lat, center_lon = geocode_result[0], geocode_result[1]
    except Exception:
        center_lat, center_lon = 51.7548, -1.2544  # Fallback

    # Initialize map
    m = folium.Map(location=[center_lat, center_lon], zoom_start=13)

    # Add markers based on route type and current selections
    click_instruction = ""
    if route_type == "Circular":
        click_instruction = "Click on the map to set the Origin (green)."
        if origin:
            folium.Marker(
                origin, popup="Origin", icon=folium.Icon(color="green")
            ).add_to(m)
    elif route_type == "Point-to-point":
        if not origin:
            click_instruction = "Click on the map to set the Origin (green)."
        elif not dest:
            click_instruction = "Click on the map to set the Destination (red)."
            folium.Marker(
                origin, popup="Origin", icon=folium.Icon(color="green")
            ).add_to(m)
        else:  # Both set
            click_instruction = "Origin and Destination set. Click to reset."
            folium.Marker(
                origin, popup="Origin", icon=folium.Icon(color="green")
            ).add_to(m)
            folium.Marker(
                dest, popup="Destination", icon=folium.Icon(color="red")
            ).add_to(m)

    st.write(click_instruction)
    map_data = st_folium(m, height=450, width=700)

    # Process map clicks
    if map_data["last_clicked"]:
        last_click_processed = st.session_state.get("_last_click_processed", None)
        current_click_coords = (
            map_data["last_clicked"]["lat"],
            map_data["last_clicked"]["lng"],
        )

        if current_click_coords != last_click_processed:
            st.session_state._last_click_processed = (
                current_click_coords  # Store processed click
            )

            if route_type == "Circular":
                st.session_state.origin = current_click_coords
                st.info(
                    f"Origin set: {current_click_coords[0]:.6f}, {current_click_coords[1]:.6f}"
                )
            elif route_type == "Point-to-point":
                if not st.session_state.origin:  # If origin not set, set it
                    st.session_state.origin = current_click_coords
                    st.info(
                        f"Origin set: {current_click_coords[0]:.6f}, {current_click_coords[1]:.6f}"
                    )
                elif (
                    not st.session_state.dest
                ):  # If origin set but dest not set, set dest
                    st.session_state.dest = current_click_coords
                    st.info(
                        f"Destination set: {current_click_coords[0]:.6f}, {current_click_coords[1]:.6f}"
                    )
                else:  # If both are set, clicking resets both
                    st.session_state.origin = current_click_coords  # Set new origin
                    st.session_state.dest = None  # Clear destination
                    st.info(
                        f"Origin reset: {current_click_coords[0]:.6f}, {current_click_coords[1]:.6f}. Click again for Destination."
                    )

            st.rerun()  # Rerun to update map markers and instructions

    # Display current selections and clear buttons
    st.sidebar.subheader("Selected Points")
    if route_type == "Circular":
        if origin:
            st.sidebar.success(f"Origin: {origin[0]:.5f}, {origin[1]:.5f}")
            if st.sidebar.button("Clear Origin"):
                st.session_state.origin = None
                st.session_state._last_click_processed = None
                st.rerun()
        else:
            st.sidebar.warning("Click map for Origin.")
    elif route_type == "Point-to-point":
        if origin:
            st.sidebar.success(f"Origin: {origin[0]:.5f}, {origin[1]:.5f}")
            if st.sidebar.button("Clear Origin"):
                st.session_state.origin = None
                # Also clear destination if origin is cleared
                st.session_state.dest = None
                st.session_state._last_click_processed = None
                st.rerun()
        else:
            st.sidebar.warning("Click map for Origin.")

        if dest:
            st.sidebar.success(f"Destination: {dest[0]:.5f}, {dest[1]:.5f}")
            if st.sidebar.button("Clear Destination"):
                st.session_state.dest = None
                st.session_state._last_click_processed = (
                    None  # Allow re-clicking same spot for origin if needed
                )
                st.rerun()
        elif origin:  # Only show if origin is set
            st.sidebar.warning("Click map for Destination.")

    # Get final origin/dest values from session state for routing
    origin = st.session_state.origin
    dest = st.session_state.dest

else:  # Use text input
    st.sidebar.subheader("Enter Coordinates")
    if route_type == "Point-to-point":
        origin_str = st.sidebar.text_input(
            "Origin (lat,lon)",
            (
                f"{st.session_state.origin[0]},{st.session_state.origin[1]}"
                if st.session_state.origin
                else "51.7548,-1.2544"
            ),
        )
        dest_str = st.sidebar.text_input(
            "Destination (lat,lon)",
            (
                f"{st.session_state.dest[0]},{st.session_state.dest[1]}"
                if st.session_state.dest
                else "51.7591,-1.2577"
            ),
        )
        try:
            origin = tuple(map(float, origin_str.split(",")))
            dest = tuple(map(float, dest_str.split(",")))
            if len(origin) != 2 or len(dest) != 2:
                raise ValueError
            st.session_state.origin = origin  # Update session state
            st.session_state.dest = dest
        except:
            st.sidebar.error("Invalid coordinate format. Use lat,lon")
            origin, dest = None, None
            st.session_state.origin, st.session_state.dest = None, None

    else:  # Circular route
        origin_str = st.sidebar.text_input(
            "Origin (lat,lon)",
            (
                f"{st.session_state.origin[0]},{st.session_state.origin[1]}"
                if st.session_state.origin
                else "51.7548,-1.2544"
            ),
        )
        try:
            origin = tuple(map(float, origin_str.split(",")))
            if len(origin) != 2:
                raise ValueError
            st.session_state.origin = origin  # Update session state
            st.session_state.dest = None  # Ensure dest is None for circular
            dest = None
        except:
            st.sidebar.error("Invalid coordinate format. Use lat,lon")
            origin = None
            st.session_state.origin = None

# --- Route Generation Trigger ---
st.header("Generate Route")

# Add specific parameters for circular route
target_km = None
tol_frac = None
reuse_penalty = None
sector_guidance = None
sector_weight = None

if route_type == "Circular":
    st.subheader("Circular Route Parameters")
    target_km = st.number_input(
        "Target Distance (km)", min_value=1.0, max_value=100.0, value=10.0, step=1.0
    )
    tol_frac = (
        st.slider("Distance Tolerance (%)", 0.01, 0.5, 0.1, 0.01, format="%.0f%%")
        / 100.0
    )
    reuse_penalty = st.slider(
        "Reuse Penalty",
        1.0,
        5.0,
        2.0,
        0.1,
        help="Higher values discourage reusing the same roads.",
    )
    sector_guidance = st.checkbox(
        "Use Sector Guidance",
        value=True,
        help="Encourage exploring different directions from the start.",
    )
    sector_weight = st.slider(
        "Sector Weight",
        0.0,
        1.0,
        0.2,
        0.05,
        help="How strongly to encourage new sectors (if enabled).",
    )


# --- Generate Route ---
if st.button("Generate Route"):
    # Clear any previously generated map from session state on new generation attempt
    st.session_state.generated_map = None
    # Prevent map click handler interference during the immediate rerun
    st.session_state._last_click_processed = f"button_click_{time.time()}"

    # Validation checks
    valid_inputs = True
    # Use origin/dest directly from session state for validation inside button
    origin_val = st.session_state.origin
    dest_val = st.session_state.dest
    if not place:
        st.error("Please enter a place name.")
        valid_inputs = False
    if route_type == "Circular" and not origin_val:
        st.error("Please select an origin point for the circular route.")
        valid_inputs = False
    if route_type == "Point-to-point" and (not origin_val or not dest_val):
        st.error(
            "Please select both an origin and destination point for the point-to-point route."
        )
        valid_inputs = False

    if valid_inputs:
        # --- Load Graph ---
        G = load_graph(place)

        if G:
            st.success(f"Graph loaded for {place}.")

            # --- Load POIs ---
            poi_gdf = None
            poi_tree = None
            if selected_poi_tags:
                poi_gdf = load_pois_bbox(G, place, selected_poi_tags)
                if poi_gdf is not None and not poi_gdf.empty:
                    st.success(f"Loaded {len(poi_gdf)} POIs.")
                    # Create spatial index for faster lookups
                    try:
                        from shapely.strtree import STRtree

                        poi_tree = STRtree(poi_gdf.geometry)
                    except ImportError:
                        st.warning(
                            "shapely.strtree not available. Install STRtree (`pip install strtree`) or upgrade shapely for POI indexing."
                        )
                        poi_tree = None
                    except Exception as tree_error:
                        st.warning(f"Could not create POI spatial index: {tree_error}")
                        poi_tree = None  # Ensure it's None if creation fails
                else:
                    st.warning("No POIs found for selected tags or error loading POIs.")
            else:
                st.info("No POI categories selected.")

            # --- Calculate Route ---
            path = None
            try:
                factors = ROAD_TYPE_FACTORS.copy()

                if route_type == "Circular":
                    path = route_circular(
                        G,
                        origin_val,
                        target_km,
                        length_w=length_w,
                        elev_w=elev_w,
                        factors=factors,
                        poi_tree=poi_tree,
                        poi_buffer=poi_buffer,
                        poi_reward=poi_reward,
                        tol_frac=tol_frac,
                        reuse_penalty=reuse_penalty,
                        sector_guidance=sector_guidance,
                        sector_weight=sector_weight,
                        seed=42,
                    )
                elif route_type == "Point-to-point":
                    path = route_point_to_point(
                        G,
                        origin_val,
                        dest_val,
                        length_w=length_w,
                        elev_w=elev_w,
                        factors=factors,
                        poi_tree=poi_tree,
                        poi_buffer=poi_buffer,
                        poi_reward=poi_reward,
                    )

                if path:
                    st.success("Route calculated successfully!")

                    # --- Calculate Stats ---
                    path_length_m = 0
                    total_elev_gain = 0
                    try:
                        path_length_m = sum(
                            G.edges[u, v, 0]["length"]
                            for u, v in zip(path[:-1], path[1:])
                        )
                        st.metric("Route Length", f"{path_length_m / 1000:.2f} km")
                        for u, v in zip(path[:-1], path[1:]):
                            if G.has_edge(u, v):
                                edge_data = G.get_edge_data(u, v, 0)
                                total_elev_gain += edge_data.get("elev_gain", 0)
                        st.metric("Total Elevation Gain", f"{total_elev_gain:.1f} m")
                    except Exception as stats_error:
                        st.error(f"Error calculating route stats: {stats_error}")
                        st.error(traceback.format_exc())

                    # --- Create Plot Object ---
                    try:
                        # 1) Grab the unprojected (lat/lon) graph cached in load_graph
                        G_ll = st.session_state.get("unprojected_graph")
                        if G_ll is None:
                            st.error("Unprojected graph not found — cannot plot route.")
                            st.session_state.generated_map = None
                        else:
                            # 2) Extract (lat, lon) pairs directly from G_ll.nodes
                            route_coords_ll = [
                                (
                                    G_ll.nodes[node]["y"],
                                    G_ll.nodes[node]["x"],
                                )  # (lat, lon)
                                for node in path
                            ]

                            # 3) Build folium map in WGS84
                            map_center = route_coords_ll[0]
                            plot_map = folium.Map(
                                location=map_center,
                                zoom_start=14,
                                tiles="CartoDB positron",
                            )
                            folium.PolyLine(
                                route_coords_ll,
                                color="blue",
                                weight=5,
                                opacity=0.7,
                                tooltip=f"Length: {path_length_m/1000:.2f} km",
                            ).add_to(plot_map)

                            # 4) Add start/end markers in lat/lon
                            folium.Marker(
                                route_coords_ll[0],
                                popup=f"Start\n{path_length_m/1000:.2f} km, ↑{total_elev_gain:.0f} m",
                                icon=folium.Icon(color="green", icon="play"),
                            ).add_to(plot_map)
                            if route_type == "Point-to-point":
                                folium.Marker(
                                    route_coords_ll[-1],
                                    popup="End",
                                    icon=folium.Icon(color="red", icon="stop"),
                                ).add_to(plot_map)

                            # 5) Fit bounds & store
                            plot_map.fit_bounds(route_coords_ll)
                            st.session_state.generated_map = plot_map

                    except Exception as plot_error:
                        st.error(f"Error creating map object: {plot_error}")
                        st.error(traceback.format_exc())
                        st.session_state.generated_map = None

                else:  # path is None or empty
                    st.warning(
                        "Route calculation did not return a valid path. Cannot create map."
                    )
                    st.session_state.generated_map = None

            except Exception as e:
                st.error(f"An error occurred during route generation main block: {e}")
                st.error(traceback.format_exc())
                st.session_state.generated_map = None
        else:  # G is None
            st.error("Graph could not be loaded. Cannot generate route.")
            st.session_state.generated_map = None
    else:  # Inputs invalid
        st.warning("Inputs invalid, skipping route generation.")

# --- Display Area (Outside Button Logic) ---
# This section runs on every rerun, including the one after the button click
st.subheader("Route Map")
if st.session_state.generated_map:
    # Display the map stored in session state, ADDING A KEY
    st_folium(
        st.session_state.generated_map,
        key="folium_map_display",  # Add a fixed key here
        height=500,
        width=700,
        returned_objects=[],  # Usually empty unless you need map interactions back
    )
else:
    # Show placeholder if no map has been generated yet or if generation failed
    st.info("Generate a route to display the map here.")

# Optional: Add some info about the app
st.sidebar.divider()
st.sidebar.info(
    "This app uses OSMnx, NetworkX, and Streamlit to generate routes. "
    "Elevation data from SRTM. Basemap by OpenStreetMap contributors."
)

# Limit threads for numerical libraries
# import os
# os.environ["OMP_NUM_THREADS"] = "1"
# os.environ["OPENBLAS_NUM_THREADS"] = "1"
# os.environ["MKL_NUM_THREADS"] = "1"
