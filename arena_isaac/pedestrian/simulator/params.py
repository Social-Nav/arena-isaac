import os
from pathlib import Path

from isaac_utils.utils.assets import get_assets_root_path_safe

# Get the current directory of where this extension is located
EXTENSION_FOLDER_PATH = Path(os.path.dirname(os.path.realpath(__file__)))
ROOT = str(EXTENSION_FOLDER_PATH.parent.parent.parent.resolve())

# Define the Extension Assets Path
ASSET_PATH = ROOT + "/pegasus.simulator/pegasus/simulator/assets"
ROBOTS_ASSETS = ASSET_PATH + "/Robots"

# Define the built in robots of the extension
ROBOTS = {"Iris": ROBOTS_ASSETS + "/Iris/iris.usd"}  # , "Flying Cube": ROBOTS_ASSETS + "/iris_cube.usda"}

# Setup the default simulation environments path
NVIDIA_ASSETS_PATH = get_assets_root_path_safe()

ISAAC_SIM_ENVIRONMENTS = "/Isaac/Environments"
NVIDIA_SIMULATION_ENVIRONMENTS = {
    "Default Environment": "Grid/default_environment.usd",
    "Black Gridroom": "Grid/gridroom_black.usd",
    "Curved Gridroom": "Grid/gridroom_curved.usd",
    "Hospital": "Hospital/hospital.usd",
    "Office": "Office/office.usd",
    "Simple Room": "Simple_Room/simple_room.usd",
    "Warehouse": "Simple_Warehouse/warehouse.usd",
    "Warehouse with Forklifts": "Simple_Warehouse/warehouse_with_forklifts.usd",
    "Warehouse with Shelves": "Simple_Warehouse/warehouse_multiple_shelves.usd",
    "Full Warehouse": "Simple_Warehouse/full_warehouse.usd",
    "Flat Plane": "Terrains/flat_plane.usd",
    "Rough Plane": "Terrains/rough_plane.usd",
    "Slope Plane": "Terrains/slope.usd",
    "Stairs Plane": "Terrains/stairs.usd",
}

OMNIVERSE_ENVIRONMENTS = {
    "Exhibition Hall": "omniverse://localhost/NVIDIA/Assets/Scenes/Templates/Interior/ZetCG_ExhibitionHall.usd"
}

SIMULATION_ENVIRONMENTS = {}

# Add the Isaac Sim assets to the list
for asset in NVIDIA_SIMULATION_ENVIRONMENTS:
    SIMULATION_ENVIRONMENTS[asset] = (
        os.path.join(NVIDIA_ASSETS_PATH, ISAAC_SIM_ENVIRONMENTS, NVIDIA_SIMULATION_ENVIRONMENTS[asset])
    )

# Add the omniverse assets to the list
for asset in OMNIVERSE_ENVIRONMENTS:
    SIMULATION_ENVIRONMENTS[asset] = OMNIVERSE_ENVIRONMENTS[asset]

# Define the default settings for the simulation environment
WORLD_SETTINGS = {
    'ros2': {
        "physics_dt": 1.0 / 250.0,
        "stage_units_in_meters": 1.0,
        "rendering_dt": 1.0 / 60.0,
        "device": "cpu"
    }
}
DEFAULT_WORLD_SETTINGS = WORLD_SETTINGS['ros2']
