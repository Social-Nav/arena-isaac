# Low level APIs
import os
import time
from collections import deque

import carb
import numpy as np
import omni.anim.graph.core as ag

# High level Isaac sim APIs
import omni.client
from isaac_utils.utils.assets import get_assets_root_path_safe
from omni.anim.people import PeopleSettings
from isaacsim.core.utils import prims
from omni.usd import get_stage_next_free_path
from pxr import Gf, Sdf
from scipy.spatial.transform import Rotation

from pedestrian.simulator.logic.people.animation_clock import (
    AnimationGraphAcquisition,
    AnimationTickHealth,
    MANUAL_TICK_ENV_VAR,
    animation_tick_dt,
    manual_tick_disabled_message,
    manual_tick_enabled,
)
from pedestrian.simulator.logic.people.person_controller import PersonController
from pedestrian.simulator.logic.people_manager import PeopleManager

# Extension APIs
from pedestrian.simulator.logic.state import State


class Person:
    """
    Class that implements a person in the simulation world. The person can be controlled by a controller that inherits from the PersonController class.
    """

    # Get root assets path from setting, if not set, get the Isaac-Sim asset path
    setting_dict = carb.settings.get_settings()
    people_asset_folder = setting_dict.get(PeopleSettings.CHARACTER_ASSETS_PATH)
    character_root_prim_path = setting_dict.get(PeopleSettings.CHARACTER_PRIM_PATH)
    assets_root_path = None

    if not character_root_prim_path:
        character_root_prim_path = "/World/Characters"

    if people_asset_folder:
        assets_root_path = people_asset_folder
    else:
        root_path = get_assets_root_path_safe()
        assets_root_path = os.path.join(root_path, 'Isaac/People/Characters')

    character_skel_root_stage_path: str

    def __init__(
        self,
        world,
        stage_prefix: str,
        character_name: str | None = None,
        init_pos=[0.0, 0.0, 0.0],
        init_yaw=0.0,
        controller: PersonController | None = None,
        backend=None
    ):
        """Initializes the person object

        Args:
            stage_prefix (str): The name the person will present in the simulator when spawned on the stage.
            character_name (str): The name of the person in the USD file. Use the Person.get_character_asset_list() method to get the list of available characters.
            init_pos (list): The initial position of the vehicle in the inertial frame (in ENU convention). Defaults to [0.0, 0.0, 0.0].
            init_yaw (float): The initial orientation of the person in rad. Defaults to 0.0.
            controller (PersonController): A controller to add some custom behaviour to the movement of the person. Defaults to None.
        """

        # Get the current world at which we want to spawn the vehicle
        self._world = world
        self._current_stage = self._world.stage

        # Variable that will hold the current state of the vehicle
        self._state = State()
        self._state.position = np.array(init_pos)
        self._state.orientation = Rotation.from_euler('z', init_yaw, degrees=False).as_quat()

        # Set the target position for the character
        self._target_positions = deque[np.ndarray]()
        self._target_speed = 0.0

        # Animation-graph state.  These must exist before the physics callbacks below are
        # registered: both callbacks reach for `character_graph` on their very first invocation.
        self._character_graph = None
        self._anim_acquisition = AnimationGraphAcquisition()
        self._anim_tick_enabled = manual_tick_enabled()
        self._anim_tick_health = AnimationTickHealth()
        self._anim_tick_broken_reported = False
        if not self._anim_tick_enabled:
            carb.log_warn(manual_tick_disabled_message())

        # Save the name with which the vehicle will appear in the stage
        # and the character model that will be loaded into the simulator
        self._stage_prefix = stage_prefix

        # The name of the character in the USD file
        self._character_name = character_name

        # Get the USD file corresponding to the character
        self.char_usd_file = Person.get_path_for_character_prim(character_name)

        # Spawn the agent in the world
        self.spawn_agent(self.char_usd_file, self._stage_prefix, init_pos, init_yaw)

        # Set the controller for the person if any and initialize it
        self._controller = controller
        if self._controller:
            self._controller.initialize(self)

        # Set the backend for publishing the state of the person
        self._backend = backend
        if self._backend:
            self._backend.initialize(self)

        # Add a callback to the physics engine to update the current state of the person
        if not self._world.physics_callback_exists(cb_path := self._stage_prefix + "/state"):
            self._world.add_physics_callback(cb_path, self.update_state)

        # Add the update method to the physics callback if the world was received
        # so that we can apply the new references to be tracked by the person
        if not self._world.physics_callback_exists(cb_path := self._stage_prefix + "/update"):
            self._world.add_physics_callback(cb_path, self.update)

        # Set the flag that signals if the simulation is running or not
        self._sim_running = False

        # Add a callback to start/stop of the simulation once the play/stop button is hit
        if not self._world.timeline_callback_exists(cb_path := self._stage_prefix + "/start_stop_sim"):
            self._world.add_timeline_callback(cb_path, self.sim_start_stop)

    @property
    def character_graph(self):
        """The animation graph of the person.

        Returns:
            CharacterGraph: The animation graph of the person, or None while it cannot be acquired.
        """
        if self._character_graph is None:
            self._acquire_character_graph()
        return self._character_graph

    def _acquire_character_graph(self):
        """Try once to acquire the animation graph, throttling USD authoring and never failing quietly.

        The animation graph is the only transport that moves a rendered pedestrian, so a permanent
        failure here freezes every pedestrian.  Before this method existed the property simply
        returned None, both physics callbacks returned early, and the run reported nothing at all.
        """
        now = time.monotonic()
        action = self._anim_acquisition.next_action(now)
        if action == "author":
            try:
                self.add_animation_graph_to_agent()
            except Exception as exc:
                carb.log_warn(
                    f"[PedestrianAnimation] applying the AnimationGraphAPI to "
                    f"{self.character_skel_root_stage_path} raised {exc!r}; will retry"
                )
        graph = ag.get_character(self.character_skel_root_stage_path)
        if graph:
            self._character_graph = graph
            carb.log_warn(self._anim_acquisition.success_message(
                now, self.character_skel_root_stage_path))
            return
        message = self._anim_acquisition.failure_message(
            now, self.character_skel_root_stage_path)
        if message:
            carb.log_error(message)

    @property
    def state(self):
        """The state of the person.

        Returns:
            State: The current state of the person, i.e., position, orientation, linear and angular velocities...
        """
        return self._state

    def sim_start_stop(self, event):
        """
        Callback that is called every time there is a timeline event such as starting/stoping the simulation.

        Args:
            event: A timeline event generated from Isaac Sim, such as starting or stoping the simulation.
        """

        # If the start/stop button was pressed, then call the start and stop methods accordingly
        if self._world.is_playing() and self._sim_running == False:
            self._sim_running = True
            self.start()

        if self._world.is_stopped() and self._sim_running == True:
            self._sim_running = False
            self.stop()

    def start(self):
        """
        Method that is called when the simulation starts. This method can be used to initialize any variables.
        """
        if self._controller:
            self._controller.start()

    def stop(self):
        """
        Method that is called when the simulation stops. This method can be used to reset any variables.
        """
        if self._controller:
            self._controller.stop()

    def update(self, dt: float):
        """
        Method that implements the logic to make the person move around in the simulation world and also play the animation

        Args:
            dt (float): The time elapsed between the previous and current function calls (s).
        """

        # Note: this is done to avoid the error of the character_graph being None. The animation graph is only created after the simulation starts
        if not self.character_graph:
            # failed to acquire character graph
            return

        # Call the controller update method that should update the reference of the target position
        if self._controller:
            self._controller.update(dt)

        THRESHOLD_DISTANCE = 0.3  # m
        while self._target_positions and np.linalg.norm(self._target_position - self._state.position) < THRESHOLD_DISTANCE:
            # set next target
            self._target_positions.popleft()

        if self._target_positions:
            # targets not empty
            extended_target = self._target_position + ((self._target_position - self._state.position) / np.linalg.norm(self._target_position - self._state.position)) * THRESHOLD_DISTANCE
            self.character_graph.set_variable("PathPoints", [carb.Float3(self._state.position), carb.Float3(extended_target)])
            self.character_graph.set_variable("Action", "Walk")
            self.character_graph.set_variable("Walk", self._target_speed)
            commanded_speed = float(self._target_speed)

        else:
            # at target position, stop moving
            self.character_graph.set_variable("Walk", 0.0)
            self.character_graph.set_variable("Action", "Idle")
            commanded_speed = 0.0

        # Advance the animation by the physics dt that just elapsed.  Without this the graph is only
        # evaluated on Kit application updates, i.e. once per ARENA_ISAAC_RENDER_EVERY_N_STEPS
        # physics steps, and the rendered character receives a fraction of the locomotion time it
        # needs.  This does not change the render cadence in any way.
        self._tick_character_animation(dt, commanded_speed)

        # If we have a backend, update the state of the person
        if self._backend:
            self._backend.update(self._state, dt)

        # if self.character_skel_root_stage_path is not None:
        #     PeopleManager.get_people_manager().add_person(self.character_skel_root_stage_path, self)

    def _tick_character_animation(self, dt: float, commanded_speed: float):
        """Advance this character's animation graph by the elapsed physics ``dt``.

        ``omni.anim.graph.core`` normally evaluates the graph from Kit's application update, which
        the eval loop performs only on rendered steps, so the animation was starved of time.
        ``Character.update(dt)`` supplies the time explicitly and needs no application update.

        The full ``dt`` is supplied on every step, rendered ones included: once a character has been
        ticked manually its automatic evaluation stops, so skipping the rendered step would lose that
        step's time rather than avoid double counting.  See ``animation_clock.animation_tick_dt``.

        Failure is never silent.  ``Character.update()`` returns ``None`` and does not raise when the
        timeline is not playing, so the return value cannot be trusted; progress is measured instead.
        """
        graph = self._character_graph
        if graph is None or not self._anim_tick_enabled:
            return

        tick = getattr(graph, "update", None)
        if not callable(tick):
            if not self._anim_tick_broken_reported:
                self._anim_tick_broken_reported = True
                carb.log_error(
                    "[PedestrianAnimation] omni.anim.graph.core Character has no update(dt): "
                    "cannot drive the animation from the physics step, so rendered pedestrians will "
                    f"lag the logical positions the evaluation grades against. Character type is "
                    f"{type(graph).__name__}. Set {MANUAL_TICK_ENV_VAR}=0 to accept that "
                    "deliberately."
                )
            return

        try:
            tick(animation_tick_dt(dt))
        except Exception as exc:
            if not self._anim_tick_broken_reported:
                self._anim_tick_broken_reported = True
                carb.log_error(
                    f"[PedestrianAnimation] Character.update(dt) raised {exc!r} for "
                    f"{self.character_skel_root_stage_path}; rendered pedestrians will lag the "
                    "logical positions the evaluation grades against"
                )
            return

        report = self._anim_tick_health.record(dt, commanded_speed, self._state.position)
        if report is not None:
            severity, message = report
            if severity == "error":
                carb.log_error(message)
            else:
                carb.log_warn(message)

    def update_target_positions(self, positions, walk_speed=1.0):
        """
        Method that updates the target position of the person to which it will move towards.

        Args:
            position (list): A list with the x, y, z coordinates of the target position.
        """
        self._target_positions.extend(positions)
        self._target_speed = walk_speed

    def update_state(self, dt: float):
        """
        Method that is called at every physics step to retrieve and update the current state of the person, i.e., get
        the current position, orientation, linear and angular velocities and acceleration of the person.

        Args:
            dt (float): The time elapsed between the previous and current function calls (s).
        """

        if not self.character_graph:
            # failed to acquire character graph
            return

        # Get the current position of the person
        pos = carb.Float3(0, 0, 0)
        rot = carb.Float4(0, 0, 0, 0)
        self.character_graph.get_world_transform(pos, rot)

        # Update the current state of the person
        self._state.position = np.array([pos[0], pos[1], pos[2]])
        self._state.orientation = np.array([rot.x, rot.y, rot.z, rot.w])

        # Signal the controller the updated state
        if self._controller:
            self._controller.update_state(self._state)

    def spawn_agent(self, usd_file, stage_name, init_pos, init_yaw):

        # If there is no XForm primitive in the stage to hold all the people, create one
        if not self._current_stage.GetPrimAtPath(Person.character_root_prim_path):
            prims.create_prim(Person.character_root_prim_path, "Xform")

        # If the base biped character is not present in the stage, spawn it
        if not self._current_stage.GetPrimAtPath(Person.character_root_prim_path + "/Biped_Setup"):
            prim = prims.create_prim(Person.character_root_prim_path + "/Biped_Setup", "Xform", usd_path=Person.assets_root_path + "/Biped_Setup.usd")
            prim.GetAttribute("visibility").Set("invisible")

        # Spawn the person in the world
        self.prim = prims.create_prim(stage_name, "Xform", usd_path=usd_file)
        self._flush_app()
        
        # Set the initial position and orientation of the person
        self.prim.GetAttribute("xformOp:translate").Set(Gf.Vec3d(float(init_pos[0]), float(init_pos[1]), float(init_pos[2])))

        if type(self.prim.GetAttribute("xformOp:orient").Get()) == Gf.Quatf:
            self.prim.GetAttribute("xformOp:orient").Set(Gf.Quatf(Gf.Rotation(Gf.Vec3d(0, 0, 1), float(init_yaw)).GetQuat()))
        else:
            self.prim.GetAttribute("xformOp:orient").Set(Gf.Rotation(Gf.Vec3d(0, 0, 1), float(init_yaw)).GetQuat())

        # Get the Skeleton root of the character
        self.character_skel_root, root_path = Person._transverse_prim(self._current_stage, self._stage_prefix)
        if root_path is None:
            raise RuntimeError(f"Could not find SkelRoot for character {self._character_name} at stage prefix {self._stage_prefix}")
        self.character_skel_root_stage_path = root_path

        # Add the current person to the person manager
        PeopleManager.get_people_manager().add_person(self._stage_prefix, self)

    def add_animation_graph_to_agent(self):
        """Author the AnimationGraphAPI onto this character's SkelRoot.

        Note on ``RemoveAnimationGraphAPICommand``.  ``docs/benchmark/troubleshooting.md:238`` tells
        reviewers to reject patches that call it *from the HuNav replay path*, because a direct-pose
        workaround used it to detach the graph and slide the prim by hand.  The call below is the
        opposite: it is a setup-time remove-then-reapply so ``ApplyAnimationGraphAPICommand`` cannot
        collide with an API already present (e.g. from the character asset, or from a previous spawn
        at the same stage path), and the graph is left *enabled* two lines later.  It never writes a
        transform.  It is only reached while ``character_graph`` is unresolved, and
        ``_acquire_character_graph`` throttles how often that happens -- an unthrottled
        remove-then-reapply on every physics step could tear down an API that was in the middle of
        becoming visible through Fabric.
        """

        # Get the animation graph that we are going to add to the person
        animation_graph = self._current_stage.GetPrimAtPath(Person.character_root_prim_path + "/Biped_Setup/CharacterAnimation/AnimationGraph")

        # Remove the animation graph attribute if it exists
        if self.character_skel_root is not None:
            omni.kit.commands.execute("RemoveAnimationGraphAPICommand", paths=[Sdf.Path(self.character_skel_root.GetPrimPath())])

        # Add the animation graph to the character
        if self.character_skel_root is not None:
            omni.kit.commands.execute("ApplyAnimationGraphAPICommand", paths=[Sdf.Path(self.character_skel_root.GetPrimPath())], animation_graph_path=Sdf.Path(animation_graph.GetPrimPath()))
            self._flush_app()

    @staticmethod
    def _flush_app():
        if str(os.environ.get("ARENA_ISAAC_PERSON_FLUSH_APP_UPDATE", "0")).strip().lower() in {"0", "false", "no", "off"}:
            return
        try:
            omni.kit.app.get_app().update()
        except Exception:
            pass

    @staticmethod
    def _transverse_prim(stage, stage_prefix):

        # Check if the prim is the one we are looking for
        prim = stage.GetPrimAtPath(stage_prefix)

        # If the prim is the one we are looking for, return it
        if prim.GetTypeName() == "SkelRoot":
            return prim, stage_prefix

        # Otherwise, get all the children of the prim and keep transversing until we find the SkelRoot
        children = prim.GetAllChildren()

        # If there are no children, return
        if not children or len(children) == 0:
            return None, None

        # Recursively look through the children to get the SkelRoot
        for child in children:
            prim_child, child_stage_prefix = Person._transverse_prim(stage, stage_prefix + "/" + child.GetName())

            if prim_child is not None:
                return prim_child, child_stage_prefix

        return None, None

    @staticmethod
    def get_character_asset_list():
        # List all files in characters directory
        result, folder_list = omni.client.list("{}/".format(Person.assets_root_path))

        if result != omni.client.Result.OK:
            carb.log_error("Unable to get character assets from provided asset root path.")
            return

        # Prune items from folder list that are not directories.
        pruned_folder_list = [folder.relative_path for folder in folder_list
                              if (folder.flags & omni.client.ItemFlags.CAN_HAVE_CHILDREN) and not folder.relative_path.startswith(".")]

        return pruned_folder_list

    @staticmethod
    def get_path_for_character_prim(agent_name):

        # Check if a folder with agent_name exists. If exists we load the character, else we load a random character
        agent_folder = os.path.join(Person.assets_root_path, agent_name)
        result, properties = omni.client.stat(agent_folder)

        # Attempt to load the character if it exists, otherwise load a random character
        if result != omni.client.Result.OK:
            carb.log_error(f"Character folder does not exist: {agent_name}. Available: {Person.get_character_asset_list()}")
            return None

        # Get the usd present in the character folder
        character_folder = "{}/{}".format(Person.assets_root_path, agent_name)
        character_usd = Person.get_usd_in_folder(character_folder)

        # Return the character name (folder name) and the usd path to the character
        return "{}/{}".format(character_folder, character_usd)

    @staticmethod
    def get_usd_in_folder(character_folder_path):
        result, folder_list = omni.client.list(character_folder_path)

        if result != omni.client.Result.OK:
            carb.log_error("Unable to read character folder path at {}".format(character_folder_path))
            return

        for item in folder_list:
            if item.relative_path.endswith(".usd"):
                return item.relative_path

        carb.log_error("Unable to file a .usd file in {} character folder".format(character_folder_path))

    def destroy(self):
        """
        Method that will delete the person from the simulation world.
        """

        # Remove the physics callbacks (guard against already-removed callbacks)
        for cb in (self._stage_prefix + "/state", self._stage_prefix + "/update"):
            try:
                if self._world.physics_callback_exists(cb):
                    self._world.remove_physics_callback(cb)
            except Exception as e:
                carb.log_warn(f"Exception while removing physics callback '{cb}': {e}")

        # Remove the timeline callback
        try:
            if self._world.timeline_callback_exists(self._stage_prefix + "/start_stop_sim"):
                self._world.remove_timeline_callback(self._stage_prefix + "/start_stop_sim")
        except Exception as e:
            carb.log_warn(f"Exception while removing timeline callback: {e}")

        # Delete the prim from the stage
        try:
            prims.delete_prim(self._stage_prefix)
        except Exception as e:
            carb.log_warn(f"Exception while deleting prim '{self._stage_prefix}': {e}")
        # NOTE: do NOT call PeopleManager.remove_person() here;
        # remove_person() already called destroy() to get here, calling back
        # would be an infinite recursion and uses the wrong key anyway.


    @property
    def position(self) -> np.ndarray:
        return self._state.position

    @property
    def _target_position(self) -> np.ndarray:
        if not self._target_positions:
            return np.array(self._state.position)
        return np.array(self._target_positions[0])

    @property
    def last_waypoint(self) -> np.ndarray:
        if not self._target_positions:
            return np.array(self._state.position)
        return np.array(self._target_positions[-1])

    @property
    def path(self) -> str:
        return self._stage_prefix
