import base64
import copy
import io
import json
import logging
import numpy as np
import os
import pickle
from PIL import ImageDraw, ImageFilter, Image
import threading
import time
import json # Ensure json is imported

from config import MAX_TOKENS, TEMPERATURE, DIRECT_NAVIGATION, SAMBANOVA_BASE_URL, SAMBANOVA_VISION_MODEL_ID, SAMBANOVA_TOOL_MODEL_ID, SAMBANOVA_STRATEGIST_MODEL_ID
from agent.prompts import SYSTEM_PROMPT, FULL_NAVIGATOR_PROMPT, NAVIGATION_PROMPT, META_KNOWLEDGE_PROMPT, META_KNOWLEDGE_CLEANUP_PROMPT, META_KNOWLEDGE_SUMMARIZER, SAMBANOVA_STRATEGIST_PROMPT # Added SAMBANOVA_STRATEGIST_PROMPT
from secret_api_keys import *
from agent.emulator import Emulator
from agent.tool_definitions import *
# from agent.utils import convert_anthropic_message_history_to_google_format, extract_tool_calls_from_gemini # Removed

from openai import OpenAI # Ensure this import is present
# Remove Anthropic and Gemini clients as they are no longer used
# from anthropic import Anthropic
# from google import genai
# from google.genai import types
# from google.genai.errors import ServerError
from openai.types import responses
from openai import BadRequestError

from typing import Any, Optional

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

BASE_IMAGE_SIZE = (160, 144)  # In tiles, 16 x 10 columns and 16 x 9 rows

# MAX_TOKENS_OPENAI = 50000 # Removed previously, verifying

# Handles making an automatically updating collision map of an area as the model paths through it.
class LocationCollisionMap:
    def __init__(self, initial_collision_map: np.ndarray, initial_sprite_locations: set[tuple[int, int]], initial_coords: tuple[int, int]):
        # initial_collision_map is a 9 x 10 player-centered collision map which is 0 is impassable and 1 otherwise
        # Internally we store an expanding map based on locations we've been, with -1 in unknown spots, 2 in sprite locations, 3 in player location, and otherwise 0/1 as well.
        # We just accept that moving NPC locations are going to be inaccurate.
        # Note that while player coords are column, row, by default what we get from the collision map tooling is row, column
        self.player_coords = initial_coords
        self.col_offset = initial_coords[0] - 4
        self.row_offset = initial_coords[1] - 4
        self.internal_map = -np.ones((10, 9), dtype=np.int8) # We make our map the "proper" order that everything else is.
        for row in range(9):
            for col in range(10):
                if row == 4 and col == 4:  # player character
                    self.internal_map[col][row] = 3
                elif (col, row) in initial_sprite_locations:
                    self.internal_map[col][row] = 2
                else:
                    self.internal_map[col][row] = initial_collision_map[row][col]
        self.distances: dict[tuple[int, int], int] = {}

    def update_map(self, collision_map: np.ndarray, sprite_locations: set[tuple[int, int]], coords: tuple[int, int]):
        # Remove the previous player marker. Most convenient to do it right now.
        self.internal_map[self.player_coords[0] - self.col_offset][self.player_coords[1] - self.row_offset] = 1
        self.player_coords = coords

        new_min_col = coords[0] - 4
        new_min_row = coords[1] - 4
        new_max_col = coords[0] + 5
        new_max_row = coords[1] + 4
        cur_size = self.internal_map.shape
        # First check if we need to move the boundaries of the array. Numpy pad makes this easy!
        expand_col_front = self.col_offset - new_min_col if new_min_col < self.col_offset else 0
        expand_col_back = new_max_col - (self.col_offset + cur_size[0] - 1) 
        expand_col_back = expand_col_back if expand_col_back > 0 else 0
        expand_row_front = self.row_offset - new_min_row if new_min_row < self.row_offset else 0
        expand_row_back = new_max_row - (self.row_offset + cur_size[1] - 1)
        expand_row_back = expand_row_back if expand_row_back > 0 else 0
        self.internal_map = np.pad(self.internal_map, pad_width=[(expand_col_front, expand_col_back), (expand_row_front, expand_row_back)], constant_values=-1)

        self.col_offset = min(new_min_col, self.col_offset)
        self.row_offset = min(new_min_row, self.row_offset)

        # Now update the map
        local_col_offset = new_min_col - self.col_offset
        local_row_offset = new_min_row - self.row_offset
        for row in range(9):
            for col in range(10):
                if row == 4 and col == 4:  # player character
                    self.internal_map[col + local_col_offset][row + local_row_offset] = 3
                    continue
                # if self.internal_map[col + local_col_offset][row + local_row_offset] != -1:
                    # continue  # No need to set, and if we do it's just going to lead to extra sprites for moving NPCs
                if (col, row) in sprite_locations:
                    self.internal_map[col + local_col_offset][row + local_row_offset] = 2
                else:
                    self.internal_map[col + local_col_offset][row + local_row_offset] = collision_map[row][col]
        self.distances = self.compute_effective_distance_to_tiles()

    def compute_effective_distance_to_tiles(self) -> dict[tuple[int, int], int]:
        # Basically do a distance fill
        depth = 99
        visited_tiles = set([self.player_coords])
        cur_tiles = set([self.player_coords])
        distances: dict[tuple[int, int], int] = {}
        for d in range(depth):
            new_tiles: set[tuple[int, int]] = set()
            for tile in cur_tiles:
                candidate_tiles = ((tile[0] + 1, tile[1]), (tile[0] - 1, tile[1]), (tile[0], tile[1] + 1), (tile[0], tile[1] - 1))  # I feel like there's a smarter way
                for candidate in candidate_tiles:
                    shifted_col = candidate[0] - self.col_offset
                    shifted_row = candidate[1] - self.row_offset
                    if shifted_col < 0 or shifted_row < 0 or shifted_col > self.internal_map.shape[0] - 1 or shifted_row > self.internal_map.shape[1] - 1:
                        continue
                    if candidate in visited_tiles:
                        continue
                    if self.internal_map[shifted_col][shifted_row] == 1:   # the only passable scenario
                        new_tiles.add(candidate)
                        distances[candidate] = d + 1
                    visited_tiles.add(candidate)
            cur_tiles = new_tiles
        return distances

    def generate_buttons_to_coord(self, col: int, row: int) -> Optional[list[str]]:
        starting_distance = self.distances.get((col, row))
        if starting_distance is None:
            return None # invalid
        distance = starting_distance
        button_list = []
        # Basically look for tiles that are labelled with successively lower numbers
        while distance > 0:
            # just pick whichever happens to work first.
            left = self.distances.get((col - 1, row))
            if (left and left == distance - 1) or (col - 1, row) == self.player_coords:
                button_list.append("right")
                col -= 1
                distance -= 1
                continue
            right = self.distances.get((col + 1, row))
            if (right and right == distance - 1) or (col + 1, row) == self.player_coords:
                button_list.append("left")
                distance -= 1
                col += 1
                continue
            up = self.distances.get((col, row - 1))
            if (up and up == distance - 1) or (col, row - 1) == self.player_coords:
                button_list.append("down")
                distance -= 1
                row -= 1
                continue
            down = self.distances.get((col, row + 1))
            if (down and down == distance - 1) or (col, row + 1) == self.player_coords:
                button_list.append("up")
                distance -= 1
                row += 1
                continue
            breakpoint()
        
        # now reverse and return
        button_list.reverse()
        return button_list
    
    @staticmethod
    def make_ascii_segment(input_str: str, width: int, col: int, row: int):
        # Basically output ascii map blocks of a consistent width, using a given input_str and local coordinates. Also adds | on the front side and includes it in the width.
        base_str = f"{input_str}({col},{row})"
        # pads always at the end.
        if len(base_str) > width - 1:
            raise ValueError("Not enough space to fit this!")
        base_str += (width - 1 - len(base_str))*" "
        return f"|{base_str}"

    def to_ascii(self, local_location_tracker: Optional[list[list[bool]]]=None) -> str:

        # We prepare two identical versions simultaneously: A readable nice ASCII for humans, and the long-winded one for models

        horizontal_labels = list(range(self.col_offset, self.col_offset+self.internal_map.shape[0]))

        
        row_width = 35
        horizontal_border = "       +" + "".join("Column " + str(x) + " "*(row_width - len(str(x)) - 7) for x in horizontal_labels) + "+"
        horizontal_border_human = "       +" + "".join(str(x) + " "*(4-len(str(x))) for x in horizontal_labels) + "+"

        lines = []
        lines_human = []
        # Add legend to human version
        if local_location_tracker:
            lines_human.extend(
                [
                    "",
                    "Legend:",
                    "██ - Wall/Obstacle",
                    "·· - CHECK HERE: Path/Walkable",
                    "SS - Sprite",
                    "PP - Player Character",
                    "xx - AVOID GOING HERE - Already Explored",
                    "uu - CHECK HERE: Blank = Unknown/Unvisited",
                    "Numbers - How many tiles away this tile is to reach."
                ]
            )
        else:
            lines_human.extend(
                [
                    "",
                    "Legend:",
                    "██ - Wall/Obstacle",
                    "·· - Path/Walkable",
                    "SS - Sprite",
                    "PP - Player Character",
                    "uu - Blank = Unknown/Unvisited"
                ]
            )

        lines += [f"({self.col_offset}, {self.row_offset})", horizontal_border]
        lines_human += [f"({self.col_offset}, {self.row_offset})", horizontal_border_human]
        for row_num, this_row in enumerate(self.internal_map.transpose()):  # transposing makes printing easier
            real_row = self.row_offset + row_num
            row = f"Row: {str(real_row) + ' ' * (2 - len(str(real_row)))}"
            row_human = row + "|"
            for col_num, col in enumerate(this_row):
                real_col = self.col_offset + col_num
                if col == -1:
                    row += self.make_ascii_segment("Check here", row_width, real_col, real_row)
                    row_human += " uu "
                elif col == 0:
                    row += self.make_ascii_segment("Impassable", row_width, real_col, real_row)
                    row_human += " ██ "
                elif col == 1: 
                    # Potentially place a distance marker:
                    row_piece = ""
                    row_piece_human = ""
                    distance = self.distances.get((real_col, real_row))
                    if distance:  # removes 0 and None
                        row_piece += "StepsToReach:" + str(distance) + " " * (4 - len(str(distance))) + " "
                        row_piece_human += str(distance) + " " * (4 - len(str(distance)))
                    if local_location_tracker and real_col > -1 and real_row > -1 and real_col < len(local_location_tracker) and real_row < len(local_location_tracker[real_col]) and local_location_tracker[real_col][real_row]:
                        row_piece += "Explored"
                        if not row_piece_human:
                            row_human += " xx "
                    else:
                        row_piece += "Passable"
                        if not row_piece_human:
                            row_human += " ·· "
                    row += self.make_ascii_segment(row_piece, row_width, real_col, real_row)
                    row_human += row_piece_human
                elif col == 2:
                    row += self.make_ascii_segment("NPC/Object", row_width, real_col, real_row)
                    row_human += " SS "
                elif col == 3:
                    row += self.make_ascii_segment("PLAYER", row_width, real_col, real_row)
                    row_human += " PP "
            row += f"|{str(real_row)}"
            row_human += f"|{str(real_row)}"
            lines.append(row)
            lines_human.append(row_human)
        lines.append(horizontal_border + f"({self.col_offset + self.internal_map.shape[0] - 1}, {self.row_offset + self.internal_map.shape[1] - 1})")
        lines_human.append(horizontal_border_human + f"({self.col_offset + self.internal_map.shape[0] - 1}, {self.row_offset + self.internal_map.shape[1] - 1})")


        # Join all lines with newlines
        output = "\n".join(lines)
        with open("mapping_log.txt", "w", encoding="utf-8") as fw:
            fw.write("\n".join(lines_human))
            fw.write("\n\n" + "MODEL VERSION:" +"\n\n")
            fw.write(output)
        return output

# Updates a text file over time to write the last X blocks of text to a text file so we can see it well
# with tail -F or something.
# TODO: This should definitely be a display buffer or something, not 100000 writes to the hard disk.
class TextDisplay:
    FILE_NAME = "text_output.txt"

    def __init__(self, message_depth=20):
        self.text_buffer = []
        self.message_depth = message_depth

    def add_message(self, message: str):
        self.text_buffer.append(message)
        logger.info(message)
        if len(self.text_buffer) > self.message_depth:
            self.text_buffer = self.text_buffer[1:]
        with open(self.FILE_NAME, "w", encoding="utf-8") as fw:
            fw.write("\n\n".join(self.text_buffer))


class SimpleAgent:
    def __init__(
        self, 
        rom_path, 
        headless=True, 
        sound=False, 
        max_history=60, 
        load_state=None, 
        location_history_length=40, 
        location_archive_file_name: Optional[str]=None, 
        use_full_collision_map: bool=True,
        pyboy_main_thread: bool=False
    ):
        """Initialize the simple agent.

        Args:
            rom_path: Path to the ROM file
            headless: Whether to run without display
            sound: Whether to enable sound
            max_history: Maximum number of messages in history before summarization
        """
        self.emulator = Emulator()
        self.pyboy_main_thread = pyboy_main_thread
        self.emulator_init_kwargs = {"rom_path": rom_path, "headless": headless, "sound": sound, "pyboy_main_thread": self.pyboy_main_thread}
        if not self.pyboy_main_thread:
            self.emulator.initialize(**self.emulator_init_kwargs)
        
        # Initialize SambaNova client
        self.sambanova_client = OpenAI(
            baseURL=SAMBANOVA_BASE_URL, 
            api_key=os.environ.get("SAMBANOVA_API_KEY")
        )

        self.running = True
        # TODO: OKAY LOOK this was originally a pretty small state and that it got out of hand.
        self.message_history = [{"role": "system", "content": "You are a helpful assistant."}]
        # self.openai_message_history = [{"role": "user", "content": "You may now begin playing."}] # Removed
        self.max_history = max_history
        self.location_history_length = location_history_length
        self.location_archive_file_name = location_archive_file_name
        self.location_history: list[tuple[str, tuple[int, int]]] = []
        # location -> row -> col -> label. By nesting the dicts we make the lookup
        # faster, but this is a crappy way of doing it.
        self.label_archive: dict[str, dict[int, dict[int, str]]] = {}
        # the dedicated location tracker that the model may turn on
        self.location_tracker_activated: bool = False
        self.location_tracker: dict[str, list[list[bool]]] = {}  # True if visited, False if not, could be numpy but eh for now.
        self.steps_since_checkpoint = 0
        self.steps_since_label_reset = 0
        self.last_location: Optional[str] = None
        self.map_tool_map: dict[str, str] = {}  # location -> map
        self.fully_mapped_locations: set[str] = set()  # Unused for now
        self.full_collision_map: dict[str, LocationCollisionMap] = {}
        self.use_full_collision_map = use_full_collision_map  # Do I need to save this?
        self.absolute_step_count = 0
        self.all_visited_locations: set[str] = set()
        self.location_milestones: list[tuple[str, int]] = []
        self.text_display = TextDisplay()
        self.last_coords = None  # A bit more precise, since it includes detailed trajectories from push button and navigate to.
        self.checkpoints = []  # A long-running list of achievements, used to track internal progress.
        self.detailed_navigator_mode = False # This will be re-evaluated or removed based on new model-specific task routing
        # self.navigator_message_history = [{"role": "user", "content": "Please begin navigating!"}] # Removed
        # self.openai_navigator_message_history = [{"role": "user", "content": "Please begin navigating!"}] # Removed
        self.steps_since_location_shift = 0
        self.no_navigate_here = ""
        self.navigation_location = ""
        self._steps_completed = 0
        self.load_state = load_state

        if load_state and not self.pyboy_main_thread:
            logger.info(f"Loading saved state from {load_state}")
            self.emulator.load_state(load_state)
            self.load_location_archive(location_archive_file_name)
        elif load_state:
            self.load_location_archive(location_archive_file_name)


    # This does the overlay for the model.
    def get_screenshot_base64(
            self, screenshot: Image.Image, upscale=1, add_coords: bool=True,
            player_coords: Optional[tuple[int, int]]=None, location: Optional[str]=None, relative_square_size=8):
        """Convert PIL image to base64 string."""
        # Resize if needed
        if upscale > 1:
            new_size = (screenshot.width * upscale, screenshot.height * upscale)
            screenshot = screenshot.resize(new_size)

        past_locations = self.location_history
        location_labels = self.label_archive.get(location)
        if location_labels is None:
            # this sucks man
            for key, value in self.label_archive.items():
                if location.lower() == key.lower():
                    location_labels = value
                    break
        if location_labels is None:
            location_labels = {}
        local_location_tracker = self.location_tracker.get(location, [])

        collision_map = self.emulator.pyboy.game_wrapper.game_area_collision()
        downsampled_terrain = self.emulator._downsample_array(collision_map)

        sprite_locations = self.emulator.get_sprites()

        if not self.emulator.get_in_combat():
            shape = screenshot.size
            # Draw some eye-searing lines across the image that nonetheless might make it more obvious to the LLM that this is a grid.
            for x in range(0, shape[0], shape[0]//10):
                ImageDraw.Draw(screenshot).line(((x, 0), (x, shape[1] - 1)), fill=(255, 0, 0))
            for y in range(0, shape[1], shape[1]//9):
                ImageDraw.Draw(screenshot).line(((0, y), (shape[0] - 1, y)), fill=(255, 0, 0))

            # add coordinate labels (note: if scale is too small it may be unreadable)
            # The assumption is the central square is the player's current location, which is 4, 4
            # Rows 0 - 8, Cols 0 - 9
            if add_coords:
                assert player_coords is not None
                tile_size = 16 * upscale
                mid_length = tile_size/2
                for row in range(0, 9):
                    # For bad legacy reasons location labels is row first
                    real_row = player_coords[1] + row - 4
                    local_cols = location_labels.get(real_row, {})
                    for col in range(0, 10):
                        if row == 4 and col == 4:
                            continue  # Skip the player themselves.
                        real_col = player_coords[0] + col - 4
                        label = local_cols.get(real_col, "")
                        tile_label = f"{str(real_col)}, {str(real_row)}"
                        if label:
                            tile_label += "\n" + label
                        if (col, row) not in sprite_locations:
                            if downsampled_terrain[row][col] == 0:
                                # ImageDraw.Draw(screenshot).rectangle(((col * tile_size + (relative_square_size - 1)*mid_length/relative_square_size, row * tile_size + (relative_square_size - 1)*mid_length/relative_square_size), (col * tile_size + (relative_square_size + 1)*mid_length/relative_square_size, row * tile_size + (relative_square_size + 1)*mid_length/relative_square_size)), (255, 0, 0))
                                tile_label += "\n" + "IMPASSABLE"
                            else:
                                # ImageDraw.Draw(screenshot).rectangle(((col * tile_size + (relative_square_size - 1)*mid_length/relative_square_size, row * tile_size + (relative_square_size - 1)*mid_length/relative_square_size), (col * tile_size + (relative_square_size + 1)*mid_length/relative_square_size, row * tile_size + (relative_square_size + 1)*mid_length/relative_square_size)), (0, 255, 255))
                                if local_location_tracker and real_col > -1 and real_row > -1 and real_col < len(local_location_tracker) and real_row < len(local_location_tracker[real_col]) and local_location_tracker[real_col][real_row]:
                                    # ImageDraw.Draw(screenshot).rectangle(((col * tile_size + (relative_square_size - 1)*mid_length/relative_square_size, row * tile_size + (relative_square_size - 1)*mid_length/relative_square_size), (col * tile_size + (relative_square_size + 1)*mid_length/relative_square_size, row * tile_size + (relative_square_size + 1)*mid_length/relative_square_size)), (0, 0, 255))
                                    tile_label += "\n" + "EXPLORED"
                                elif (location, (real_col, real_row)) in past_locations:
                                    # ImageDraw.Draw(screenshot).rectangle(((col * tile_size + (relative_square_size - 1)*mid_length/relative_square_size, row * tile_size + (relative_square_size - 1)*mid_length/relative_square_size), (col * tile_size + (relative_square_size + 1)*mid_length/relative_square_size, row * tile_size + (relative_square_size + 1)*mid_length/relative_square_size)), (0, 255, 0))         
                                    tile_label += "\n" + "RECENTLY\nVISITED"
                                else:
                                    tile_label += "\n" + "CHECK\nHERE"
                        else:
                            # ImageDraw.Draw(screenshot).rectangle(((col * tile_size + (relative_square_size - 1)*mid_length/relative_square_size, row * tile_size + (relative_square_size - 1)*mid_length/relative_square_size), (col * tile_size + (relative_square_size + 1)*mid_length/relative_square_size, row * tile_size + (relative_square_size + 1)*mid_length/relative_square_size)), (255, 0, 255))
                            tile_label += "\n" + "NPC/OBJECT"
                        font_size = 8 # Default font size
                        # if MODEL == "GEMINI": # Removed model-specific font size
                        #     font_size = 12
                        ImageDraw.Draw(screenshot).text((col * tile_size + mid_length/2, row * tile_size + mid_length/2), tile_label, (255, 0, 0), font_size=font_size)
        screenshot.save("test.png")  # expensive, remove later

        # Convert to base64
        buffered = io.BytesIO()
        screenshot.save(buffered, format="PNG")
        return base64.standard_b64encode(buffered.getvalue()).decode()

    # Important for maintaining state between runs.
    # I am aware that this has grown into a list of like 17 pickle dumps. It got out of hand.
    def save_location_archive(self, pkl_path: str) -> None:
        # TODO: I should really just make this a clear state variable.
        with open(pkl_path, 'wb') as fw:
            pickle.dump(self.label_archive, fw)
            pickle.dump(self.location_history, fw)
            pickle.dump(self.message_history, fw)
            pickle.dump(self.location_tracker, fw)
            pickle.dump(self.location_tracker_activated, fw)
            pickle.dump(self.steps_since_checkpoint, fw)
            pickle.dump(self.steps_since_label_reset, fw)
            pickle.dump(self.last_location, fw)
            pickle.dump(self.map_tool_map, fw)
            pickle.dump(self.fully_mapped_locations, fw)
            # pickle.dump(self.openai_message_history, fw) # Removed from saving
            pickle.dump(self.full_collision_map, fw)
            pickle.dump(self.absolute_step_count, fw)
            pickle.dump(self.all_visited_locations, fw)
            pickle.dump(self.location_milestones, fw)
            pickle.dump(self.last_coords, fw)
            pickle.dump(self.checkpoints, fw)
            pickle.dump(self.detailed_navigator_mode, fw)
            # pickle.dump(self.navigator_message_history, fw) # Removed
            # pickle.dump(self.openai_navigator_message_history, fw) # Removed
            pickle.dump(self.steps_since_location_shift, fw)
            pickle.dump(self.no_navigate_here, fw)
            pickle.dump(self.navigation_location, fw)

    def load_location_archive(self, pkl_path: str) -> None:
        try:
            with open(pkl_path, 'rb') as fr:
                self.label_archive = pickle.load(fr)
                self.location_history = pickle.load(fr)
                self.message_history = pickle.load(fr)
                try:  # temporary legacy: older pickles don't have this
                    self.location_tracker = pickle.load(fr)
                    self.location_tracker_activated = pickle.load(fr)
                    self.steps_since_checkpoint = pickle.load(fr)
                    self.steps_since_label_reset = pickle.load(fr)
                    self.last_location = pickle.load(fr)
                    self.map_tool_map = pickle.load(fr)
                    self.fully_mapped_locations = pickle.load(fr)
                    try: # Safely try to load and discard openai_message_history for old pickles
                        pickle.load(fr) # This was self.openai_message_history in old pickles
                    except (EOFError, KeyError, AttributeError, pickle.UnpicklingError): # Broader catch
                        logger.info("Attempted to load openai_message_history from old pickle, but it was missing or malformed. Skipping.")
                        pass 
                    self.full_collision_map = pickle.load(fr)
                    self.absolute_step_count = pickle.load(fr)
                    self.all_visited_locations = pickle.load(fr)
                    self.location_milestones = pickle.load(fr)
                    self.last_coords = pickle.load(fr)
                    self.checkpoints = pickle.load(fr)
                    self.detailed_navigator_mode = pickle.load(fr) # Keep loading for now, though functionality is reduced
                    try: # Safely try to load and discard navigator_message_history
                        pickle.load(fr)
                    except (EOFError, KeyError, AttributeError, pickle.UnpicklingError):
                        logger.info("Attempted to load navigator_message_history from old pickle, but it was missing or malformed. Skipping.")
                        pass
                    try: # Safely try to load and discard openai_navigator_message_history
                        pickle.load(fr)
                    except (EOFError, KeyError, AttributeError, pickle.UnpicklingError):
                        logger.info("Attempted to load openai_navigator_message_history from old pickle, but it was missing or malformed. Skipping.")
                        pass
                    self.steps_since_location_shift = pickle.load(fr)
                    self.no_navigate_here = pickle.load(fr)
                    self.navigation_location = pickle.load(fr)
                except Exception:
                    pass
        except FileNotFoundError:
            logger.warn("No Location archive! Making new one...")
        if self.message_history:
            last_message = self.message_history[-1]
            # OpenAI format: assistant message with tool_calls is not a list in 'content'
            # It has 'tool_calls' attribute directly on the message object.
            # User message with tool results (role: tool) has 'content' as string.
            if last_message["role"] == "assistant" and hasattr(last_message, "tool_calls") and last_message.tool_calls is not None:
                # This case is generally fine, it means the assistant requested tools.
                pass
            elif last_message["role"] == "user" and isinstance(last_message.get("content"), list):
                 # This was an old format check, if content is a list and contains 'tool_use', it's likely old Claude format.
                 # For OpenAI, user message content is typically a string or list of dicts (text/image_url).
                 # Popping might be too aggressive if it's a valid multi-part user message.
                 # For now, let's assume history is correctly formatted by other parts of the code.
                 pass
        # Ensure message_history starts with a system prompt if it's empty or doesn't have one.
        if not self.message_history or self.message_history[0].get("role") != "system":
            system_prompt = SYSTEM_PROMPT if SYSTEM_PROMPT else "You are a helpful AI assistant."
            self.message_history.insert(0, {"role": "system", "content": system_prompt})


    # Save tokens...
    @staticmethod
    def strip_text_map_and_images_from_history(message_history: list[dict[str, Any]]) -> None:
        # We go through everything that's not the most recent tool_result/message and clip out images and
        # text_based maps to save tokens.
        # OpenAI format uses a list of content blocks for user messages, or string content for assistant messages.
        # Tool calls are also specific.
        for message_index in range(len(message_history) - 2): # Iterate up to the second to last message
            message = message_history[message_index]
            if message["role"] == "user":
                if isinstance(message["content"], list):
                    new_content_list = []
                    for content_item in message["content"]:
                        if content_item["type"] == "image_url":
                            # Replace image with a placeholder text
                            new_content_list.append({"type": "text", "text": "Screenshot omitted in history for brevity"})
                        elif content_item["type"] == "text":
                            text = content_item["text"]
                            try:
                                # Remove everything between [TEXT_MAP] tags
                                first, second_half = text.split("[TEXT_MAP]", 1)
                                _, third = second_half.split("[/TEXT_MAP]", 1)
                                new_content_list.append({"type": "text", "text": first + "TEXT MAP omitted to save redundancy" + third})
                            except ValueError: # Handles cases where tags are not found
                                new_content_list.append(content_item) # Keep original text if tags not found
                        else:
                            new_content_list.append(content_item) # Keep other types of content
                    message["content"] = new_content_list
            # Assistant messages might contain tool_calls, which are generally fine token-wise.
            # If assistant messages also directly embed large text or images (not typical for tool responses),
            # similar stripping logic would be needed here.
            # Tool (function) responses (role: 'tool') are usually JSON strings and should be managed if they become too large.
            # For now, focusing on user messages with images/text_maps.

    def update_and_get_full_collision_map(self, location, coords):
        collision_map = self.emulator.pyboy.game_wrapper.game_area_collision()
        downsampled_terrain = self.emulator._downsample_array(collision_map)
        local_location_tracker = self.location_tracker.get(location, [])
        # slightly more efficient than setdefault
        this_map = self.full_collision_map.get(location)
        if this_map is None:
            self.full_collision_map[location] = LocationCollisionMap(downsampled_terrain, self.emulator.get_sprites(), coords)
            return self.full_collision_map[location].to_ascii(local_location_tracker)
        else:
            this_map.update_map(downsampled_terrain, self.emulator.get_sprites(), coords)
            return this_map.to_ascii(local_location_tracker)
        
    def get_all_location_labels(self, location: str) -> list[tuple[tuple[int, int], str]]:
        all_labels: list[tuple[tuple[int, int], str]] = []
        this_location = self.label_archive.get(location)
        if this_location is None:
            # this sucks man
            for key, value in self.label_archive.items():
                if location.lower() == key.lower():
                    this_location = value
                    break
        if this_location is not None and this_location:
            max_row = max(this_location.keys())
            for nearby_row in range(max_row + 1):
                this_row = this_location.get(nearby_row)
                if this_row is not None:
                    max_col = max(this_row.keys())
                    for nearby_col in range(max_col + 1):
                        this_col = this_row.get(nearby_col)
                        if this_col is not None:
                            all_labels.append(((nearby_col, nearby_row), this_col))  # Note that we only care about our current location
        return all_labels
    
    def press_buttons(self, buttons: list[str], wait: bool, tool_id: str) -> dict[str, Any]:
        self.text_display.add_message(f"[Buttons] Pressing: {buttons} (wait={wait})")
        
        result, last_coords = self.emulator.press_buttons(buttons, wait)
        
        self.last_coords = last_coords
        
        # Get game state from memory after the action
        memory_info, location, coords = self.emulator.get_state_from_memory()
        # Log the memory state after the tool call
        logger.info(f"[Memory State after action]")
        logger.info(memory_info)
        
        collision_map = self.emulator.get_collision_map()
        if collision_map:
            logger.info(f"[Collision Map after action]\n{collision_map}")

        # TODO: Maybe python has good queues for this, but queue is not iterable for display
        self.location_history.insert(0, (location, coords))
        if len(self.location_history) > self.location_history_length:
            self.location_history.pop()
        if self.location_tracker_activated:
            cols = self.location_tracker.setdefault(location, [])
            # This is leaning hard on Python list append optimization... Maybe there are better structures?
            # col first
            if coords[0] > len(cols) - 1:
                if len(cols) == 0:
                    cols.extend(list() for _ in range(0, coords[0] + 1))  # Note that you can't do []*coords[0], because then the same list goes into each entry
                else:
                    cols.extend([False for _ in range(0, len(cols[0]))] for _ in range(0, coords[0] + 1))
            if coords[1] > len(cols[0]) - 1:
                # this is awkward
                for col in cols:
                    # This is actually too much (it would be coords[1] - len(col) + 1) but the overallocation is probably a good idea.
                    col.extend(False for _ in range(0, coords[1] + 1))
            cols[coords[0]][coords[1]] = True

        all_labels = self.get_all_location_labels(location)


        # Return tool result as a dictionary
        # Simplified: detailed_navigator_mode removed, so always return the full context.
        # 'result' here refers to the button press outcome from self.emulator.press_buttons.
        button_press_screen_text = result 

        screenshot = self.emulator.get_screenshot()
        screenshot_b64 = self.get_screenshot_base64(screenshot, upscale=4, add_coords=True, player_coords=coords, location=location)
        last_checkpoints = '\n'.join(self.checkpoints[-10:])
        content = [
                {"type": "text", "text": f"Pressed buttons: {', '.join(buttons)}. Screen text after press: {button_press_screen_text}"},
                {"type": "text", "text": "\nHere is a screenshot of the screen after your button presses:"},
                {
                    "type": "image_url", 
                    "image_url": {"url": f"data:image/png;base64,{screenshot_b64}"}
                },
                {"type": "text", "text": f"\nGame state information from memory after your action:\n{memory_info}"},
                {"type": "text", "text": f"\nLabeled nearby locations: {','.join(f'{label_coords}: {label}' for label_coords, label in all_labels)}"},
                {"type": "text", "text": f"Here are up to your last {str(self.location_history_length)} locations between commands (most recent first), to help you remember where you've been:/n{'/n'.join(f'{x[0]}, {x[1]}' for x in self.location_history)}"},
                {"type": "text", "text": f"Here are your last 10 checkpoints:\n{last_checkpoints}"},
                {"type": "text", "text": f"You have been in this location for {self.steps_since_location_shift} steps"}
            ]
        # The previous check for self.emulator.get_in_combat() and adding a note about detailed_navigator_mode is removed.
        if not self.emulator.get_in_combat() and self.use_full_collision_map:
            content.append({"type": "text", "text": "Here is a map of this RAM location compiled so far:\n\n[TEXT_MAP]" + self.update_and_get_full_collision_map(location, coords) + "\n\n[/TEXT_MAP]"})
        return {
            "type": "tool_result",
            "tool_use_id": tool_id,
            "content": content,
        }
                "content": [
                    {"type": "text", "text": (
                        f"Pressed buttons: {', '.join(buttons)}"
                    )}
                ],
            }
        else:
            # Get a fresh screenshot after executing the buttons
            if self.detailed_navigator_mode and not self.emulator.get_in_combat():
                # In navigator mode it gets confused if the screenshot/text_based isn't in the user prompt, so we trim it to save tokens.
                # TODO: That may not actually be true; there was another coding error. But this is already done so...
                last_checkpoints = '\n'.join(self.checkpoints[-10:])
                content = [
                        {"type": "text", "text": f"Navigation result: {result}"},
                        {"type": "text", "text": f"\nGame state information from memory after your action:\n{memory_info}"},
                        {"type": "text", "text": f"\nLabeled nearby locations: {','.join(f'{coords}: {label}' for coords, label in all_labels)}"},
                        {"type": "text", "text": f"Here are up to your last {str(self.location_history_length)} locations between commands (most recent first), to help you remember where you've been:/n{'/n'.join(f'{x[0]}, {x[1]}' for x in self.location_history)}"},
                        {"type": "text", "text": f"Here are your last 10 checkpoints:\n{last_checkpoints}"},
                        {"type": "text", "text": f"You have been in this location for {self.steps_since_location_shift} steps"}
                    ]
                return {
                    "type": "tool_result",
                    "tool_use_id": tool_id,
                    "content": content,
                }
            else:
                screenshot = self.emulator.get_screenshot()
                screenshot_b64 = self.get_screenshot_base64(screenshot, upscale=4, add_coords=True, player_coords=coords, location=location)
                last_checkpoints = '\n'.join(self.checkpoints[-10:])
                content = [
                        {"type": "text", "text": f"Pressed buttons: {', '.join(buttons)}"},
                        {"type": "text", "text": "\nHere is a screenshot of the screen after your button presses:"},
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": screenshot_b64,
                            },
                        },
                        {"type": "text", "text": f"\nGame state information from memory after your action:\n{memory_info}"},
                        {"type": "text", "text": f"\nLabeled nearby locations: {','.join(f'{label_coords}: {label}' for label_coords, label in all_labels)}"},
                        {"type": "text", "text": f"Here are up to your last {str(self.location_history_length)} locations between commands (most recent first), to help you remember where you've been:/n{'/n'.join(f'{x[0]}, {x[1]}' for x in self.location_history)}"},
                        {"type": "text", "text": f"Here are your last 10 checkpoints:\n{last_checkpoints}"},
                        {"type": "text", "text": f"You have been in this location for {self.steps_since_location_shift} steps"}
                    ]
                if self.emulator.get_in_combat():  # Only possible if navigator mode has been running.
                    content.append({"type": "text", "text": "NOTE: A Navigator version of Claude has been handling overworld movement for you, so your location may have shifted substantially. Please handle this battle for now."})
                if not self.emulator.get_in_combat() and self.use_full_collision_map:
                    content.append({"type": "text", "text": "Here is a map of this RAM location compiled so far:\n\n[TEXT_MAP]" + self.update_and_get_full_collision_map(location, coords) + "\n\n[/TEXT_MAP]"})
                return {
                    "type": "tool_result",
                    "tool_use_id": tool_id,
                    "content": content,
                }


    # TODO: An obvious refactor would be to move some of these into their own functions.
    def process_tool_call(self, tool_call_obj: Any) -> dict[str, Any]:
        """
        Process a single tool call.
        Expects tool_call_obj to be an OpenAI ChatCompletionMessageToolCall object.
        """
        tool_name = tool_call_obj.function.name
        tool_id = tool_call_obj.id
        try:
            tool_input = json.loads(tool_call_obj.function.arguments)
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding tool arguments for {tool_name} (ID: {tool_id}): {e}")
            return {
                "type": "tool_result",
                "tool_use_id": tool_id,
                "content": [{"type": "text", "text": f"Error: Invalid arguments for tool {tool_name}. Arguments must be valid JSON."}],
            }

        logger.info(f"Processing tool call: {tool_name} (ID: {tool_id}) with input: {tool_input}")
        
        # Check for 'explanation_of_action' if it's a convention we want to maintain
        if 'explanation_of_action' in tool_input:
            self.text_display.add_message(f"[Text from tool input] {tool_input['explanation_of_action']}")

        if tool_name == "press_buttons":
            buttons = tool_input["buttons"]
            wait = tool_input.get("wait", True)
            return self.press_buttons(buttons, wait, tool_id)
        elif tool_name == "navigate_to":
            row = tool_input["row"]
            col = tool_input["col"]
            memory_info, location, coords = self.emulator.get_state_from_memory()
            self.text_display.add_message(f"[Navigation] Navigating to: ({col}, {row})")  # 8, 3 -> 6, 4 is 2, 5
            
            # The navigator goes to location on screen, with 0,0 at the top left.
            local_col = col - coords[0] + 4
            local_row = row - coords[1] + 4

            status, path = self.emulator.find_path(local_row, local_col)
            last_coords = coords
            next_coords = coords
            if path:
                for direction in path:
                    self.emulator.press_buttons([direction], True, wait_for_finish=False)
                    cur_coords = self.emulator.get_coordinates()
                    if cur_coords != next_coords:
                        last_coords = next_coords
                        next_coords = cur_coords
                self.last_coords = last_coords
                result = f"Navigation successful: followed path with {len(path)} steps"
            else:
                result = f"Navigation failed: {status}"
            
            # Get game state from memory after the action
            memory_info, location, coords = self.emulator.get_state_from_memory()

            # Get a fresh screenshot after executing the buttons
            screenshot = self.emulator.get_screenshot()
            screenshot_b64 = self.get_screenshot_base64(screenshot, upscale=4, add_coords=True, player_coords=coords, location=location)
            
            # Log the memory state after the tool call
            logger.info(f"[Memory State after action]")
            logger.info(memory_info)
            
            collision_map = self.emulator.get_collision_map()
            if collision_map:
                logger.info(f"[Collision Map after action]\n{collision_map}")

            # TODO: Maybe python has good queues for this, but queue is not iterable for display
            self.location_history.insert(0, (location, coords))
            if len(self.location_history) > self.location_history_length:
                self.location_history.pop()
            if self.location_tracker_activated:
                if coords[0] >= 0 or coords[1] >= 0:
                    # Covers edge cases, principally when moving between areas.
                    cols = self.location_tracker.setdefault(location, [])
                    # This is leaning hard on Python list append optimization... Maybe there are better structures?
                    # col first
                    if coords[0] > len(cols) - 1:
                        if len(cols) == 0:
                            cols.extend(list() for _ in range(0, coords[0] + 1))  # Note that you can't do []*coords[0], because then the same list goes into each entry
                        else:
                            cols.extend([False for _ in range(0, len(cols[0]))] for _ in range(0, coords[0]))
                    if coords[1] > len(cols[0]) - 1:
                        # this is awkward
                        for col in cols:
                            # This is actually too much (it would be coords[1] - len(col) + 1) but the overallocation is probably a good idea.
                            col.extend(False for _ in range(0, coords[1] + 1))
                    cols[coords[0]][coords[1]] = True

            # TODO: eventually do this more reasonably. For now we do this extraordinarily dumb approach.
            all_labels = self.get_all_location_labels(location)

            # Return tool result as a dictionary
            # Simplified: detailed_navigator_mode removed.
            screenshot = self.emulator.get_screenshot()
            screenshot_b64 = self.get_screenshot_base64(screenshot, upscale=4, add_coords=True, player_coords=coords, location=location)
            last_checkpoints = '\n'.join(self.checkpoints[-10:])
            # Standardized content structure for navigate_to tool result
                    content = [
                            {"type": "text", "text": f"Navigation result: {result}"},
                            {"type": "text", "text": f"\nGame state information from memory after your action:\n{memory_info}"},
                            {"type": "text", "text": f"\nLabeled nearby locations: {','.join(f'{coords}: {label}' for coords, label in all_labels)}"},
                            {"type": "text", "text": f"Here are up to your last {str(self.location_history_length)} locations between commands (most recent first), to help you remember where you've been:/n{'/n'.join(f'{x[0]}, {x[1]}' for x in self.location_history)}"},
                            {"type": "text", "text": f"Here are your last 10 checkpoints:\n{last_checkpoints}"},
                            {"type": "text", "text": f"You have been in this location for {self.steps_since_location_shift} steps"}
                        ]
                    return {
                        "type": "tool_result",
                        "tool_use_id": tool_id,
                        "content": content,
                    }
                else:
                    screenshot = self.emulator.get_screenshot()
                    screenshot_b64 = self.get_screenshot_base64(screenshot, upscale=4, add_coords=True, player_coords=coords, location=location)
                    last_checkpoints = '\n'.join(self.checkpoints[-10:])
                    content = [
                            {"type": "text", "text": f"Navigation result: {result}"},
                            {"type": "text", "text": "\nHere is a screenshot of the screen after navigation:"},
                            {
                                "type": "image_url", 
                                "image_url": {"url": f"data:image/png;base64,{screenshot_b64}"}
                            },
                            {"type": "text", "text": f"\nGame state information from memory after your action:\n{memory_info}"},
                            {"type": "text", "text": f"\nLabeled nearby locations: {','.join(f'{coords}: {label}' for coords, label in all_labels)}"},
                            {"type": "text", "text": f"Here are up to your last {str(self.location_history_length)} locations between commands (most recent first), to help you remember where you've been:/n{'/n'.join(f'{x[0]}, {x[1]}' for x in self.location_history)}"},
                            {"type": "text", "text": f"Here are your last 10 checkpoints:\n{last_checkpoints}"},
                            {"type": "text", "text": f"You have been in this location for {self.steps_since_location_shift} steps"}
                        ]
                    if not self.emulator.get_in_combat() and self.use_full_collision_map:
                        content.append({"type": "text", "text": "Here is an text_based map of this RAM location compiled so far:\n\n" + self.update_and_get_full_collision_map(location, coords)})
                    # Removed combat check related to detailed_navigator_mode.
                    return {
                        "type": "tool_result",
                        "tool_use_id": tool_id,
                        "content": content,
                    }
        elif tool_name == "navigate_to_offscreen_coordinate":
            row = tool_input["row"]
            col = tool_input["col"]
            memory_info, location, coords = self.emulator.get_state_from_memory()
            full_map = self.update_and_get_full_collision_map(location, coords)
            
            final_distance = self.full_collision_map[location].distances.get((col, row))
            if final_distance is None:
                 return {
                    "type": "tool_result",
                    "tool_use_id": tool_id,
                    "content": [
                        {"type": "text", "text": f"Invalid coordinates; Navigation too far or not possible."}
                    ],
                }

            if DIRECT_NAVIGATION:
                self.text_display.add_message(f"Navigating with existing map...")
                buttons = self.full_collision_map[location].generate_buttons_to_coord(col, row)
                wait = True
            else:
                query = f"""Please take a look at the attached text_based map.

    Please consider in detail how the player character (labeled PP) can reach the coordinate ({col},{row}). Keep the following in mind:

#### SPECIAL NAVIGATION INSTRUCTIONS WHEN TRYING TO REACH A LOCATION #####
Pay attention to the following procedure when trying to reach a specific location (if you know the coordinates).
1. Inspect the text_based map
2. Find where your destination is on the map using the coordinate system (column, row).
3. Trace a path from there back to the player character (PP) following the StepsToReach numbers on the map, in descending order.
    3a. So if your destination is StepsToReach 20, then it is necessary to go through StepsToReach 19, StepsToReach 18...descending all the way to 1 and then PP.
4. Navigate via the REVERSE of this path.
###########################################



    Think through your movement like this

    To get to (col, row),
    1. I would move left from (col, row)
    2. To get there, I would move up from (col, row)
    etc.

    MAKE SURE TO PRINT OUT THE ENTIRE PATH IN TEXT. AND DOUBLE-CHECK WHETHER YOUR ARE PASSING THROUGH IMPASSABLE TILES.

    then use the provided "press_buttons" tool to send the necessary commands. Remember that it will be in reverse order.

    """ + full_map

                messages = [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": query,
                            }
                        ],
                    }
                ]
                # This section now uses self.sambanova_client for the sub-LLM call.
                # It will use the SAMBANOVA_STRATEGIST_MODEL_ID as it's a reasoning task.
                messages_for_nav_assist = [
                    {"role": "system", "content": "You are an expert navigator. Your task is to determine the sequence of button presses to reach a coordinate based on a map. Follow the path rules strictly."},
                    {"role": "user", "content": query} # query already contains the map and instructions
                ]
                try:
                    response = self.sambanova_client.chat.completions.create(
                        model=SAMBANOVA_STRATEGIST_MODEL_ID,
                        messages=messages_for_nav_assist,
                        tools=[DISTANT_NAVIGATOR_BUTTONS_OPENAI], # Ensure this is defined in OpenAI format
                        tool_choice="auto", # Let the model decide if it needs to call the tool
                        temperature=TEMPERATURE, # Could be lower for more deterministic navigation
                        max_tokens=MAX_TOKENS 
                    )
                    
                    nav_assistant_msg = response.choices[0].message
                    if nav_assistant_msg.tool_calls:
                        nav_tool_call = nav_assistant_msg.tool_calls[0] # Expecting one tool call for buttons
                        if nav_tool_call.function.name == "press_buttons":
                            nav_tool_input = json.loads(nav_tool_call.function.arguments)
                            buttons = nav_tool_input["buttons"]
                            wait = nav_tool_input.get("wait", True)
                            self.text_display.add_message(f"Distant Navigator Advice: Pressing {buttons}")
                        else:
                            logger.error(f"Distant navigator returned unexpected tool: {nav_tool_call.function.name}")
                            buttons = [] # Fallback
                            wait = True
                    else:
                        # Model decided not to use tool, might have text response
                        text_response = nav_assistant_msg.content if nav_assistant_msg.content else "No buttons provided by distant navigator."
                        self.text_display.add_message(f"Distant Navigator Text Response: {text_response}")
                        logger.warn(f"Distant navigator did not call press_buttons. Response: {text_response}")
                        buttons = [] # Fallback
                        wait = True
                except Exception as e:
                    logger.error(f"Error calling SambaNova for distant navigation assist: {e}")
                    buttons = [] # Fallback
                    wait = True
            return self.press_buttons(buttons, wait, tool_id) # tool_id is from the original tool_call
         
        elif tool_name == "bookmark_location_or_overwrite_label":
            location = tool_input["location"]
            row = tool_input["row"]
            col = tool_input["col"]
            label = tool_input["label"]
            self.text_display.add_message(f"Logging {location},  ({col}, {row}) as {label}")
            self.label_archive.setdefault(location.lower(), {}).setdefault(row, {})[col] = label
            return {
                "type": "tool_result",
                "tool_use_id": tool_id,
                "content": [
                    {"type": "text", "text": f"Location Labeled: {location}, ({col}, {row}) as {label}"}
                ],
            }
        elif tool_name == "mark_checkpoint":
            self.steps_since_checkpoint = 0
            self.steps_since_label_reset = 0
            self.location_tracker_activated = False
            achievement = tool_input["achievement"]
            self.checkpoints.append(achievement)
            self.text_display.add_message(f"Checkpoint marked: {achievement}")
            return {
                "type": "tool_result",
                "tool_use_id": tool_id,
                "content": [
                    {"type": "text", "text": f"Checkpoint set!"}
                ],
            }
        elif tool_name == "navigation_assistance":
            # Currently not used
            assist_str = self.navigation_assistance(tool_input["navigation_goal"])
            return {
                "type": "tool_result",
                "tool_use_id": tool_id,
                "content": [
                    {"type": "text", "text": assist_str}
                ],
            }
        elif tool_name == "detailed_navigator":
            self.detailed_navigator_mode = True
            self.navigation_location = self.emulator.get_location()
            self.navigator_message_history = [{"role": "user", "content": "Please begin navigating!"}]
            self.openai_navigator_message_history = [{"role": "user", "content": "Please begin navigating!"}]
            return {
                "type": "tool_result",
                "tool_use_id": tool_id,
                "content": [
                    {"type": "text", "text": "Navigator Mode Activated."}
                ],
            }
        else:
            logger.error(f"Unknown tool called: {tool_name}")
            return {
                "type": "tool_result",
                "tool_use_id": tool_id,
                "content": [
                    {"type": "text", "text": f"Error: Unknown tool '{tool_name}'"}
                ],
            }


    def run(self, num_steps=1, save_every=10, save_file_name: Optional[str] = None, _running_in_thread=False):
        """Main agent loop.

        Args:
            num_steps: Number of steps to run for
        """

        if self.pyboy_main_thread and not _running_in_thread:
            thread = threading.Thread(target=self.run, kwargs={"num_steps": num_steps, "save_every": save_every, "save_file_name": save_file_name, "_running_in_thread": True})
            thread.start()

            self.emulator.initialize(**self.emulator_init_kwargs) 

            return self._steps_completed

        logger.info(f"Starting agent loop for {num_steps} steps")

        if self.pyboy_main_thread:
            self.emulator.wait_for_pyboy()

            if self.load_state:
                logger.info(f"Loading saved state from {self.load_state}")
                self.emulator.load_state(self.load_state)

        # start emulator loop
        steps_completed = 0
        while self.running and steps_completed < num_steps:
            try:
                location = self.emulator.get_location()
                coords = self.emulator.get_coordinates()
                if location not in self.all_visited_locations:
                    self.text_display.add_message(f"New Location reached! {location} at {self.absolute_step_count}")
                    self.location_milestones.append((location, self.absolute_step_count))
                    self.all_visited_locations.add(location)
                self.last_coords = coords
                malformed = False # Will be set if tool call parsing fails.
                token_usage = 0
                
                self.strip_text_map_and_images_from_history(self.message_history)
                messages_for_api = copy.deepcopy(self.message_history)
                
                # Determine Task and Model
                current_task_requires_vision = not self.emulator.get_in_combat() and (self.steps_since_location_shift < 2 or self.steps_since_checkpoint < 3 or self.steps_since_label_reset < 3) # Heuristic
                model_id_to_use = SAMBANOVA_VISION_MODEL_ID if current_task_requires_vision else SAMBANOVA_TOOL_MODEL_ID
                
                # Construct user message content parts
                user_content_parts = []
                memory_info, current_location, current_coords = self.emulator.get_state_from_memory()
                all_labels_text = ', '.join(f'{cl}: {lab}' for cl, lab in self.get_all_location_labels(current_location))
                last_checkpoints_text = '\n'.join(self.checkpoints[-10:])
                
                game_state_text = (
                    f"Current Location: {current_location} at Coords: {current_coords}. In Combat: {self.emulator.get_in_combat()}\n"
                    f"Memory Info: {memory_info}\n"
                    f"Labeled Nearby Locations: {all_labels_text}\n"
                    f"Last 10 Checkpoints: {last_checkpoints_text}\n"
                    f"Steps since last location shift: {self.steps_since_location_shift}. Steps since checkpoint: {self.steps_since_checkpoint}."
                )
                user_content_parts.append({"type": "text", "text": game_state_text})

                if not self.emulator.get_in_combat():
                    map_text = self.update_and_get_full_collision_map(current_location, current_coords)
                    user_content_parts.append({"type": "text", "text": f"\n[TEXT_MAP]\n{map_text}\n[/TEXT_MAP]\n"})

                if model_id_to_use == SAMBANOVA_VISION_MODEL_ID:
                    screenshot = self.emulator.get_screenshot()
                    screenshot_b64 = self.get_screenshot_base64(screenshot, upscale=1, add_coords=True, player_coords=current_coords, location=current_location)
                    user_content_parts.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{screenshot_b64}"}
                    })
                    user_content_parts.append({"type": "text", "text": "\nAnalyze the current game screen and state. Your primary goal is to progress in the game. Decide the next best action or set of actions. Use tools if appropriate. Explain your reasoning before acting."})
                else: # SAMBANOVA_TOOL_MODEL_ID
                     user_content_parts.append({"type": "text", "text": "\nBased on the current game state, decide the next best action or set of actions to progress in the game. Use tools if appropriate. Explain your reasoning before acting."})
                
                # Add the fully constructed user message to messages_for_api
                messages_for_api.append({"role": "user", "content": user_content_parts})
                
                # Ensure the first message is a system prompt if self.message_history was empty or cleared.
                if not messages_for_api or messages_for_api[0].get("role") != "system":
                    system_prompt_content = SYSTEM_PROMPT if SYSTEM_PROMPT else "You are an AI agent playing a game. Your primary goal is to explore the game world and progress. Reason about the game state and decide on the best course of action. Use tools if appropriate. Explain your thought process."
                    messages_for_api.insert(0, {"role": "system", "content": system_prompt_content })

                logger.info(f"Calling Vision/Tool Model: {model_id_to_use}. Messages count: {len(messages_for_api)}")
                
                vision_tool_model_tool_calls = None
                vision_tool_model_response_text = ""
                strategist_acted_this_turn = False # Flag to skip general tool processing if strategist acts

                try:
                    response = self.sambanova_client.chat.completions.create(
                        model=model_id_to_use,
                        messages=messages_for_api,
                        tools=OPENAI_TOOLS if model_id_to_use == SAMBANOVA_TOOL_MODEL_ID else None, # Changed to OPENAI_TOOLS
                        tool_choice="auto" if model_id_to_use == SAMBANOVA_TOOL_MODEL_ID else None,
                        temperature=TEMPERATURE,
                        max_tokens=MAX_TOKENS 
                    )
                    token_usage = response.usage.total_tokens if response.usage else 0 # Accumulate token usage
                    logger.info(f"SambaNova Vision/Tool Model ({model_id_to_use}) response usage: {token_usage} tokens.")

                    response_message = response.choices[0].message
                    vision_tool_model_tool_calls = response_message.tool_calls
                    vision_tool_model_response_text = response_message.content if response_message.content else ""
                    
                    # Update history with the user message that was sent to the vision/tool model
                    self.message_history.append(messages_for_api[-1])
                    
                    # Prepare and add assistant's response (text and tool calls from vision/tool model) to history
                    assistant_message_for_history = {"role": "assistant"}
                    assistant_content_parts = []
                    if vision_tool_model_response_text:
                        assistant_content_parts.append({"type": "text", "text": vision_tool_model_response_text})
                        self.text_display.add_message(f"[Vision/Tool Model Text] {vision_tool_model_response_text}")
                    if vision_tool_model_tool_calls:
                        assistant_message_for_history["tool_calls"] = vision_tool_model_tool_calls
                        for tc in vision_tool_model_tool_calls:
                            self.text_display.add_message(f"[Vision/Tool Model Tool Call] Requesting: {tc.function.name} ID: {tc.id} Args: {tc.function.arguments}")
                    
                    if assistant_content_parts: # Add text content if it exists
                        assistant_message_for_history["content"] = assistant_content_parts
                    elif not vision_tool_model_tool_calls : # No text and no tool calls, ensure content is not None for OpenAI SDK
                         assistant_message_for_history["content"] = "" # Or None, if the SDK handles it. Empty string is safer.
                    
                    self.message_history.append(assistant_message_for_history)

                except Exception as e:
                    logger.error(f"Error calling SambaNova Vision/Tool Model ({model_id_to_use}): {e}", exc_info=True)
                    vision_tool_model_response_text = f"Error interacting with Vision/Tool LLM: {e}"
                    # Update history with user turn and error from assistant
                    self.message_history.append(messages_for_api[-1]) # User message
                    self.message_history.append({"role": "assistant", "content": vision_tool_model_response_text})


                # STRATEGIST MODEL CALL (if not in combat and vision/tool model didn't request immediate critical tools)
                # Condition to call strategist: Not in combat, AND ( (vision model gave some description AND no tools) OR (heuristic like few steps in loc) )
                # For now, let's simplify: call if not in combat and vision model provided some text.
                if not self.emulator.get_in_combat() and vision_tool_model_response_text and not vision_tool_model_tool_calls:
                    game_state_summary_strat = "\n".join(self.checkpoints[-10:]) if self.checkpoints else "No checkpoints recorded yet."
                    # current_vision_text_description is vision_tool_model_response_text
                    formatted_loc_hist_strat = "\n".join([f'{ln} @ {cd}' for ln, cd in self.location_history[:5]])
                    all_labels_strat = self.get_all_location_labels(current_location) # current_location from above
                    formatted_labels_strat = "\n".join([f'{lc}: {lt}' for lc, lt in all_labels_strat[:5]])
                    current_objective_strat = self.checkpoints[-1] if self.checkpoints else "Explore the area and make progress."

                    strategist_prompt_filled = SAMBANOVA_STRATEGIST_PROMPT.format(
                        game_state_summary=game_state_summary_strat,
                        screen_description=vision_tool_model_response_text, # Output from the vision model
                        player_coords=str(current_coords), # current_coords from above
                        location=current_location,
                        location_history=formatted_loc_hist_strat,
                        labeled_locations=formatted_labels_strat,
                        current_objective=current_objective_strat
                    )
                    
                    self.text_display.add_message("[Text] Asking strategist for a plan...")
                    # For strategist, we typically don't send the whole history, just the formatted prompt.
                    strategist_call_messages = [{"role": "user", "content": strategist_prompt_filled}]
                    # The SAMBANOVA_STRATEGIST_PROMPT itself can act as a system prompt if formatted correctly
                    # or add a system message if needed:
                    # strategist_call_messages.insert(0, {"role": "system", "content": "You are a game strategist..."})

                    try:
                        strategist_response = self.sambanova_client.chat.completions.create(
                            model=SAMBANOVA_STRATEGIST_MODEL_ID,
                            messages=strategist_call_messages,
                            temperature=TEMPERATURE, 
                            max_tokens=MAX_TOKENS, # Or a smaller value for button list
                        )
                        token_usage += strategist_response.usage.total_tokens if strategist_response.usage else 0
                        strategist_output_text = strategist_response.choices[0].message.content
                        # Add strategist interaction to history
                        self.message_history.append({"role": "user", "content": strategist_prompt_filled}) # User prompt to strategist
                        self.message_history.append({"role": "assistant", "content": strategist_output_text}) # Strategist raw response
                        self.text_display.add_message(f"[Text] Strategist response: {strategist_output_text}")

                        try:
                            planned_buttons = json.loads(strategist_output_text)
                            if isinstance(planned_buttons, list) and all(isinstance(b, str) for b in planned_buttons) and planned_buttons:
                                self.text_display.add_message(f"[Strategist Plan] Executing buttons: {planned_buttons}")
                                _, self.last_coords = self.emulator.press_buttons(planned_buttons, wait=True)
                                # Update game state after planned buttons
                                memory_info, current_location, current_coords = self.emulator.get_state_from_memory() # Refresh state
                                self.location_history.insert(0, (current_location, current_coords))
                                if len(self.location_history) > self.location_history_length: self.location_history.pop()
                                # Add observation of action to history for next turn context
                                self.message_history.append({
                                    "role": "user", # Or "observation" if we make that a role
                                    "content": f"Executed strategist plan: {planned_buttons}. New location: {current_location} @ {current_coords}. RAM: {memory_info}"
                                })
                                strategist_acted_this_turn = True 
                            else:
                                self.text_display.add_message("[Error] Strategist output was not a valid non-empty JSON list of strings.")
                        except json.JSONDecodeError:
                            self.text_display.add_message(f"[Error] Strategist output was not valid JSON: {strategist_output_text}")
                        except Exception as e:
                            self.text_display.add_message(f"[Error] Error executing strategist plan: {e}")
                    except Exception as e:
                        self.text_display.add_message(f"[Error] Strategist LLM call failed: {e}")
                        self.message_history.append({"role": "user", "content": strategist_prompt_filled}) # Log the prompt
                        self.message_history.append({"role": "assistant", "content": f"Error calling strategist: {e}"})
                
                # Process tool calls from the VISION/TOOL model (if strategist didn't act)
                if not strategist_acted_this_turn and vision_tool_model_tool_calls:
                    tool_responses_for_history = []
                    for tool_call_obj in vision_tool_model_tool_calls:
                        tool_result_from_processing = self.process_tool_call(tool_call_obj)
                        actual_content_for_history = json.dumps(tool_result_from_processing["content"])
                        tool_responses_for_history.append({
                            "role": "tool",
                            "tool_call_id": tool_call_obj.id,
                            "name": tool_call_obj.function.name,
                            "content": actual_content_for_history 
                        })
                    if tool_responses_for_history:
                        self.message_history.extend(tool_responses_for_history)
                
                elif not strategist_acted_this_turn and not vision_tool_model_tool_calls and not vision_tool_model_response_text:
                    # If no action from strategist, no tools from vision/tool model, and no text from vision/tool model
                    self.message_history.append(
                        {"role": "user", "content": [{"type": "text", "text": "No specific action was determined. Please assess the situation and continue playing."}]}
                    )

            except Exception as e: # Catch broader exceptions in the main loop processing (e.g. before API call)
                logger.error(f"Error in agent run loop step: {e}", exc_info=True)
                error_message_for_history = {"role": "user", "content": f"Encountered an error in the previous step: {e}. Attempting to recover."}
                if not self.message_history or self.message_history[-1]["role"] != "user":
                     self.message_history.append(error_message_for_history)
                elif isinstance(self.message_history[-1]["content"], list) and not any(part.get("text","").startswith("Encountered an error") for part in self.message_history[-1]["content"] if isinstance(part, dict)):
                     self.message_history.append(error_message_for_history)
                elif isinstance(self.message_history[-1]["content"], str) and not self.message_history[-1]["content"].startswith("Encountered an error"):
                     self.message_history.append(error_message_for_history)


            # Increment step counters
            steps_completed += 1
            self.absolute_step_count += 1
            self.steps_since_checkpoint += 1
            
            # Location change logic
            new_location_name = self.emulator.get_location() # Renamed for clarity
            if self.last_location != new_location_name:
                if self.last_coords is not None and not self.emulator.get_in_combat() and self.last_location is not None:
                    self.label_archive.setdefault(self.last_location, {}).setdefault(self.last_coords[1], {})[self.last_coords[0]] = f"Entrance to {new_location_name} (Approximate)"
                self.steps_since_location_shift = 0
                self.steps_since_label_reset = 0 # Reset on location change
                self.text_display.add_message(f"Location changed from {self.last_location} to {new_location_name}")
            else:
                self.steps_since_location_shift += 1
            self.last_location = new_location_name # Update last_location

            # Auto-activate location tracker
            if self.steps_since_checkpoint > 50 and not self.location_tracker_activated: # Increased threshold
                self.location_tracker_activated = True
                self.location_tracker = {} # Reset tracker
                self.text_display.add_message("Location tracker activated due to extended steps since checkpoint.")

            # Auto-clear labels (simplified)
            if self.steps_since_label_reset > 150: # Reduced threshold
                self.text_display.add_message("Attempting to clear non-approximate labels for current location.")
                self.steps_since_label_reset = 0
                # Ensure new_location_name is defined; it should be from earlier in the loop.
                # If new_location_name might not be defined, use self.last_location as a fallback.
                loc_to_clear_labels = new_location_name if 'new_location_name' in locals() and new_location_name else self.last_location
                if loc_to_clear_labels: # Ensure we have a location
                    current_loc_labels = self.label_archive.get(loc_to_clear_labels)
                    if current_loc_labels:
                        for r_key in list(current_loc_labels.keys()): # Iterate over keys for safe deletion
                            for c_key in list(current_loc_labels[r_key].keys()):
                                if "approximate" not in current_loc_labels[r_key][c_key].lower():
                                    del current_loc_labels[r_key][c_key]
                            if not current_loc_labels[r_key]: # If row becomes empty
                                del current_loc_labels[r_key]
            else: # Increment only if not reset
                self.steps_since_label_reset += 1
            
            logger.info(f"Completed step {steps_completed}/{num_steps}")
            self.text_display.add_message(f"Absolute step count: {self.absolute_step_count}")

            # Summarize history if needed
            # Check token_usage from the last successful API call
            if len(self.message_history) >= self.max_history or \
               (token_usage > 170000 and model_id_to_use != SAMBANOVA_VISION_MODEL_ID): 
                self.agentic_summary()

            # Save progress
            if save_file_name is not None and not steps_completed % save_every:
                self.emulator.save_state(save_file_name)
                self.save_location_archive(self.location_archive_file_name)
                with open("location_milestones.txt", "w") as fw:
                    fw.write(str(self.location_milestones))

        # Loop end handling
        if save_file_name is not None and steps_completed > 0 : # Ensure saving happens if loop ran
            logger.info("Saving state at end of run")
            self.emulator.save_state(save_file_name)
            self.save_location_archive(self.location_archive_file_name)
            with open("location_milestones.txt", "w") as fw:
                fw.write(str(self.location_milestones))

        if not self.running or self.pyboy_main_thread:
            self.emulator.stop()
        self._steps_completed = steps_completed
        return steps_completed

    def navigation_assistance(self, navigation_goal: str) -> str:
        logger.info(f"[Agent] Running Navigation Assist...")
        
        _, location, coords = self.emulator.get_state_from_memory()

        collision_map = self.update_and_get_full_collision_map(location, coords)

        this_location = self.label_archive.get(location)
        if this_location is None:
            # this sucks man
            for key, value in self.label_archive.items():
                if location.lower() == key.lower():
                    this_location = value
                    break

        labels = "No Labels yet."
        all_labels = []
        if this_location is not None:
            for row_ind, this_row in this_location.items():
                for col_ind, this_col in this_row.items():
                    all_labels.append(((col_ind, row_ind), this_col)) 

        if all_labels:                
            labels = ','.join(f'{label_coords}: {label}' for label_coords, label in all_labels)

        mapping_query = f"""Here is a map of the current location:

        Current location: {location}

        {collision_map}

        Remember, higher numbers in the first coordinate are to the RIGHT. Higher numbers in the second coordinate are DOWN.
        
        Here are some labels:

        {labels}

        Here is the current navigation goal:

        {navigation_goal}
        """
        # Use SAMBANOVA_STRATEGIST_MODEL_ID for navigation assistance as it's a text-based reasoning task
        full_text = self.prompt_text_reply(NAVIGATION_PROMPT, mapping_query, False, SAMBANOVA_STRATEGIST_MODEL_ID, False)
        self.text_display.add_message(f"Navigation Advice: {full_text}")
        return full_text
    
    def prompt_text_reply(self, system_instructions: str, user_prompt: str, include_history: bool, samba_model_id: str, include_screenshot_in_user_prompt: bool) -> str:
        """
        Generic method to get a text reply from a SambaNova model.
        Manages history inclusion and screenshot addition to the user prompt.
        """
        messages_for_call = []
        if system_instructions:
            messages_for_call.append({"role": "system", "content": system_instructions})
        
        if include_history:
            history_copy = copy.deepcopy(self.message_history)
            self.strip_text_map_and_images_from_history(history_copy) # Strip from the copy
             # Ensure system prompt from history_copy isn't duplicated if system_instructions is also provided
            if system_instructions and history_copy and history_copy[0]["role"] == "system":
                messages_for_call.extend(history_copy[1:]) # Skip system prompt from history
            else:
                messages_for_call.extend(history_copy)

        current_user_content_parts = [{"type": "text", "text": user_prompt}]

        if include_screenshot_in_user_prompt:
            _, location, coords = self.emulator.get_state_from_memory()
            screenshot = self.emulator.get_screenshot()
            screenshot_b64 = self.get_screenshot_base64(screenshot, upscale=1, add_coords=True, player_coords=coords, location=location) 
            current_user_content_parts.append({
                "type": "image_url", 
                "image_url": {"url": f"data:image/png;base64,{screenshot_b64}"}
            })
        
        messages_for_call.append({"role": "user", "content": current_user_content_parts})
        
        # Safety check: if no system prompt is present at the start, add a generic one.
        if not messages_for_call or messages_for_call[0].get("role") != "system":
             messages_for_call.insert(0, {"role": "system", "content": "You are a helpful AI assistant."})

        try:
            response = self.sambanova_client.chat.completions.create(
                model=samba_model_id, 
                messages=messages_for_call,
                temperature=TEMPERATURE, 
                max_tokens=MAX_TOKENS # Consider if MAX_TOKENS is always appropriate or if it should be context-dependent
            )
            reply_text = response.choices[0].message.content if response.choices[0].message.content else ""
            token_usage = response.usage.total_tokens if response.usage else 0
            logger.info(f"SambaNova (prompt_text_reply for {samba_model_id}) usage: {token_usage} tokens. Reply length: {len(reply_text)}")
        except Exception as e:
            logger.error(f"Error calling SambaNova in prompt_text_reply ({samba_model_id}): {e}", exc_info=True)
            reply_text = f"Error generating text reply: {e}"
        
        return reply_text
    
    def agentic_summary(self):
        self.text_display.add_message(f"[Agent] Generating Facts Analysis, standby...")

        memory_info, location, coords = self.emulator.get_state_from_memory()
        try:
            # Attempt to get previous summary text. It might be in a list of content parts.
            # The first message is system, second is user summary.
            if len(self.message_history) > 1 and isinstance(self.message_history[1]["content"], list):
                previous_summary_parts = [part["text"] for part in self.message_history[1]["content"] if part["type"] == "text" and "CONVERSATION HISTORY SUMMARY" in part["text"]]
                previous_summary = " ".join(previous_summary_parts).replace("CONVERSATION HISTORY SUMMARY (representing previous messages):","").strip()

            elif len(self.message_history) > 1 and isinstance(self.message_history[1]["content"], str) : # older format or direct string
                previous_summary = self.message_history[1]["content"]
            else: # Default if no suitable history found
                 previous_summary = "Start of the Game or no previous summary found!"

        except (TypeError, IndexError, KeyError) as e: 
            logger.warn(f"Could not extract previous summary for agentic_summary: {e}")
            previous_summary = "Error extracting previous summary!"
        if not previous_summary.strip() or previous_summary == "You are a helpful assistant.": # Check if it's just the system prompt
            previous_summary = "No previous game summary available."
        
        last_checkpoints = '\n'.join(self.checkpoints[-10:])
        all_labels_text = ', '.join(f'{cl}: {lab}' for cl, lab in self.get_all_location_labels(location))


        if not self.emulator.get_in_combat():
            collision_map = self.update_and_get_full_collision_map(location, coords)
        else:
            collision_map = "In combat, map not generated." if location not in self.full_collision_map else self.full_collision_map[location].to_ascii(self.location_tracker.get(location, []))

        prompt = f"""
Current Game State for Summarization:
RAM Information: {memory_info}
Location: {location}, Coords: {coords}, In Combat: {self.emulator.get_in_combat()}
Steps Since last Location Shift: {self.steps_since_location_shift}
TEXT_MAP: 
{collision_map}
Last 10 Checkpoints: {last_checkpoints}
Labeled nearby locations: {all_labels_text}
Previous game summary: {previous_summary}
A game screenshot is attached.

Your task is to deduce the current overall state of the game, progress, and immediate objectives based on the provided information and conversation history.
This is for an agentic summary to condense the history. Be concise but comprehensive.
Your job is NOT to play the game now, but to summarize the state FOR the game-playing agent.
"""

        # Get the FACTS - Use Strategist Model for this reasoning task, include history and current screenshot
        response1 = self.prompt_text_reply(META_KNOWLEDGE_PROMPT, prompt, True, SAMBANOVA_STRATEGIST_MODEL_ID, True) 
        logger.info(f"Facts Stage 1 (Meta Knowledge): {response1}")
        # Clean Facts - Use Strategist Model, no history needed beyond response1, no screenshot
        response2 = self.prompt_text_reply(META_KNOWLEDGE_CLEANUP_PROMPT, response1, False, SAMBANOVA_STRATEGIST_MODEL_ID, False)
        logger.info(f"Facts Stage 2 (Cleanup): {response2}")
        # Summarize for real - Use Strategist Model, include history that led to summary, no screenshot
        response3 = self.prompt_text_reply(META_KNOWLEDGE_SUMMARIZER, response2, True, SAMBANOVA_STRATEGIST_MODEL_ID, False) 
        self.text_display.add_message(f"Final Summary: {response3}")
        with open("agentic_summary.txt", "w", encoding="utf-8") as fw:
            fw.write(f"Fact Stage 1:\n{response1}\n\nFact Stage 2:\n{response2}\n\nFinal Summary:\n{response3}")

        # Get a fresh screenshot for the new history start
        screenshot = self.emulator.get_screenshot()
        screenshot_b64 = self.get_screenshot_base64(screenshot, upscale=1, add_coords=True, player_coords=coords, location=location) # Upscale 1 for history

        # Replace message history with just the summary, in OpenAI SDK format
        system_prompt_content = SYSTEM_PROMPT if SYSTEM_PROMPT else "You are a helpful AI assistant playing a game."
        self.message_history = [
            {"role": "system", "content": system_prompt_content},
            {
                "role": "user", 
                "content": [
                    {
                        "type": "text",
                        "text": f"CONVERSATION HISTORY SUMMARY (representing previous game turns): {response3}" # Removed max_history reference as it's not strictly tied to summarization trigger always
                    },
                    {
                        "type": "text",
                        "text": "\nCurrent game screenshot for reference after summary:"
                    },
                    {
                        "type": "image_url", 
                        "image_url": {"url": f"data:image/png;base64,{screenshot_b64}"}
                    },
                    {
                        "type": "text",
                        "text": "You were just asked to summarize your playthrough. The summary and current screen are above. Continue playing."
                    },
                ]
            }
        ]
        
    def stop(self):
        """Stop the agent."""
        self.running = False
        self.emulator.stop()


if __name__ == "__main__":
    # Get the ROM path relative to this file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    rom_path = os.path.join(os.path.dirname(current_dir), "pokemon.gb")

    # Create and run agent
    agent = SimpleAgent(rom_path)

    try:
        steps_completed = agent.run(num_steps=10)
        logger.info(f"Agent completed {steps_completed} steps")
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt, stopping")
    finally:
        agent.stop()

    
                        location = self.emulator.get_location()
                        exploration_command = self.mapping_tool()
                        local_map = self.map_tool_map.get(
                            location
                        )
                        if local_map is not None:
                            tool_results.append(
                                {"type": "text", "text": f"HERE IS AN ASCII MAP OF THIS LOCATION BASED ON YOUR EXPLORATION: {local_map}"})
                        if exploration_command is not None:
                            tool_results.append(
                                {"type": "text", "text": f"EXPLORATION COMMAND: {exploration_command}"})
                            logger.info(f"Mapping tool forcing redirection: {exploration_command}")
                        with open("mapping_log.txt", "w", encoding="utf-8") as fw:
                            fw.write(local_map)"""
                    if malformed:
                        tool_results.append({"type": "text", "text": f"WARNING: MALFORMED TOOL CALL. Call using the function call, not in the text."})

                    # Add tool results to message history
                    messages_here.append(
                        {"role": "user", "content": tool_results}  # type: ignore
                    )
                    
                    # Check if we need to summarize the history
                    if self.detailed_navigator_mode and not self.emulator.get_in_combat():
                        # No agentic in navigator mode
                        if len(self.navigator_message_history) >= self.max_history:
                            # Truncation is not as straightforward as I'd like, because of the potential to break tool calls
                            self.navigator_message_history = self.navigator_message_history[self.max_history - len(self.navigator_message_history):]
                            remove = False
                            for k, message in enumerate(self.navigator_message_history):
                                # If we see a tool_call we're clear, because we don't have async tool calls.
                                for content in message['content']:
                                    if isinstance(message['content'], str):
                                        continue
                                    if content["type"] == "tool_call":
                                        break
                                    elif content["type"] == "tool_result":  # uh-oh, we're going to have to remove everything up to this.
                                        remove = True
                                        break
                            if remove:
                                self.navigator_message_history = self.navigator_message_history[k + 1:]
                        if len(self.openai_navigator_message_history) >= self.max_history: 
                            self.openai_navigator_message_history = self.openai_navigator_message_history[self.max_history - len(self.openai_navigator_message_history):]
                            # Let's scroll in, if we run into a tool_call_output, let's get rid of it since it's an orphaned tool_call.
                            
                            if self.openai_navigator_message_history and self.openai_navigator_message_history[0].get('type') == "function_call_output":
                                self.openai_navigator_message_history = self.openai_navigator_message_history[1:]

                    else:
                        # MODEL constant is removed. The condition for agentic_summary might need adjustment
                        # if token limits or history length considerations are different for SambaNova.
                        if len(self.message_history) >= self.max_history or token_usage > 170000: # Assuming similar token limits for now
                            self.agentic_summary()
                if not tool_calls:  # type: ignore
                    # Sometimes it just stalls out mysteriously or says some text.
                    messages_here.append(
                        {"role": "user", "content": [{"text": "Can you please continue playing the game?", "type": "text"}]}  # type: ignore
                    )

                steps_completed += 1
                self.absolute_step_count += 1
                self.steps_since_checkpoint += 1
                self.steps_since_label_reset += 1
                self.steps_since_location_shift += 1
                if self.steps_since_location_shift > 300 and not self.detailed_navigator_mode:  # Since Claude absolutely refuses to ask for help.
                    self.detailed_navigator_mode = True
                    self.navigation_location = location
                    self.navigator_message_history = [{"role": "user", "content": "Please begin navigating!"}]
                    self.openai_navigator_message_history = [{"role": "user", "content": "Please begin navigating!"}]
                if self.steps_since_checkpoint > 50 and not self.location_tracker_activated:
                    self.location_tracker_activated = True
                    self.location_tracker = {}
                _, location, _ = self.emulator.get_state_from_memory()
                if self.last_location != location:
                    if self.last_coords is not None and not self.emulator.get_in_combat() and self.last_location is not None:
                        self.label_archive.setdefault(self.last_location, {}).setdefault(self.last_coords[1], {})[self.last_coords[0]] = f"Entrance to {location} (Approximate)"
                    self.steps_since_location_shift = 0
                    self.steps_since_label_reset = 0
                    # The navigator turns OFF the moment the location changes.
                    if self.detailed_navigator_mode and location != self.no_navigate_here and location != self.navigation_location:
                        self.text_display.add_message("New Location reached; Navigator Mode Off")
                        self.message_history.append(
                            {"role": "user", "content": [{"text": "Note: Detailed Navigator Mode was just turned off since a new location was reached", "type": "text"}]}  # type: ignore
                        )
                        self.openai_message_history.append({"role": "user", "content": [{"text": "Note: Detailed Navigator Mode was just turned off since a new location was reached", "type": "text"}]})
                        # TODO: Consolidate navigator messages and clear
                    self.detailed_navigator_mode = False
                self.last_location = location
                # MODEL constant is removed. The threshold for steps_since_label_reset might need to be re-evaluated
                # or made consistent if it was model-dependent.
                if self.steps_since_label_reset > 200: # Using a single value for now, adjust if necessary
                    self.text_display.add_message("Clearing labels to clear potential bad labels...")
                    self.steps_since_label_reset = 0
                    # Hack: let's keep the ones that say "approximate" though. Those are not-Claude labels and probably fine.
                    location_archive = self.label_archive.get(location)
                    if location_archive:
                        for key, value in location_archive.items():
                            for key2, value2 in value.items():
                                if "approximate" not in value2.lower():
                                    del value[key2]
                            if not value:
                                del location_archive[key]
                logger.info(f"Completed step {steps_completed}/{num_steps}")
                self.text_display.add_message(f"Absolute step count: {self.absolute_step_count}")
                if save_file_name is not None and not steps_completed % save_every:
                    self.emulator.save_state(save_file_name)
                    self.save_location_archive(self.location_archive_file_name)
                    with open("location_milestones.txt", "w") as fw:
                        fw.write(str(self.location_milestones))

            except KeyboardInterrupt:
                logger.info("Received keyboard interrupt, stopping")
                self.running = False
            except Exception as e:
                logger.error(f"Error in agent loop: {e}")
                raise e
            
        if save_file_name is not None:
            logger.info("Saving state")
            self.emulator.save_state(save_file_name)
            self.save_location_archive(self.location_archive_file_name)
            with open("location_milestones.txt", "w") as fw:
                fw.write(str(self.location_milestones))

        if (
            not self.running
            # if the emulator is running in the main thread, we need to stop it
            # to allow the main thread to exit the emulator loop
            or self.pyboy_main_thread
        ):
            self.emulator.stop()

        self._steps_completed = steps_completed

        return steps_completed

    def navigation_assistance(self, navigation_goal: str) -> str:
        logger.info(f"[Agent] Running Navigation Assist...")
        
        _, location, coords = self.emulator.get_state_from_memory()

        collision_map = self.update_and_get_full_collision_map(location, coords)

        this_location = self.label_archive.get(location)
        if this_location is None:
            # this sucks man
            for key, value in self.label_archive.items():
                if location.lower() == key.lower():
                    this_location = value
                    break

        labels = "No Labels yet."
        all_labels = []
        if this_location is not None:
            for row_ind, this_row in this_location.items():
                for col_ind, this_col in this_row.items():
                    all_labels.append(((col_ind, row_ind), this_col)) 

        if all_labels:                
            labels = ','.join(f'{label_coords}: {label}' for label_coords, label in all_labels)

        mapping_query = f"""Here is a map of the current location:

        Current location: {location}

        {collision_map}

        Remember, higher numbers in the first coordinate are to the RIGHT. Higher numbers in the second coordinate are DOWN.
        
        Here are some labels:

        {labels}

        Here is the current navigation goal:

        {navigation_goal}
        """

        full_text = self.prompt_text_reply(NAVIGATION_PROMPT, mapping_query, False, MAPPING_MODEL, False)
        self.text_display.add_message(f"Navigation Advice: {full_text}")
        return full_text
    
    # Note: currently not used in any part of detailed_navigator_mode, so we don't have any handling for that.
    def prompt_text_reply(self, instructions: str, prompt: str, include_history: bool, model: str, include_screenshot: bool) -> str:

        if include_history:
            # Create messages for the summarization request - pass the entire conversation history
            messages = copy.deepcopy(self.message_history) 


            if len(messages) >= 3:
                if messages[-1]["role"] == "user" and isinstance(messages[-1]["content"], list) and messages[-1]["content"]:
                    messages[-1]["content"][-1]["cache_control"] = {"type": "ephemeral"}
                
                if len(messages) >= 5 and messages[-3]["role"] == "user" and isinstance(messages[-3]["content"], list) and messages[-3]["content"]:
                    messages[-3]["content"][-1]["cache_control"] = {"type": "ephemeral"}
        else:
            messages = []

        if include_screenshot:
            _, location, coords = self.emulator.get_state_from_memory()
            screenshot = self.emulator.get_screenshot()
            screenshot_b64 = self.get_screenshot_base64(screenshot, upscale=4, add_coords=True, player_coords=coords, location=location)

        # MODEL constant is removed. This function will now use self.sambanova_client.
        # The 'model' parameter to this function is also effectively removed/ignored.
        # This requires adapting the message formatting and API call to SambaNova.

        samba_messages = [] # Prepare messages in SambaNova format
        if include_history:
            # Convert self.message_history to SambaNova format
            # This is a placeholder; actual conversion logic depends on SambaNova's expected format.
            # For now, let's assume self.message_history is somewhat compatible or needs minor adjustments.
            samba_messages = copy.deepcopy(self.message_history) # Placeholder
            # Potentially strip or reformat parts of the history if needed for SambaNova.

        # Construct the current user message for SambaNova
        current_user_content_parts = []
        current_user_content_parts.append({"type": "text", "text": prompt}) # Assuming SambaNova takes text parts like this

        if include_screenshot:
            current_user_content_parts.append({
                "type": "image", # Or SambaNova's equivalent for image input
                "source": { # Or SambaNova's equivalent
                    "type": "base64",
                    "media_type": "image/png",
                    "data": screenshot_b64,
                },
            })
        
        samba_messages.append({"role": "user", "content": current_user_content_parts})


        # Placeholder for SambaNova API call
        # response = self.sambanova_client.chat.completions.create( # Or equivalent
        #     model=SAMBANOVA_STRATEGIST_MODEL_ID, # Or another appropriate model for summarization/text reply
        #     messages=samba_messages, # Ensure this is correctly formatted
        #     max_tokens=MAX_TOKENS,
        #     # system=instructions, # How SambaNova handles system/instructions
        #     temperature=TEMPERATURE,
        # )
        # # Extract the text - Adjust based on SambaNova's response structure
        # summary_text = "".join([choice.message.content for choice in response.choices if choice.message.content]) # Example

        # Due to the uncertainty of SambaNova's exact API for this,
        # I will temporarily return a placeholder summary.
        # This part MUST be updated with the correct SambaNova API calls.
        logger.error("SambaNova API call for text reply not yet implemented. Returning placeholder summary.")
        summary_text = "Placeholder summary from SambaNova integration."
        
        return summary_text
    
    def agentic_summary(self):
        self.text_display.add_message(f"[Agent] Generating Facts Analysis, standby...")

        memory_info, location, coords = self.emulator.get_state_from_memory()
        try:
            previous_summary = self.message_history[0]["content"][0]["text"]
        except TypeError:
            previous_summary = "Start of the Game!"
        last_checkpoints = '\n'.join(self.checkpoints[-10:])

        all_labels = self.get_all_location_labels(location)

        if not self.emulator.get_in_combat():
            collision_map = self.update_and_get_full_collision_map(location, coords)
        else:
            if location in self.full_collision_map:
                collision_map = self.full_collision_map[location].to_ascii(self.location_tracker.get(location, []))
            else:
                collision_map = "Not yet available"

        prompt = f"""
Here is key game information:

RAM Information: {memory_info}

Steps Since last Location Shift: {self.steps_since_location_shift}

text_based MAP: {collision_map}

Last 10 Checkpoints: {last_checkpoints}

Labeled nearby locations: {','.join(f'{coords}: {label}' for coords, label in all_labels)}

Previous game summary: {previous_summary}

A game screenshot is attached.

REMINDER: Your job is to deduce the current state of the game from that conversation, as well as additional data you will be provided,
Your job is NOT to play the game. Double-check your system prompt.
"""

        # Get the FACTS
        response1 = self.prompt_text_reply(META_KNOWLEDGE_PROMPT, prompt, True, MODEL, True)
        logger.info(f"Facts Stage 1: {response1}")
        # Clean Facts
        # MODEL constant is removed. Calls to prompt_text_reply will use SambaNova by default.
        response2 = self.prompt_text_reply(META_KNOWLEDGE_CLEANUP_PROMPT, response1, False, SAMBANOVA_STRATEGIST_MODEL_ID, False) # Explicitly pass model if needed by updated func
        logger.info(f"Facts Stage 2: {response2}")
        # Summarize for real
        response3 = self.prompt_text_reply(META_KNOWLEDGE_SUMMARIZER, response2, True, SAMBANOVA_STRATEGIST_MODEL_ID, False) # Explicitly pass model
        self.text_display.add_message(f"Final Summary: {response3}")
        with open("agentic_summary.txt", "w", encoding="utf-8") as fw:
            fw.write(response1 + "\n\n" + response2 + "\n\n" + response3)

        # Get a fresh screenshot after executing the buttons
        screenshot = self.emulator.get_screenshot()
        screenshot_b64 = self.get_screenshot_base64(screenshot, upscale=4, add_coords=True, player_coords=coords, location=location)

        # Replace message history with just the summary
        self.message_history = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"CONVERSATION HISTORY SUMMARY (representing {self.max_history} previous messages): {response3}"
                    },
                    {
                        "type": "text",
                        "text": "\n\nCurrent game screenshot for reference:"
                    },
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": screenshot_b64,
                        },
                    },
                    {
                        "type": "text",
                        "text": "You were just asked to summarize your playthrough so far, which is the summary you see above. You may now continue playing by selecting your next action."
                    },
                ]
            }
        ]
        self.openai_message_history = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": (f"CONVERSATION HISTORY SUMMARY (representing {self.max_history} previous messages): {response3}" +
                                 "\n\nCurrent game screenshot is also included." +
                                 "\n\nYou were just asked to summarize your playthrough so far, which is the summary you see above. You may now continue playing by selecting your next a.")
                    },
                    {
                        "type": "input_image",
                        "image_url": f"data:image/png;base64,{screenshot_b64}",
                    },
                ]
            }
        ]
        
    def stop(self):
        """Stop the agent."""
        self.running = False
        self.emulator.stop()


if __name__ == "__main__":
    # Get the ROM path relative to this file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    rom_path = os.path.join(os.path.dirname(current_dir), "pokemon.gb")

    # Create and run agent
    agent = SimpleAgent(rom_path)

    try:
        steps_completed = agent.run(num_steps=10)
        logger.info(f"Agent completed {steps_completed} steps")
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt, stopping")
    finally:
        agent.stop()

    