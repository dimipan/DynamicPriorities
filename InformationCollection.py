from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple
from robot_utils import RobotAction, InformationCollectionActions

# a data container that represents a location where information can be collected in the game grid
@dataclass
class InfoLocation:
    position: List[int]   # (x, y) position of the information collection location
    info_type: str        # type of information that can be collected at this location
    collection_order: int # order in which the information should be collected
    required: bool = True # whether the information is required to complete the task

class InfoCollectionSystem:
    def __init__(self, info_locations: List[InfoLocation]):
        """Initialize the information collection system."""
        self.info_locations = sorted(info_locations, key=lambda x: x.collection_order) # Sort locations by collection order
        self.collected_info = []  # Episode-specific tracking of collected information
        self.current_collection_index = 0 # Index of the current collection target
        self.total_attempts = {tuple(loc.position): 0 for loc in info_locations}    # Episode-specific tracking of attempts
        self.correct_attempts = {tuple(loc.position): 0 for loc in info_locations}  # Episode-specific tracking of successes
        self.cumulative_total_attempts = {tuple(loc.position): 0 for loc in info_locations}  # Cumulative tracking across all episodes for attempts
        self.cumulative_correct_attempts = {tuple(loc.position): 0 for loc in info_locations} # Cumulative tracking across all episodes for successes
        self.position_to_info = {tuple(loc.position): loc.info_type for loc in info_locations} # Create a mapping of positions to info types for easier lookup
        # Store original info locations for resetting if needed
        self.original_info_locations = info_locations.copy()

    def reset_episode(self):
        """Reset episode-specific variables while preserving cumulative stats."""
        self.collected_info = []
        self.current_collection_index = 0
        # Reset episode-specific counters
        self.total_attempts = {tuple(loc.position): 0 for loc in self.info_locations}
        self.correct_attempts = {tuple(loc.position): 0 for loc in self.info_locations}
    
    def collect_info(self, info_type: str, position: List[int]) -> bool:
        """Attempts to collect information of given type. Updates both attempts and successes."""
        pos_tuple = tuple(position)
        # Always increment attempts for any collection action at a valid location
        if pos_tuple in self.position_to_info:  # Only count attempts at valid info locations
            self.total_attempts[pos_tuple] += 1
            self.cumulative_total_attempts[pos_tuple] += 1
        if self.current_collection_index >= len(self.info_locations):
            return False
        expected_info = self.info_locations[self.current_collection_index]
        if info_type == expected_info.info_type and position == expected_info.position:
            self.collected_info.append(info_type)
            self.correct_attempts[pos_tuple] += 1
            self.cumulative_correct_attempts[pos_tuple] += 1
            self.current_collection_index += 1
            return True
        return False
    
    def reorder_collection_priorities(self, new_order_map: Dict[str, int]):
        """
        Reorder the collection priorities of information.
        
        Args:
            new_order_map: A dictionary mapping info_type to its new collection_order
        """
        # Update the collection order of each info location
        for loc in self.info_locations:
            if loc.info_type in new_order_map:
                loc.collection_order = new_order_map[loc.info_type]
        
        # Resort the info locations based on the new collection order
        self.info_locations = sorted(self.info_locations, key=lambda x: x.collection_order)
        
        # If we're in the middle of collection, we need to adjust the current_collection_index
        if self.collected_info:
            # Find the index of the next info type to collect
            for i, loc in enumerate(self.info_locations):
                if loc.info_type not in self.collected_info:
                    self.current_collection_index = i
                    break
    
    def get_current_priority_location(self) -> Optional[List[int]]:
        """Returns the position of the current highest priority information location."""
        if self.current_collection_index < len(self.info_locations):
            return self.info_locations[self.current_collection_index].position
        return None
    
    def get_info_type_by_order(self, order: int) -> Optional[str]:
        """Returns the info type at the specified collection order."""
        for loc in self.info_locations:
            if loc.collection_order == order:
                return loc.info_type
        return None
    
    def get_current_priority_info_type(self) -> Optional[str]:
        """Returns the current highest priority information type."""
        if self.current_collection_index < len(self.info_locations):
            return self.info_locations[self.current_collection_index].info_type
        return None

    def can_collect_at_position(self, position: List[int]) -> Optional[str]:
        """Returns collectible info type at position if collection is valid."""
        if self.current_collection_index >= len(self.info_locations):
            return None
        current_location = self.info_locations[self.current_collection_index]
        if position == current_location.position:
            return current_location.info_type
        return None
    
    def is_at_info_location(self, state: List[int]) -> bool:
        """Check if the agent is at an information collection location with the right collection count"""
        current_pos = [state[0], state[1]]
        collected_count = state[2]
        for location in self.info_locations:
            if (current_pos == location.position and 
                collected_count == location.collection_order):
                return True
        return False
    
    def get_collection_action_index(self, action_name: str, hierarchical: bool = False) -> Optional[int]:
        """Convert predicted action name to action index for the agent."""
        try:
            if hierarchical:
                return InformationCollectionActions[action_name].value
            else:
                return RobotAction[action_name].value
        except KeyError:
            return None

    def get_collection_stats(self) -> Dict[str, Dict[Tuple[int, int], float]]:
        """Returns comprehensive statistics about collection attempts."""
        stats = {
            'total_attempts_by_location': self.total_attempts.copy(),
            'correct_attempts_by_location': self.correct_attempts.copy(),
            'cumulative_total_attempts': self.cumulative_total_attempts.copy(),
            'cumulative_correct_attempts': self.cumulative_correct_attempts.copy(),
            'cumulative_success_rates': {
                pos: (self.cumulative_correct_attempts[pos] / max(self.cumulative_total_attempts[pos], 1) * 100)
                for pos in self.cumulative_total_attempts.keys()
            }
        }
        return stats

    def get_required_info_count(self) -> int:
        """Returns the number of required information points."""
        return len([info for info in self.info_locations if info.required])

    def get_collected_info_count(self) -> int:
        """Returns the number of collected information points."""
        return len(self.collected_info)
    
    def get_info_locations_by_priority(self) -> List[Tuple[str, List[int]]]:
        """Returns a list of (info_type, position) tuples sorted by collection priority."""
        return [(loc.info_type, loc.position) for loc in self.info_locations]

    def get_collection_success_rate(self) -> float:
        """Calculate the overall success rate across all locations."""
        total_attempts = sum(self.cumulative_total_attempts.values())
        total_correct = sum(self.cumulative_correct_attempts.values())
        if total_attempts == 0:
            return 0.0
        return (total_correct / total_attempts) * 100

    def print_collection_stats(self):
        """Print detailed statistics about collection attempts."""
        stats = self.get_collection_stats()
        print("\nCollection Statistics (Current Episode):")
        for pos, attempts in stats['total_attempts_by_location'].items():
            correct = stats['correct_attempts_by_location'][pos]
            success_rate = (correct / max(attempts, 1) * 100)
            print(f"Location {pos}: {attempts} attempts, {correct} correct ({success_rate:.1f}%)")
        print("\nCumulative Collection Statistics (across Training):")
        for pos in self.cumulative_total_attempts.keys():
            total = stats['cumulative_total_attempts'][pos]
            correct = stats['cumulative_correct_attempts'][pos]
            success_rate = stats['cumulative_success_rates'][pos]
            print(f"Location {pos}: {total} total attempts, {correct} correct ({success_rate:.1f}%)")
        total_attempts = sum(self.cumulative_total_attempts.values())
        total_correct = sum(self.cumulative_correct_attempts.values())
        overall_rate = (total_correct / max(total_attempts, 1) * 100)
        print(f"\nOverall Success Rate: {overall_rate:.1f}%")