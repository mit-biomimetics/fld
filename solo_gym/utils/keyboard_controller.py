from __future__ import annotations

# isaacgym
import isaacgym

# python
from abc import abstractmethod, ABC
from typing import Callable, Any, Dict
import torch


# Callback function for when event is triggered.
CallbackFn = Callable[[], None]
# Action function called when event is trigged.
# Takes the environment instance and event value.
ActionFn = Callable[[Any, int], None]


class KeyboardAction(ABC):
    """Base class for keyboard event."""

    def __init__(
        self,
        name: str,
        variable_reference: torch.Tensor = None,
        member_name: str = None,
    ):
        """Initializes the keyboard action event with variable attributes.

        Note:
            The keyboard action can be applied to a variable (passed via reference) or
            on a member in the environment class instance.

        Args:
            name (str): Name of the affected value.
            variable_reference (torch.Tensor, optional): Reference variable to alter value. Defaults to None.
            member_name (str, optional): Name of the variable in the environment. Defaults to None.

        Raises:
            ValueError -- If both reference variable and environment's member name are None or not None.
        """
        # check input
        if (variable_reference is None and member_name is None) or (
            variable_reference is not None and member_name is not None
        ):
            msg = "Invalid arguments: Action can only be applied on either reference variable or environment's member variable."
            raise ValueError(msg)
        # store input arguments
        self.name = name
        self.variable_reference = variable_reference
        self.member_name = member_name
        # disambiguate the type of mode
        if variable_reference is not None and member_name is None:
            self._ref_mode = True
        elif variable_reference is None and member_name is not None:
            self._ref_mode = False

    def __str__(self) -> str:
        """Helper string to explain keyboard action."""
        return f"Keyboard action on {self.name}."

    def get_reference(self, env) -> torch.Tensor:
        """Retrieve the variable on which event action is applied.

        Args:
            env (BaseTask): The environment/task instance.

        Returns:
            torch.Tensor: The passed variable reference or environment instance's member.
        """
        if self._ref_mode:
            return self.variable_reference
        else:
            return getattr(env, self.member_name)

    @abstractmethod
    def do(self, env, value: int):
        """Action applied by the keyboard event.

        Args:
            env (BaseTask): The environment/task instance.
            value (int): The event triggered when keyboard button pressed.
        """
        raise NotImplementedError


class DelegateHandle(KeyboardAction):
    """Pre-defined delegate that executes an event handler.

    This class exectues the function handle `delegate` when the key is pressed. If `edge_detection` is
    true, then the function executes only on rising edges (i.e. release of the key).

    The `callback` function is executed whenever the function handle is called.
    """

    def __init__(
        self,
        name: str,
        delegate: ActionFn,
        edge_detection: bool = True,
        callback: CallbackFn = None,
        variable_reference: torch.Tensor = None,
        member_name: str = None,
    ):
        """Initializes the class.

        Args:
            name (str): Name of the affected value.
            delegate (ActionFn): The function called when keyboard is pressed/released.
            edge_detection (bool, optional): Decides whether to change value on press/release. Defaults to True.
            callback (CallbackFn, optional): Function called whenever key triggered. Defaults to None.
            variable_reference (torch.Tensor, optional): Reference variable to alter value. Defaults to None.
            member_name (str, optional): Name of the variable in the environment. Defaults to None.
        """
        super().__init__(name, variable_reference, member_name)
        # store inputs
        self._delegate = delegate
        self._edge_detection = edge_detection
        self._callback = callback

    def do(self, env, value):
        """Action applied by the keyboard event.

        Args:
            env (BaseTask): The environment/task instance.
            value (int): The event triggered when keyboard button pressed.
        """
        # if no event triggered return.
        if self._edge_detection and value == 0:
            return
        # resolve action based on press/release
        self._delegate(env, value)
        # trigger callback function
        if self._callback is not None:
            self._callback()


class Delta(DelegateHandle):
    """Keyboard action that increments the value of reference variable by scalar amount."""

    def __init__(
        self,
        name: str,
        amount: float,
        variable_reference: torch.Tensor,
        callback: CallbackFn = None,
    ):
        """Initializes the class.

        Args:
            name (str): Name of the affected value.
            amount (float): The amount by which to increment.
            variable_reference (torch.Tensor): Reference variable to alter value.
            callback (CallbackFn, optional): Function called whenever key triggered. Defaults to None.
        """
        self.amount = amount

        # delegate function
        def addDelta(env, value):
            self.variable_reference += self.amount

        # initialize parent
        super().__init__(name, addDelta, True, callback, variable_reference, None)

    def __str__(self) -> str:
        if self.amount >= 0:
            return f"Increments the variable {self.name} by {self.amount}"
        else:
            return f"Decrements the variable {self.name} by {-self.amount}"


class Switch(DelegateHandle):
    """Keyboard action that toggles between values of reference variable."""

    def __init__(
        self,
        name: str,
        start_state: torch.Tensor,
        toggle_state: torch.Tensor,
        variable_reference: torch.Tensor,
        callback: CallbackFn = None,
    ):
        """Initializes the class.

        Args:
            name (str): Name of the affected value.
            start_state (torch.Tensor): Initial value of reference variable.
            toggle_state (torch.Tensor): Toggled value of reference variable.
            variable_reference (torch.Tensor): Reference variable to alter value.
            callback (CallbackFn, optional): Function called whenever key triggered. Defaults to None.
        """
        # copy inputs to class
        self.start_state = start_state
        self.toggle_state = toggle_state
        self.variable_reference = variable_reference
        # initial state of toggle switch
        self.switch_value = True

        # delegate function
        def switchState(env, value):
            # switch between state depending on switch's value
            if self.switch_value:
                new_state = self.toggle_state
            else:
                new_state = self.start_state
            # store value into reference variable
            self.variable_reference[:] = new_state
            # toggle switch to other state
            self.switch_value = not self.switch_value

        # initialize parent
        super().__init__(name, switchState, True, callback, variable_reference, None)

    def __str__(self) -> str:
        return f"Toggles the variable {self.name} between {self.toggle_state} and {self.start_state}."


class Button(Switch):
    """Sets the variable to value only while keyboard button is pressed."""

    def __init__(
        self,
        name: str,
        start_state: torch.Tensor,
        toggle_state: torch.Tensor,
        variable_reference: torch.Tensor,
        callback: CallbackFn = None,
    ):
        """Initializes the class.

        Args:
            name (str): Name of the affected value.
            start_state (torch.Tensor): Initial value of reference variable.
            toggle_state (torch.Tensor): Toggled value of reference variable.
            variable_reference (torch.Tensor): Reference variable to alter value.
            callback (CallbackFn, optional): Function called whenever key triggered. Defaults to None.
        """
        # initialize toggle switch
        super().__init__(name, start_state, toggle_state, variable_reference, callback)
        # trigger event only when key is pressed
        self._edge_detection = False

    def __str__(self) -> str:
        return f"Sets the variable {self.name} to {self.toggle_state} only while key is pressed."


class KeyBoardController:
    """Wrapper around IsaacGym viewer to handle different keyboard actions."""

    def __init__(self, env, key_actions: Dict[str, KeyboardAction]):
        """Initializes the class.

        Args:
            env (BaseTask): The environment/task instance.
            key_actions (Dict[str, KeyboardAction]): The pairs of key buttons and their actions.
        """
        # store inputs
        self._env = env
        self._key_actions = key_actions
        # setup the keyboard event subscriber
        for key_name in self._key_actions.keys():
            key_enum = getattr(isaacgym.gymapi.KeyboardInput, f"KEY_{key_name.capitalize()}")
            env.gym.subscribe_viewer_keyboard_event(env.viewer, key_enum, key_name)

    def update(self, env):
        """Update the reference variables by querying viewer events."""
        # gather all events on viewer
        events = env.gym.query_viewer_action_events(env.viewer)
        # iterate over events
        for event in events:
            key_pressed = event.action
            if key_pressed in self._key_actions:
                cfg = self._key_actions[key_pressed]
                cfg.do(env, event.value)

    def print_options(self):
        print("[KeyboardController] Key-action pairs:")
        for key_name, action in self._key_actions.items():
            print(f"\t{key_name}: {action}")
