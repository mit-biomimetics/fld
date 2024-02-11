# Legged Gym Utilities

## Keyboard Controller

By overwriting the `_get_keyboard_events()` method, a custom keyboard controller can be added to the environment. The keyboard controller subscribes to IsaacGym's keyboard-system, therefore the events are only caught, if the IsaacGym window is focused.

### Example

```python
from humanoid_gym.utils.keyboard_controller import KeyboardAction, Button, Delta, Switch

    def _get_keyboard_events(self) -> Dict[str, KeyboardAction]:
        # Simple keyboard controller for linear and angular velocity

        def print_command():
            print(f"New command: {self.commands[0]}")

        key_board_events = {
            'u' : Delta("lin_vel_x", amount =  0.1,  variable_reference = self.commands[:, 0], callback = print_command),
            'j' : Delta("lin_vel_x", amount = -0.1, variable_reference = self.commands[:, 0], callback = print_command),
            'h' : Delta("lin_vel_y", amount =  0.1,  variable_reference = self.commands[:, 1], callback = print_command),
            'k' : Delta("lin_vel_y", amount = -0.1, variable_reference = self.commands[:, 1], callback = print_command),
            'y' : Delta("ang_vel_z", amount =  0.1,  variable_reference = self.commands[:, 2], callback = print_command),
            'i' : Delta("ang_vel_z",amount = -0.1, variable_reference = self.commands[:, 2], callback = print_command),
            'm' : Button("some_var", 0, 1, self.commands[:, someIndex], print_command)
            'n' : Switch("some_other_var", 0, 1, self.commands[:, someIndex], print_command)        
        }
        return key_board_events
```

A parent keyboard can also be extended by calling the `super()` method:

```python
    def _get_keyboard_events(self) -> Dict[str, KeyboardAction]:
        basic_keyboard = super()._get_keyboard_events()
        basic_keyboard['x'] = Button("new_var", 0, 1, self.commands[:, someIndex], None)
        return basic_keyboard
```

The following keyboard events are available:

|**Classname** | **Parameters** | **Description** |
|--------------|----------------|-----------------|
| Delta | `amount`, `variable_reference`, `change_callback` (optional) | Increments the `reference_variable` by its amount and calls the `change_callback` if it was passed |
| Button | `start_state`, `toggle_state`, `variable_reference`, `callback` (optional) | Sets `variable_reference[:] = toggle_state` for the duration the button is held down. Resets to `start_state` afterwards. Calls the `callback` if it was passed. |
| Switch | `start_state`, `toggle_state`, `index`, `variable_reference`, `callback` (optional) | Toggles `variable_reference[:]` between the `toggle_state` and `start_state` every time the button is pressed and released. Calls the `callback` if it was passed. |
| DelegateHandle | `delegate`, `edge_detection`, `callback` | Exectues the function handle `delegate` when the key was pressed. If `edge_detection` is true, it only executes in on rising edges. Executes the `callback` whenever the function handle was called. |

With the `DelegateHandle` keyboard-event basically every desired action can be implemented. `Delta`, `Button` and `Switch` are only commonly used helpers.

### **Available keys**

The list of keys you can use (e.g. `basic_keyboard['KEY_NAME']`) be found below. The Controller takes the key in the dictionary (e.g. `x`), transforms it to capital letters and prepends `KEY_` to it. So to get an event on `KEY_RIGHT_ALT`, you have to add `basic_keyboard['right_alt'] = [...]`.

<details>
 <summary> Click here to see all options </summary>
    KEY_SPACE,
    KEY_APOSTROPHE,
    KEY_COMMA,
    KEY_MINUS,
    KEY_PERIOD,
    KEY_SLASH,
    KEY_0,
    KEY_1,
    KEY_2,
    KEY_3,
    KEY_4,
    KEY_5,
    KEY_6,
    KEY_7,
    KEY_8,
    KEY_9,
    KEY_SEMICOLON,
    KEY_EQUAL,
    KEY_A,
    KEY_B,
    KEY_C,
    KEY_D,
    KEY_E,
    KEY_F,
    KEY_G,
    KEY_H,
    KEY_I,
    KEY_J,
    KEY_K,
    KEY_L,
    KEY_M,
    KEY_N,
    KEY_O,
    KEY_P,
    KEY_Q,
    KEY_R,
    KEY_S,
    KEY_T,
    KEY_U,
    KEY_V,
    KEY_W,
    KEY_X,
    KEY_Y,
    KEY_Z,
    KEY_LEFT_BRACKET,
    KEY_BACKSLASH,
    KEY_RIGHT_BRACKET,
    KEY_GRAVE_ACCENT,
    KEY_ESCAPE,
    KEY_TAB,
    KEY_ENTER,
    KEY_BACKSPACE,
    KEY_INSERT,
    KEY_DEL,
    KEY_RIGHT,
    KEY_LEFT,
    KEY_DOWN,
    KEY_UP,
    KEY_PAGE_UP,
    KEY_PAGE_DOWN,
    KEY_HOME,
    KEY_END,
    KEY_CAPS_LOCK,
    KEY_SCROLL_LOCK,
    KEY_NUM_LOCK,
    KEY_PRINT_SCREEN,
    KEY_PAUSE,
    KEY_F1,
    KEY_F2,
    KEY_F3,
    KEY_F4,
    KEY_F5,
    KEY_F6,
    KEY_F7,
    KEY_F8,
    KEY_F9,
    KEY_F10,
    KEY_F11,
    KEY_F12,
    KEY_NUMPAD_0,
    KEY_NUMPAD_1,
    KEY_NUMPAD_2,
    KEY_NUMPAD_3,
    KEY_NUMPAD_4,
    KEY_NUMPAD_5,
    KEY_NUMPAD_6,
    KEY_NUMPAD_7,
    KEY_NUMPAD_8,
    KEY_NUMPAD_9,
    KEY_NUMPAD_DEL,
    KEY_NUMPAD_DIVIDE,
    KEY_NUMPAD_MULTIPLY,
    KEY_NUMPAD_SUBTRACT,
    KEY_NUMPAD_ADD,
    KEY_NUMPAD_ENTER,
    KEY_NUMPAD_EQUAL,
    KEY_LEFT_SHIFT,
    KEY_LEFT_CONTROL,
    KEY_LEFT_ALT,
    KEY_LEFT_SUPER,
    KEY_RIGHT_SHIFT,
    KEY_RIGHT_CONTROL,
    KEY_RIGHT_ALT,
    KEY_RIGHT_SUPER,
    KEY_MENU
</details>

An the exact list depends on your isaacgym version and can be found in the docs folder of your local isaacgym_lib copy: `isaacgym_lib/docs/api/python/enum_py.html#isaacgym.gymapi.KeyboardInput`.
