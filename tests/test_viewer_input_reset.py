import os
import unittest
from unittest.mock import patch

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "hide")

import pygame

from _00_environment.engine import Engine
from _00_environment.viewer import Viewer


class _FakePressedState:
    def __init__(self, pressed_keys=None):
        self.pressed_keys = set(pressed_keys or ())

    def __getitem__(self, key_value):
        return key_value in self.pressed_keys


class ViewerInputResetTests(unittest.TestCase):
    def setUp(self):
        engine = Engine(False, False, False)
        self.viewer = Viewer(engine)
        self.viewer.headless = False

    def tearDown(self):
        self.viewer.close()

    def test_reset_input_state_keeps_currently_held_direction_key(self):
        self.viewer.pressed_keys = {pygame.K_LEFT, pygame.K_RIGHT}
        self.viewer.bnw_code_buffer = "30"

        with patch(
            "pygame.key.get_pressed",
            return_value=_FakePressedState({pygame.K_LEFT}),
        ):
            self.viewer.reset_input_state()

        self.assertEqual("", self.viewer.bnw_code_buffer)
        self.assertIn(pygame.K_LEFT, self.viewer.pressed_keys)
        self.assertNotIn(pygame.K_RIGHT, self.viewer.pressed_keys)

    def test_get_human_input_uses_live_keyboard_state_after_reset(self):
        self.viewer.pressed_keys.clear()

        with patch.object(self.viewer, "_process_events", return_value=None):
            with patch(
                "pygame.key.get_pressed",
                return_value=_FakePressedState({pygame.K_LEFT, pygame.K_UP}),
            ):
                user_input, action_name = self.viewer.get_human_input(1)

        self.assertEqual(-1, user_input.x_direction)
        self.assertEqual(-1, user_input.y_direction)
        self.assertEqual(0, user_input.power_hit)
        self.assertEqual("idle", action_name)


if __name__ == "__main__":
    unittest.main()
