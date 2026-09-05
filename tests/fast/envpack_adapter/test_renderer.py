from __future__ import annotations

import base64
import sys
import types
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from orbit_plugins.envpack_adapter.renderer import observation_to_chat_message


_PNG_1X1 = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/p9sAAAAASUVORK5CYII="
)


class EnvpackRendererTest(unittest.TestCase):
    def test_observation_media_bytes_are_model_input_source(self) -> None:
        try:
            import PIL  # noqa: F401
        except ModuleNotFoundError:
            self.skipTest("requires Pillow")

        client_mod = types.ModuleType("envpack.client")

        def prepare_observation_message(observation, *, role, strict_placeholders):
            self.assertTrue(strict_placeholders)
            return SimpleNamespace(
                message={"role": role, "content": [{"type": "image"}, {"type": "text", "text": observation.text}]},
                media=[SimpleNamespace(kind="image", bytes=_PNG_1X1)],
                media_hashes=["expected-sha"],
                artifacts=[SimpleNamespace(uri="artifact://not-used")],
            )

        client_mod.prepare_observation_message = prepare_observation_message
        package_mod = types.ModuleType("envpack")
        package_mod.client = client_mod

        with patch.dict(sys.modules, {"envpack": package_mod, "envpack.client": client_mod}):
            rendered = observation_to_chat_message(SimpleNamespace(text="board"))

        self.assertEqual(rendered.message["role"], "user")
        self.assertEqual(rendered.media_hashes, ["expected-sha"])
        self.assertEqual(len(rendered.images), 1)
        self.assertEqual(rendered.images[0].size, (1, 1))
        self.assertEqual(rendered.artifacts[0].uri, "artifact://not-used")


if __name__ == "__main__":
    unittest.main()
