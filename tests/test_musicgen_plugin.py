# -*- encoding: utf-8 -*-
"""Tests for the Nendo MusicGen plugin."""
import unittest

from nendo import Nendo, NendoConfig

nd = Nendo(
    config=NendoConfig(
        library_path="./library",
        log_level="INFO",
        plugins=["nendo_plugin_musicgen"],
    )
)


class MusicGenPluginTest(unittest.TestCase):
    def test_run_musicgen_generation(self):
        nd.library.reset(force=True)
        track = nd.library.add_track(file_path="tests/assets/test.wav")
        gen_collection = nd.plugins.musicgen(
            track=track,
            n_samples=2,
            prompt="rnb, funky, fast, futuristic",
            bpm=116,
            key="C",
            scale="Major",
            model="facebook/musicgen-small",
            duration=10,
            conditioning_length=10,
        )

        self.assertEqual(len(nd.library.get_collection_tracks(gen_collection.id)), 2)
        self.assertEqual(len(nd.library.get_tracks()), 3)

    def test_run_process_musicgen_generation(self):
        nd.library.reset(force=True)
        track = nd.library.add_track(file_path="tests/assets/test.wav")
        gen_collection = track.process(
            "nendo_plugin_musicgen",
            n_samples=2,
            prompt="rnb, funky, fast, futuristic",
            bpm=116,
            key="C",
            scale="Major",
            model="facebook/musicgen-small",
            duration=10,
            conditioning_length=10,
        )

        self.assertEqual(len(nd.library.get_collection_tracks(gen_collection.id)), 2)
        self.assertEqual(len(nd.library.get_tracks()), 3)

    def test_run_musicgen_unconditional(self):
        nd.library.reset(force=True)

        gen_collection = nd.plugins.musicgen(
            n_samples=2,
            prompt="rnb, funky, fast, futuristic",
            bpm=116,
            key="C",
            scale="Major",
            model="GrandaddyShmax/musicgen-small",
            duration=5,
        )

        self.assertEqual(len(nd.library.get_collection_tracks(gen_collection.id)), 2)
        self.assertEqual(len(nd.library.get_tracks()), 2)

    def test_run_musicgen_melody_conditioning(self):
        nd.library.reset(force=True)
        track = nd.library.add_track(file_path="tests/assets/test.wav")

        gen_collection = nd.plugins.musicgen(
            track=track,
            n_samples=2,
            prompt="rnb, funky, fast, futuristic",
            bpm=116,
            key="C",
            scale="Major",
            model="GrandaddyShmax/musicgen-melody",
            duration=10,
            use_melody_conditioning=True,
        )

        self.assertEqual(len(nd.library.get_collection_tracks(gen_collection.id)), 2)
        self.assertEqual(len(nd.library.get_tracks()), 3)

    def test_run_process_musicgen_melody_conditioning(self):
        nd.library.reset(force=True)
        track = nd.library.add_track(file_path="tests/assets/test.wav")

        gen_collection = track.process(
            "nendo_plugin_musicgen",
            n_samples=2,
            prompt="rnb, funky, fast, futuristic",
            bpm=116,
            key="C",
            scale="Major",
            model="GrandaddyShmax/musicgen-melody",
            duration=10,
            use_melody_conditioning=True,
        )

        self.assertEqual(len(nd.library.get_collection_tracks(gen_collection.id)), 2)
        self.assertEqual(len(nd.library.get_tracks()), 3)


if __name__ == "__main__":
    unittest.main()
