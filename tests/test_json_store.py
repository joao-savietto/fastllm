import json
import os
import shutil
import tempfile
import unittest

from fastllm.store.json_store import JSONChatStorage


class TestJSONChatStorage(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.store = JSONChatStorage(storage_dir=self.test_dir)

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_save_and_get_all(self):
        msg1 = {"role": "user", "content": "hello"}
        msg2 = {"role": "assistant", "content": "hi"}

        self.store.save(msg1, "session1")
        self.store.save(msg2, "session1")

        messages = self.store.get_all("session1")
        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[0], msg1)
        self.assertEqual(messages[1], msg2)

        # Test default session
        self.store.save(msg1)
        self.assertEqual(len(self.store.get_all()), 1)

    def test_persistence(self):
        # Test that data persists across instances
        msg = {"role": "user", "content": "hello"}
        self.store.save(msg, "session1")

        # Create new instance pointing to same dir
        new_store = JSONChatStorage(storage_dir=self.test_dir)
        messages = new_store.get_all("session1")
        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0], msg)

    def test_del_session(self):
        self.store.save({"a": 1}, "session1")
        self.assertTrue(
            os.path.exists(os.path.join(self.test_dir, "session1.json"))
        )

        self.store.del_session("session1")
        self.assertFalse(
            os.path.exists(os.path.join(self.test_dir, "session1.json"))
        )
        self.assertEqual(self.store.get_all("session1"), [])

    def test_del_all_sessions(self):
        self.store.save({"a": 1}, "s1")
        self.store.save({"b": 2}, "s2")

        self.store.del_all_sessions()
        self.assertEqual(self.store.get_all("s1"), [])
        self.assertEqual(self.store.get_all("s2"), [])
        self.assertEqual(len(os.listdir(self.test_dir)), 0)

    def test_get_message(self):
        msg1 = {"role": "user", "content": "1"}
        msg2 = {"role": "user", "content": "2"}
        self.store.save(msg1, "s1")
        self.store.save(msg2, "s1")

        self.assertEqual(self.store.get_message(0, "s1"), msg1)
        self.assertEqual(self.store.get_message(1, "s1"), msg2)

        with self.assertRaises(IndexError):
            self.store.get_message(2, "s1")

        with self.assertRaises(KeyError):
            self.store.get_message(0, "nonexistent")

    def test_set_message(self):
        msg1 = {"role": "user", "content": "1"}
        self.store.save(msg1, "s1")

        new_msg = {"role": "user", "content": "new"}
        self.store.set_message(0, new_msg, "s1")

        self.assertEqual(self.store.get_message(0, "s1"), new_msg)

        with self.assertRaises(IndexError):
            self.store.set_message(1, new_msg, "s1")

    def test_del_message(self):
        msg1 = {"role": "user", "content": "1"}
        msg2 = {"role": "user", "content": "2"}
        self.store.save(msg1, "s1")
        self.store.save(msg2, "s1")

        self.store.del_message(0, "s1")
        messages = self.store.get_all("s1")
        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0], msg2)

        with self.assertRaises(IndexError):
            self.store.del_message(5, "s1")

        with self.assertRaises(KeyError):
            self.store.del_message(0, "nonexistent")

    def test_sanitization(self):
        # Test path traversal or weird characters
        self.store.save({"a": 1}, "../badsession")
        # Should be saved as ..badsession.json or something safe, definitely not in parent dir
        # My implementation: keep alnum, -, _
        # so ..badsession -> badsession

        self.assertTrue(
            os.path.exists(os.path.join(self.test_dir, "badsession.json"))
        )

    def test_non_dict_message(self):
        # Test handling of objects with .dict() method
        class MockModel:
            def dict(self):
                return {"mock": "data"}

        self.store.save(MockModel(), "s_model")
        messages = self.store.get_all("s_model")
        self.assertEqual(messages[0], {"mock": "data"})
