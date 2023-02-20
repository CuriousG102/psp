import logging
import unittest

import color_grid


class ColorGridTest(unittest.TestCase):
    # def get_env(self, num_cells_per_dim, num_colors_per_cell, evil_level):

    def test_split_reward_space_max_evil(self):
        self.assertEqual(
            color_grid._split_reward_space([0, (.8, 1)], 4, True),
            [1, 3])
        self.assertEqual(
            color_grid._split_reward_space([(0, 1)], 4, True),
            [4]
        )
        self.assertEqual(
            color_grid._split_reward_space([0, (.4, .6), (.8, 1)], 4, True),
            [1, 2, 1]
        )

        with self.assertRaises(AssertionError):
            color_grid._split_reward_space([0, (.8, 1)], 1, True)

        with self.assertRaises(AssertionError):
            color_grid._split_reward_space([0, 1], 3, True)

    def test_split_reward_space_no_max_evil(self):
        with self.assertRaises(AssertionError):
            color_grid._split_reward_space([0, (.8, 1)], 2, False)
        with self.assertRaises(AssertionError):
            color_grid._split_reward_space([0], 1, False)

        self.assertEqual(
            color_grid._split_reward_space([0, (.8, 1)], 3, False),
            [1, 2])

    def test_get_colors_for_action_and_reward_max_evil(self):
        with self.assertLogs(level=logging.WARNING) as cm:
            self.assertEqual(
                color_grid._get_colors_for_action_and_reward_max_evil(
                    9, [0, 2], [(0, 1)]),
                ([2, 2], [2]))
        self.assertEqual(
            cm.output,
            [f'WARNING:root:{color_grid._get_num_colors_log_message(9, 1)}'])

    def test_get_reward_idx(self):
        self.assertEqual(
            color_grid._get_reward_idx(0, [0, (.1, .3), (.6, 1.)], [1, 3, 2]),
            0
        )
        self.assertEqual(
            color_grid._get_reward_idx(
                .2/3 * 0 + .1, [0, (.1, .3), (.6, 1.)], [1, 3, 2]),
            1
        )
        self.assertEqual(
            color_grid._get_reward_idx(
                .2/3 * 1 + .1, [0, (.1, .3), (.6, 1.)], [1, 3, 2]),
            2
        )
        self.assertEqual(
            color_grid._get_reward_idx(
                .2/3 * 2 + .1, [0, (.1, .3), (.6, 1.)], [1, 3, 2]),
            3
        )
        self.assertEqual(
            color_grid._get_reward_idx(.7, [0, (.1, .3), (.6, 1.)], [1, 3, 2]),
            4,
        )
        self.assertEqual(
            color_grid._get_reward_idx(.9, [0, (.1, .3), (.6, 1.)], [1, 3, 2]),
            5,
        )

    def test_get_action_idx(self):
        self.assertEqual(
            color_grid._get_action_idx(
                [-2 / 3, 0, -1, 0, -2 / 3, 0, 0], [0, 2, 4], [3, 2, 3]),
            0
        )
        self.assertEqual(
            color_grid._get_action_idx(
                [-2 / 3, 0, -1, 0, 0 + 1e-6, 0, 0], [0, 2, 4], [3, 2, 3]),
            1
        )
        self.assertEqual(
            color_grid._get_action_idx(
                [-2 / 3, 0, -1, 0, 2 / 3 + 1e-6, 0, 0], [0, 2, 4], [3, 2, 3]),
            2
        )

        self.assertEqual(
            color_grid._get_action_idx(
                [-2 / 3, 0, 1, 0, -2 / 3, 0, 0], [0, 2, 4], [3, 2, 3]),
            3
        )
        self.assertEqual(
            color_grid._get_action_idx(
                [-2 / 3, 0, 1, 0, 0 + 1e-6, 0, 0], [0, 2, 4], [3, 2, 3]),
            4
        )
        self.assertEqual(
            color_grid._get_action_idx(
                [-2 / 3, 0, 1, 0, 2 / 3 + 1e-6, 0, 0], [0, 2, 4], [3, 2, 3]),
            5
        )

        self.assertEqual(
            color_grid._get_action_idx(
                [1e-6, 0, -1, 0, -2 / 3, 0, 0], [0, 2, 4], [3, 2, 3]),
            6
        )
        self.assertEqual(
            color_grid._get_action_idx(
                [1e-6, 0, -1, 0, 0 + 1e-6, 0, 0], [0, 2, 4], [3, 2, 3]),
            7
        )
        self.assertEqual(
            color_grid._get_action_idx(
                [1e-6, 0, -1, 0, 2 / 3 + 1e-6, 0, 0], [0, 2, 4], [3, 2, 3]),
            8
        )

        self.assertEqual(
            color_grid._get_action_idx(
                [1e-6, 0, 1, 0, -2 / 3, 0, 0], [0, 2, 4], [3, 2, 3]),
            9
        )
        self.assertEqual(
            color_grid._get_action_idx(
                [1e-6, 0, 1, 0, 0 + 1e-6, 0, 0], [0, 2, 4], [3, 2, 3]),
            10
        )
        self.assertEqual(
            color_grid._get_action_idx(
                [1e-6, 0, 1, 0, 2 / 3 + 1e-6, 0, 0], [0, 2, 4], [3, 2, 3]),
            11
        )

    def test_maximum_evil_lt_min_colors(self):
        pass

    def test_maximum_evil(self):
        pass

    def test_evil_reward(self):
        pass

    def test_evil_action(self):
        pass

    def test_evil_sequence(self):
        pass

    def test_min_evil(self):
        pass

    def test_random(self):
        pass

    def test_none(self):
        pass


if __name__ == '__main__':
    unittest.main()
