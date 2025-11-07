import numpy as np
from typing import Optional



class ReplayBuffer:
    """
        Replay buffer holds sample trajectories
    """

    @staticmethod
    def from_numpy(
        y: np.ndarray,
        z: np.ndarray,
        done: Optional[np.ndarray]=None,
    ):
        size, y_dim = y.shape
        _, z_dim = z.shape
        
        buffer = ReplayBuffer(
            capacity=size,
            y_dim=y_dim,
            z_dim=z_dim
        )
        print("loading data from numpy array ...")
        for i in range(size):
            buffer.push(
                y=y[i],
                z=z[i],
                done=done[i] if done is not None else False,
            )
        return buffer

    def __init__(
        self,
        capacity: int,
        y_dim: int,
        z_dim: int,
    ):
        self.capacity = capacity

        self.y_dim = y_dim
        self.z_dim = z_dim

        self.ys = np.zeros((capacity, y_dim), dtype=np.float32)
        self.zs = np.zeros((capacity, z_dim), dtype=np.float32)
        self.done = np.zeros((capacity, 1), dtype=bool)

        self.index = 0
        self.is_filled = False

    def __len__(self):
        return self.capacity if self.is_filled else self.index

    def push(
        self,
        y,
        z,
        done,
    ):
        """
            Add experience (single step) to the replay buffer
        """
        self.ys[self.index] = y
        self.zs[self.index] = z
        self.done[self.index] = done

        self.index = (self.index + 1) % self.capacity
        self.is_filled = self.is_filled or self.index == 0

    def sample(
        self,
        batch_size: int,
        chunk_length: int
    ):
        done = self.done.copy()
        done[-1] = 1
        episode_ends = np.where(done)[0]

        all_indexes = np.arange(len(self))
        distances = episode_ends[np.searchsorted(episode_ends, all_indexes)] - all_indexes + 1
        valid_indexes = all_indexes[distances >= chunk_length]

        sampled_indexes = np.random.choice(valid_indexes, size=batch_size)
        sampled_ranges = np.vstack([
            np.arange(start, start + chunk_length) for start in sampled_indexes
        ])

        sampled_ys = self.ys[sampled_ranges].reshape(
            batch_size, chunk_length, self.ys.shape[1]
        )
        sampled_zs = self.zs[sampled_ranges].reshape(
            batch_size, chunk_length, self.zs.shape[1],
        )
        sampled_done = self.done[sampled_ranges].reshape(
            batch_size, chunk_length, 1
        )

        return sampled_ys, sampled_zs, sampled_done