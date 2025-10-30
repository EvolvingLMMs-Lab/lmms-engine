# Copyright 2024 Huggingface co. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import math
import os

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh


class ProcessGroupManager:
    def __init__(self, tp_size, cp_size, pp_size, dp_size, ep_size=1):
        self.global_rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.local_rank = int(os.environ.get("LOCAL_RANK", self.global_rank % self.world_size))

        assert (
            self.world_size == tp_size * cp_size * pp_size * dp_size
        ), f"World size ({self.world_size}) != TP ({tp_size}) * CP ({cp_size}) * PP ({pp_size}) * DP ({dp_size})"

        assert pp_size == 1, "PP size must be 1 for now"
        if ep_size > 1:
            assert ep_size % (cp_size * tp_size) == 0 and (dp_size * cp_size * tp_size) % ep_size == 0

        self.tp_size = tp_size
        self.cp_size = cp_size
        self.pp_size = pp_size
        self.dp_size = dp_size
        self.ep_size = ep_size

        self.device_mesh = init_device_mesh(
            "cuda",
            (dp_size, pp_size, cp_size, tp_size),
            mesh_dim_names=["dp", "pp", "cp", "tp"],
        )

        # Currently pp and tp is always 1, so fsdp is the same as dp
        # TODO: support pp > 1
        fsdp_size = dp_size * cp_size

        if ep_size > 1:
            assert self.cp_size == 1, "Currently only support cp = 1 for EP"
            dp_shard_mod_ep = self.dp_size * self.cp_size * self.tp_size // self.ep_size
            dp_shard_in_ep = self.ep_size // (self.cp_size * self.tp_size)
            self.device_mesh = init_device_mesh(
                "cuda",
                (dp_shard_mod_ep, dp_shard_in_ep),
                mesh_dim_names=["dp_shard_mod_ep", "dp_shard_in_ep"],
            )
            self.device_mesh["dp_shard_mod_ep", "dp_shard_in_ep"]._flatten(mesh_dim_name="fsdp")
            self.device_mesh["dp_shard_in_ep"]._flatten(mesh_dim_name="ep")
            self.ep_world_size = ep_size
        else:
            self.device_mesh["dp", "cp"]._flatten(mesh_dim_name="fsdp")
            self.ep_world_size = 1

        self.grid = torch.arange(self.world_size).view(
            dp_size, pp_size, cp_size, tp_size
        )  # DP * PP * CP * TP * EP grid
        # Find the position of the current process in the grid
        self.dp_rank, self.pp_rank, self.cp_rank, self.tp_rank = (
            (self.grid == self.global_rank).nonzero().flatten().tolist()
        )

        # Process group creation - Update indexing to match new grid order
        self.tp_group = dist.new_subgroups_by_enumeration(
            [self.grid[d, p, c, :].tolist() for d in range(dp_size) for p in range(pp_size) for c in range(cp_size)]
        )[0]
        self.cp_group = dist.new_subgroups_by_enumeration(
            [self.grid[d, p, :, t].tolist() for d in range(dp_size) for p in range(pp_size) for t in range(tp_size)]
        )[0]
        self.pp_group = dist.new_subgroups_by_enumeration(
            [self.grid[d, :, c, t].tolist() for d in range(dp_size) for c in range(cp_size) for t in range(tp_size)]
        )[0]
        self.dp_group = dist.new_subgroups_by_enumeration(
            [self.grid[:, p, c, t].tolist() for p in range(pp_size) for c in range(cp_size) for t in range(tp_size)]
        )[0]

        if ep_size > 1:
            self.ep_grid = torch.arange(dp_size).view(dp_shard_mod_ep, dp_shard_in_ep)
            self.ep_group = dist.new_subgroups_by_enumeration(
                [self.ep_grid[d, :].tolist() for d in range(dp_shard_mod_ep)]
            )[0]

        self.world_group = dist.group.WORLD

        # Update group IDs with new grid ordering
        self.tp_group_ids = self.grid[self.dp_rank, self.pp_rank, self.cp_rank, :].tolist()
        self.cp_group_ids = self.grid[self.dp_rank, self.pp_rank, :, self.tp_rank].tolist()
        self.pp_group_ids = self.grid[self.dp_rank, :, self.cp_rank, self.tp_rank].tolist()
        self.dp_group_ids = self.grid[:, self.pp_rank, self.cp_rank, self.tp_rank].tolist()

        # Data parallelism
        self.dp_world_size = dist.get_world_size(group=self.dp_group)
        self.dp_first_rank = self.dp_group_ids[0]
        self.dp_last_rank = self.dp_group_ids[-1]

        self.cp_world_size = dist.get_world_size(group=self.cp_group)
        self.cp_first_rank = self.cp_group_ids[0]
        self.cp_last_rank = self.cp_group_ids[-1]

        self.pp_world_size = dist.get_world_size(group=self.pp_group)
        self.pp_first_rank = self.pp_group_ids[0]
        self.pp_last_rank = self.pp_group_ids[-1]

        self.tp_world_size = dist.get_world_size(group=self.tp_group)
        self.tp_first_rank = self.tp_group_ids[0]
        self.tp_last_rank = self.tp_group_ids[-1]

        if ep_size > 1:
            self.ep_world_size = dist.get_world_size(group=self.ep_group)

    def __str__(self):
        return f"TP({self.tp_size})-CP({self.cp_size})-PP({self.pp_size})-DP({self.dp_size})-EP({self.ep_size})-Rank({self.global_rank})"

    @property
    def enable_tp(self):
        return self.tp_size > 1

    @property
    def enable_cp(self):
        return self.cp_size > 1

    @property
    def enable_pp(self):
        return self.pp_size > 1

    @property
    def enable_ep(self):
        return self.ep_size > 1

    @property
    def enable_parallel(self):
        return self.enable_tp or self.enable_pp or self.enable_ep


def setup_process_group_manager(tp_size, cp_size, pp_size, dp_size, ep_size=1):
    global process_group_manager
    process_group_manager = ProcessGroupManager(tp_size, cp_size, pp_size, dp_size, ep_size)
