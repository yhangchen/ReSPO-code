# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2026 ReSPO Authors
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
"""Minimal verl entry point that installs ReSPO's actor worker."""

import hydra
import ray

from verl.experimental.reward_loop import migrate_legacy_reward_impl
from verl.single_controller.ray import RayWorkerGroup
from verl.trainer.main_ppo import TaskRunner as VerlTaskRunner
from verl.trainer.main_ppo import run_ppo
from verl.trainer.ppo.ray_trainer import Role
from verl.trainer.ppo.utils import need_reference_policy
from verl.utils.device import auto_set_device


class TaskRunner(VerlTaskRunner):
    """Use the standard verl trainer with a worker that registers ReSPO."""

    def add_actor_rollout_worker(self, config):
        from recipe.respo.code.workers import ActorRolloutRefWorker

        lora_rank = config.actor_rollout_ref.model.get("lora", {}).get("rank", 0)
        if lora_rank <= 0:
            lora_rank = config.actor_rollout_ref.model.get("lora_rank", 0)
        ref_in_actor = lora_rank > 0 or config.actor_rollout_ref.model.get("lora_adapter_path") is not None
        role = Role.ActorRolloutRef if need_reference_policy(config) and not ref_in_actor else Role.ActorRollout

        self.role_worker_mapping[role] = ray.remote(ActorRolloutRefWorker)
        self.mapping[role] = "global_pool"
        return ActorRolloutRefWorker, RayWorkerGroup


@hydra.main(config_path="../../../verl/trainer/config", config_name="ppo_trainer", version_base=None)
def main(config):
    auto_set_device(config)
    config = migrate_legacy_reward_impl(config)
    runner = ray.remote(num_cpus=1)(TaskRunner)
    run_ppo(config, task_runner_class=runner)


if __name__ == "__main__":
    main()
