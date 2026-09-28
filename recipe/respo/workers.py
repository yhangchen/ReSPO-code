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
"""Actor worker that registers ReSPO in each Ray process."""

from verl.single_controller.base.decorator import Dispatch, register
from verl.workers.engine_workers import ActorRolloutRefWorker as VerlActorRolloutRefWorker

__all__ = ["ActorRolloutRefWorker"]


def _fsdp_model_is_on_cpu(module) -> bool:
    """Return whether an FSDP module is currently offloaded to CPU."""
    import torch.nn as nn

    if not isinstance(module, nn.Module):
        return False

    from verl.utils.fsdp_utils import fsdp_version

    if fsdp_version(module) == 1:
        handles = getattr(module, "_all_handles", [])
        return bool(handles) and handles[0].flat_param.data.device.type == "cpu"

    parameter = next(module.parameters(), None)
    return parameter is not None and parameter.device.type == "cpu"


def _patch_fsdp_cpu_reload(engine) -> None:
    """Reload an actor offloaded during rollout before its next FSDP pass."""
    from verl.utils.fsdp_utils import load_fsdp_model_to_gpu
    from verl.workers.engine.fsdp.transformer_impl import EngineEvalModeCtx, EngineTrainModeCtx

    class AutoReloadTrainMode(EngineTrainModeCtx):
        def __enter__(self):
            if _fsdp_model_is_on_cpu(self.engine.module):
                load_fsdp_model_to_gpu(self.engine.module)
            return super().__enter__()

    class AutoReloadEvalMode(EngineEvalModeCtx):
        def __enter__(self):
            if _fsdp_model_is_on_cpu(self.engine.module):
                load_fsdp_model_to_gpu(self.engine.module)
            return super().__enter__()

    engine.train_mode = lambda **kwargs: AutoReloadTrainMode(engine, **kwargs)
    engine.eval_mode = lambda **kwargs: AutoReloadEvalMode(engine, **kwargs)


class ActorRolloutRefWorker(VerlActorRolloutRefWorker):
    """Standard verl worker with ReSPO registration and FSDP reload support."""

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        import recipe.respo.core_algos  # noqa: F401

        super().init_model()

        if "actor" in self.role:
            from verl.workers.engine.fsdp.transformer_impl import FSDPEngine

            engine = self.actor.engine
            if isinstance(engine, FSDPEngine) and not engine._is_offload_param:
                _patch_fsdp_cpu_reload(engine)
