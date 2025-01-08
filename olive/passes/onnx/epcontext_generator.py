# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import logging
from pathlib import Path
from typing import Any, Dict

from olive.hardware import AcceleratorSpec
from olive.model import ONNXModelHandler
from olive.model.utils import resolve_onnx_path
from olive.passes.olive_pass import Pass
from olive.passes.pass_config import PassConfigParam

logger = logging.getLogger(__name__)


class QNPUEPContextGenerator(Pass):
    """Create EPContext cache model from a ONNX model with ORT-QNN for Qualcomm NPU."""

    run_on_target = True

    @classmethod
    def _default_config(cls, accelerator_spec: AcceleratorSpec) -> Dict[str, PassConfigParam]:
        return {
            "inference_settings": PassConfigParam(
                type_=dict,
                default_value=None,
                required=False,
                description=("Inference settings for the model. If not provided, default settings are used."),
            ),
            "optimization_mode": PassConfigParam(
                type_=str,
                default_value="3",
                required=False,
                description=(
                    "HTP graph optimization mode. Default is 3.\n"
                    "0:	default.\n"
                    "1:	faster preparation time, less optimal graph.\n"
                    "2: longer preparation time, more optimal graph.\n"
                    "3: longest preparation time, most likely even more"
                    " optimal graph."
                ),
            ),
            "context_embed_mode": PassConfigParam(
                type_=str,
                default_value="0",
                required=False,
                description=("Whether to embed the context in the model. Default is 0."),
            ),
        }

    def _run_for_config(
        self,
        model: ONNXModelHandler,
        config: Dict[str, Any],
        output_model_path: str,
    ) -> ONNXModelHandler:
        self._check_onnx_version()

        # Only support Qualcomm NPU for now
        assert self.accelerator_spec.accelerator_type == "npu"
        assert self.accelerator_spec.execution_provider == "QNNExecutionProvider"

        # TODO(zhengte): parse model instead of checking suffix
        if model.model_path.endswith(".onnx_ctx.onnx"):
            logger.warning("Model already has EPContext. Skip generating EPContext.")
            return model

        # Get full path of the EPContext model to be generated
        onnx_file_name = "model.onnx_ctx.onnx"
        epcontext_model_path = resolve_onnx_path(
            output_model_path,
            model_filename=onnx_file_name,
        )

        # Add EPContext generation required settings to inference_settings
        inference_settings = config["inference_settings"] or {}
        inference_settings = self._merge_inference_settings(
            inference_settings=config["inference_settings"] or {},
            config=config,
            epcontext_model_path=epcontext_model_path,
        )

        # ORT-QNN will generate the EPContext model while initializing the inference session
        model.prepare_session(
            inference_settings=inference_settings,
            device=self.accelerator_spec.accelerator_type,
            execution_providers=self.accelerator_spec.execution_provider,
        )

        # Fallback to original model if EPContext model is not generated
        if not Path(epcontext_model_path).resolve().exists():
            logger.error("Failed to generate EPContext model. Using original model.")
            return model

        # EPContext model has external binary data if embed mode is 0
        # Need to put all artifacts in the a directory for packaging
        # TODO(anyone): Shall we return model as a single artifact if embed mode is 1?
        return ONNXModelHandler(
            model_path=output_model_path,
            onnx_file_name="model.onnx_ctx.onnx",
        )

    def _check_onnx_version(self):
        import onnxruntime as ort
        from packaging import version

        if "QNNExecutionProvider" not in ort.get_available_providers():
            raise RuntimeError("QNNExecutionProvider is not available in the installed ONNXRuntime!")

        # https://onnxruntime.ai/docs/execution-providers/QNN-ExecutionProvider.html#running-a-quantized-model-on-windows-arm64-onnxruntime-qnn-version--1180
        if version.parse(ort.get_version_string()) < version.parse("1.18.0"):
            raise RuntimeError("QNPUEPContextGenerator only supports ONNXRuntime version 1.17.0 or later")

    def _merge_inference_settings(
        self, inference_settings: Dict[str, Any], config: Dict[str, Any], epcontext_model_path: str
    ) -> Dict[str, Any]:
        inference_settings["execution_provider"] = "QNNExecutionProvider"

        if "session_options" not in inference_settings or inference_settings["session_options"] is None:
            inference_settings["session_options"] = {}
        session_options = inference_settings["session_options"]
        if "extra_session_config" not in session_options or session_options["extra_session_config"] is None:
            session_options["extra_session_config"] = {}
        inference_settings["session_options"]["extra_session_config"].update(
            {
                "ep.context_enable": "1",
                "ep.context_embed_mode": config["context_embed_mode"],
                "ep.context_file_path": epcontext_model_path,
            }
        )

        if "provider_options" not in inference_settings or inference_settings["provider_options"] is None:
            inference_settings["provider_options"] = [{}]
        if not isinstance(inference_settings["provider_options"], list) or not isinstance(
            inference_settings["provider_options"][0], dict
        ):
            raise ValueError("provider_options must be a sequence of object")
        inference_settings["provider_options"][0].update(
            {
                "htp_graph_finalization_optimization_mode": config["optimization_mode"],
            }
        )

        return inference_settings
