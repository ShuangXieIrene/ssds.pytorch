import io
import os
import sys
import glob
import json
import argparse

import torch
import torch.nn as nn

from ssds.core import checkpoint, config
from ssds.modeling import model_builder


class ExportModel(nn.Module):
    def __init__(self, model, nhwc):
        super(ExportModel, self).__init__()
        self.model = model
        self.nhwc = nhwc

    def forward(self, x):
        if self.nhwc:
            x = x.permute(0, 3, 1, 2).contiguous() / 255.0
        return self.model(x)


class Solver(object):
    """
    A wrapper class for the training process
    """

    def __init__(self, cfg, nhwc, render=False):
        self.cfg = cfg
        self.render = render
        self.nhwc = nhwc

        # Build model
        print("===> Building model")
        self.model = model_builder.create_model(cfg.MODEL)
        self.model.eval().cuda()
        self.anchors = model_builder.create_anchors(
            self.cfg.MODEL, self.model, self.cfg.MODEL.IMAGE_SIZE
        )

        # Print the model architecture and parameters
        if self.render:
            print("Model architectures:\n{}\n".format(self.model))
            model_builder.create_anchors(
                self.cfg.MODEL, self.model, self.cfg.MODEL.IMAGE_SIZE, self.render
            )

    def export_onnx(self, weights, export_path, batch):
        if weights != None:
            checkpoint.resume_checkpoint(self.model, weights)
        export_model = ExportModel(self.model, self.nhwc)

        import torch.onnx.symbolic_opset9 as onnx_symbolic

        def upsample_nearest2d(g, input, output_size, *args):
            # Currently, TRT 5.1/6.0 ONNX Parser does not support all ONNX ops
            # needed to support dynamic upsampling ONNX forumlation
            # Here we hardcode scale=2 as a temporary workaround
            scales = g.op("Constant", value_t=torch.tensor([1.0, 1.0, 2.0, 2.0]))
            return g.op("Upsample", input, scales, mode_s="nearest")

        onnx_symbolic.upsample_nearest2d = upsample_nearest2d

        export_model.eval().cuda()
        if self.nhwc:
            dummy_input = torch.rand(
                batch,
                self.cfg.MODEL.IMAGE_SIZE[1],
                self.cfg.MODEL.IMAGE_SIZE[0],
                3,
                requires_grad=False,
            ).cuda()
        else:
            dummy_input = torch.rand(
                batch,
                3,
                self.cfg.MODEL.IMAGE_SIZE[1],
                self.cfg.MODEL.IMAGE_SIZE[0],
                requires_grad=False,
            ).cuda()

        outputs = export_model(dummy_input)
        optional_args = dict(keep_initializers_as_inputs=True)
        input_names = ["input"]
        output_names = [
            n.format(i) for n in ["loc_{}", "conf_{}"] for i in range(len(outputs[0]))
        ]
        if export_path:
            print("Saving model weights & graph to {:s}".format(export_path))
            param = {
                "image_size": self.cfg.MODEL.IMAGE_SIZE,
                "score": self.cfg.POST_PROCESS.SCORE_THRESHOLD,
                "iou": self.cfg.POST_PROCESS.IOU_THRESHOLD,
                "max_detects": self.cfg.POST_PROCESS.MAX_DETECTIONS,
                "max_detects_per_level": self.cfg.POST_PROCESS.MAX_DETECTIONS_PER_LEVEL,
                "rescore": self.cfg.POST_PROCESS.RESCORE_CENTER,
                "use_diou": self.cfg.POST_PROCESS.USE_DIOU,
                "NHWC": self.nhwc,
                "anchors": [v.view(-1).tolist() for k, v in self.anchors.items()],
            }
            with open(export_path + ".json", "w") as output_json:
                json.dump(param, output_json, indent=2)

            torch.onnx.export(
                export_model,
                dummy_input,
                export_path,
                verbose=self.render,
                export_params=True,
                input_names=input_names,
                output_names=output_names,
                **optional_args
            )
            return False
        else:
            onnx_bytes = io.BytesIO()
            torch.onnx.export(
                export_model.cuda(),
                dummy_input,
                onnx_bytes,
                verbose=self.render,
                input_names=input_names,
                output_names=output_names,
            )
            return onnx_bytes

    def export_trt(
        self, weights, export_path, batch, precision, calibration_files, workspace_size
    ):
        if not hasattr(ssds, "_C"):
            raise AssertionError(
                "Currently ssds lib is not install with external cpp plugin,"
                "and cannot export to tensorrt model."
                "Please reinstall the ssds lib by `python setup_cpp.py clean -a install`"
            )
        onnx_bytes = self.export_onnx(weights, None, batch)
        del self.model

        model_name = self.cfg.MODEL.SSDS + "_" + self.cfg.MODEL.NETS
        anchors = [v.view(-1).tolist() for k, v in self.anchors.items()]
        if calibration_files != "":
            calibration_files = glob.glob(calibration_files)
            num_files = (len(calibration_files) // batch) * batch
            calibration_files = calibration_files[:num_files]
        else:
            calibration_files = []
        batch = 1

        from ssds._C import trtConvert

        trtConvert(
            export_path,
            onnx_bytes.getvalue(),
            len(onnx_bytes.getvalue()),
            batch,
            precision,
            self.cfg.POST_PROCESS.SCORE_THRESHOLD,
            self.cfg.POST_PROCESS.MAX_DETECTIONS,
            anchors,
            self.cfg.POST_PROCESS.IOU_THRESHOLD,
            self.cfg.POST_PROCESS.MAX_DETECTIONS,
            calibration_files,
            model_name,
            "",
            self.nhwc,
            self.render,
            workspace_size,
        )
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="export a ssds.pytorch network")
    parser.add_argument(
        "-cfg",
        "--config",
        dest="config_file",
        help="optional config file",
        default=None,
        type=str,
    )
    parser.add_argument(
        "-c", "--checkpoint", help="optional checkpoint file", default=None, type=str
    )
    parser.add_argument("-o", "--onnx", help="output onnx file", default=None, type=str)
    parser.add_argument("-t", "--trt", help="output trt file", default=None, type=str)
    parser.add_argument(
        "-b", "--batch", help="batch size for output model", default=1, type=int
    )
    parser.add_argument(
        "-p",
        "--precision",
        help="precision for output trt model",
        default="FP32",
        choices=["FP32", "FP16", "INT8"],
    )
    parser.add_argument(
        "-i",
        "--image-files",
        help="image files for calibrate output trt model with int8 model",
        default="",
        type=str,
    )
    parser.add_argument(
        "-w",
        "--max-workspace-size",
        help="The max workspace size for output plan file. The final result is"
        "1 << max-workspace-size. Example 30 for 1 GB.",
        type=int,
    )
    parser.add_argument("--nhwc", action="store_true")
    parser.add_argument("-r", "--render", action="store_true")

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()

    cfg = config.cfg_from_file(args.config_file)
    solver = Solver(cfg, args.nhwc, args.render)
    if args.onnx:
        solver.export_onnx(args.checkpoint, args.onnx, args.batch)
    if args.trt:
        solver.export_trt(
            args.checkpoint,
            args.trt,
            args.batch,
            args.precision,
            args.image_files,
            1 << args.max_workspace_size,
        )