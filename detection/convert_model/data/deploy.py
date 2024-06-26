onnx_config = dict(
    type="onnx",
    export_params=True,
    keep_initializers_as_inputs=False,
    opset_version=11,
    save_file="end2end.onnx",
    input_names=["input"],
    output_names=["dets", "labels"],
    input_shape=(768, 1280),
    optimize=True,
)
codebase_config = dict(
    type="mmyolo",
    task="ObjectDetection",
    model_type="end2end",
    post_processing=dict(
        score_threshold=0.05,
        confidence_threshold=0.005,
        iou_threshold=0.5,
        max_output_boxes_per_class=200,
        pre_top_k=5000,
        keep_top_k=100,
        background_label_id=-1,
    ),
    module=["mmyolo.deploy"],
)
backend_config = dict(
    type="tensorrt",
    common_config=dict(fp16_mode=True, max_workspace_size=1 << 30),
    model_inputs=[
        dict(
            input_shapes=dict(
                input=dict(min_shape=[1, 3, 1280, 768], opt_shape=[1, 3, 1280, 768], max_shape=[1, 3, 1280, 768])
            )
        )
    ],
)
use_efficientnms = False  # whether to replace TRTBatchedNMS plugin with EfficientNMS plugin # noqa E501
