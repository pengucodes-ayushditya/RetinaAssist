"""Run with: streamlit run app.py"""
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import torch
from PIL import Image

from src.demo_inference import (
    choose_device,
    load_grade5_model,
    load_segmentation_model,
    predict_grade5,
    predict_segmentation,
)
from src.gradcam import gradcam_overlay


GRADE_CHECKPOINT = Path("models/aptos_grade5_best.pt")
SEGMENTATION_CHECKPOINT = Path("models/idrid_segmentation_best.pt")
THRESHOLDS = Path("outputs/aptos_grade5_thresholds.json")


@st.cache_resource
def cached_grade_model(checkpoint_path, device_name):
    return load_grade5_model(checkpoint_path, torch.device(device_name))


@st.cache_resource
def cached_segmentation_model(checkpoint_path, device_name):
    return load_segmentation_model(checkpoint_path, torch.device(device_name))


st.set_page_config(page_title="RetinaAssist", page_icon="👁", layout="wide")
st.title("RetinaAssist")
st.subheader("Retinal screening research demo")
st.caption(
    "Upload a fundus image to view experimental image-quality, grading, "
    "lesion-segmentation, and explainability outputs."
)
st.warning(
    "**Research prototype only — not a medical diagnosis or medical device.** "
    "Do not use this demo to diagnose, rule out disease, or make treatment decisions."
)

with st.expander("Model scope and limitations"):
    st.write(
        "The five-grade ResNet18 model was trained on APTOS. When the local "
        "outputs/aptos_grade5_thresholds.json file is present, the demo uses "
        "thresholds selected on the APTOS validation split; otherwise it falls back "
        "to 0.5 and labels that source in the report. On the official IDRiD grading "
        "test (103 images), QWK was 0.596; at the APTOS-validation any-DR threshold, "
        "sensitivity was 0.855 and specificity was 0.559. These thresholds may not "
        "transfer to other cameras or populations."
    )
    st.write(
        "The segmentation overlay comes from a separate IDRiD-trained model. "
        "It is experimental visual evidence, not a confirmed lesion localization. "
        "Grad-CAM shows image regions that influenced a model output; it is not a "
        "lesion mask."
    )

uploaded = st.file_uploader(
    "Upload a retinal fundus image", type=["png", "jpg", "jpeg"]
)

if not GRADE_CHECKPOINT.is_file():
    st.error(f"Five-grade checkpoint not found: {GRADE_CHECKPOINT}")
elif uploaded is not None:
    rgb = np.asarray(Image.open(uploaded).convert("RGB"))
    input_col, report_col = st.columns([1, 1])

    with input_col:
        st.image(rgb, caption="Uploaded fundus image", use_container_width=True)

    try:
        device_name = str(choose_device())
        grade_model, grade_checkpoint = cached_grade_model(
            str(GRADE_CHECKPOINT), device_name
        )
        report = predict_grade5(
            rgb,
            grade_model,
            size=int(grade_checkpoint.get("size", 224)),
            thresholds_path=str(THRESHOLDS),
        )

        with report_col:
            st.header("Demo diagnosis")
            quality = report["quality"]

            st.subheader("Image quality")
            st.write(f"**Assessment:** {quality['reason']}")
            q1, q2, q3 = st.columns(3)
            q1.metric("Sharpness", f"{quality['sharpness']:.2f}")
            q2.metric("Brightness", f"{quality['brightness']:.1f}")
            q3.metric("Illuminated area", f"{quality['coverage']:.1%}")

            if quality.get("warnings"):
                st.warning("Image-quality warnings: " + ", ".join(quality["warnings"]))
            else:
                st.success("No image-quality warnings detected.")

            if report["status"] == "quality_review":
                st.error(
                    report["message"]
                    + " Please retake the image or request clinical assessment."
                )
            else:
                grade_name = report["grade_name"]
                grade_score = report["grade_score"]
                any_signal = report["any_dr_signal"]
                referable_signal = report["referable_dr_signal"]

                st.markdown(
                    f"**Model's highest-scoring grade:** {grade_name} "
                    f"(score {grade_score:.1%})"
                )

                if referable_signal:
                    st.error(
                        "Demo screening signal: possible referable DR. "
                        "Clinical assessment is advised."
                    )
                elif any_signal:
                    st.warning(
                        "Demo screening signal: possible DR below the model's "
                        "referable threshold. Clinical assessment is still required."
                    )
                else:
                    st.info(
                        "No DR signal at the selected demo threshold. "
                        "This result does not rule out DR."
                    )

                st.subheader("Model outputs")
                st.metric(
                    "Any-DR raw model score (0–1)",
                    f"{report['any_dr_score']:.6f}",
                    help="Uncalibrated model score; not a clinical probability.",
                )
                st.caption(
                    f"Threshold: {report['thresholds']['any_dr']:.3f} "
                    f"(selected using {report['threshold_source']})."
                )
                st.metric(
                    "Referable-DR raw score (0–1)",
                    f"{report['referable_score']:.6f}",
                    help="Uncalibrated model score; not a clinical probability.",
                )
                st.caption(
                    "Threshold: "
                    f"{report['thresholds']['referable_dr_grade_ge_2']:.3f} "
                    f"(selected using {report['threshold_source']})."
                )

                st.subheader("Five-grade score distribution")
                grade_order = ["No DR", "Mild", "Moderate", "Severe", "Proliferative"]
                grade_scores = pd.DataFrame(
                    {
                        "Raw model score": [
                            report["grade_probabilities"][grade]
                            for grade in grade_order
                        ]
                    },
                    index=grade_order,
                )
                st.bar_chart(grade_scores)
                st.caption(
                    "Scores are uncalibrated softmax outputs, not clinical probabilities."
                )

                st.subheader("Visual evidence")
                show_segmentation = st.checkbox(
                    "Run experimental lesion and optic-disc overlay", value=True
                )
                if show_segmentation:
                    if SEGMENTATION_CHECKPOINT.is_file():
                        with st.spinner("Generating tiled segmentation overlay..."):
                            segmentation_model, _ = cached_segmentation_model(
                                str(SEGMENTATION_CHECKPOINT), device_name
                            )
                            segmentation = predict_segmentation(
                                rgb, segmentation_model
                            )
                        overlay = Image.fromarray(segmentation["overlay"])
                        overlay.thumbnail((1600, 1200))
                        st.image(
                            overlay,
                            caption=(
                                "Experimental segmentation overlay: "
                                "MA, HE, EX, SE, and optic disc"
                            ),
                            use_container_width=True,
                        )
                        st.write("Predicted area by class (% of full image):")
                        class_names = {
                            "MA": "Microaneurysms",
                            "HE": "Haemorrhages",
                            "EX": "Hard exudates",
                            "SE": "Soft exudates",
                            "OD": "Optic disc",
                        }
                        st.write(
                            "Color key: **MA** microaneurysms · **HE** haemorrhages · "
                            "**EX** hard exudates · **SE** soft exudates · **OD** optic disc"
                        )
                        st.table(
                            pd.DataFrame([
                                {
                                    "Structure": class_names[code],
                                    "Predicted area (%)": segmentation[
                                        "area_percentages"
                                    ][code],
                                }
                                for code in ["MA", "HE", "EX", "SE", "OD"]
                            ])
                        )
                        st.caption(
                            "These regions are model predictions, not verified "
                            "clinical findings."
                        )
                    else:
                        st.warning(
                            f"Segmentation checkpoint not found: "
                            f"{SEGMENTATION_CHECKPOINT}"
                        )

                if st.checkbox("Show Grad-CAM for the predicted grade"):
                    with st.spinner("Generating Grad-CAM..."):
                        cam = gradcam_overlay(
                            grade_model, rgb, class_index=report["grade_index"]
                        )
                    st.image(
                        cam,
                        caption=(
                            "Grad-CAM for the selected grade. Highlighted regions "
                            "show model influence, not lesion localization."
                        ),
                        use_container_width=True,
                    )

                st.subheader("Demo summary")
                st.write(
                    f"The image passed the retinal-area quality check. The model's "
                    f"highest-scoring grade is **{grade_name}**. Its any-DR score is "
                    f"{report['any_dr_score']:.1%}, and its referable-DR score is "
                    f"{report['referable_score']:.1%}. "
                    "These outputs are for software demonstration and research review "
                    "only; a qualified clinician must interpret the image."
                )

    except Exception as exc:
        st.error(f"Unable to process this image: {exc}")

st.divider()
st.caption(
    "RetinaAssist research prototype. No downloadable report is generated. "
    "Never use this demo as a substitute for professional eye care."
)
