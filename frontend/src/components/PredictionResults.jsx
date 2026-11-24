import React, { useEffect, useState } from "react";
import {
  AlertTriangle,
  Clock,
  Sparkles,
  BrainCircuit,
  RefreshCw,
  Layers
} from "lucide-react";
import { api } from "../api";

// Canonical HAM10000 7-class disease labels and metadata
const CLASS_METADATA = {
  mel: { name: "Melanoma", fullName: "Malignant Melanoma", risk: "High Risk", color: "var(--error)" },
  bcc: { name: "Basal Cell Carcinoma", fullName: "Basal Cell Carcinoma", risk: "High Risk", color: "var(--error)" },
  akiec: { name: "Actinic Keratosis", fullName: "Actinic Keratosis / Bowen's Disease", risk: "Moderate Risk", color: "var(--warning)" },
  nv: { name: "Melanocytic Nevus", fullName: "Melanocytic Nevus (Benign Mole)", risk: "Low Risk", color: "var(--success)" },
  bkl: { name: "Benign Keratosis", fullName: "Benign Keratosis (Seborrheic)", risk: "Low Risk", color: "var(--text-main)" },
  df: { name: "Dermatofibroma", fullName: "Dermatofibroma (Benign Histiocytoma)", risk: "Low Risk", color: "#6366F1" },
  vasc: { name: "Vascular Lesion", fullName: "Vascular Lesion (Angioma / Granuloma)", risk: "Low Risk", color: "#EC4899" },
};

export default function PredictionResults({
  prediction,
  isLatest = true,
  caseImage = null,
  caseId = null,
  reviewStatus = "pending_review"
}) {
  const [visTab, setVisTab] = useState("overlay");
  const [topNCount, setTopNCount] = useState(3); // 3 or 5
  const [ensembleConstituent, setEnsembleConstituent] = useState("ensemble"); // "ensemble", "mobilenet", "efficientnet"
  const [imageLoading, setImageLoading] = useState(false);
  const [imageError, setImageError] = useState(false);

  useEffect(() => {
    const isVit = prediction?.model_name?.includes("VisionTransformer") || prediction?.model_name?.includes("ViT");
    if (isVit) {
      setVisTab("vit_attn");
    } else {
      setVisTab("overlay");
    }
    setEnsembleConstituent("ensemble");
    setImageError(false);
  }, [prediction?.model_name, prediction?.id]);

  if (!prediction) return null;

  const {
    predicted_class,
    predicted_disease_name,
    confidence_score,
    risk_level,
    class_probabilities = {},
    gradcam_overlay_url,
    gradcam_heatmap_url,
    gradcam_pp_overlay_url,
    saliency_overlay_url,
    vit_attention_overlay_url,
    vit_rollout_overlay_url,
    model_name,
    prediction_timestamp,
    is_suitable_for_ai,
    image_quality_status,
  } = prediction;

  const formattedConfidence = (confidence_score * 100).toFixed(1);

  // Ranked predictions calculation
  const sortedProbEntries = Object.entries(class_probabilities).sort((a, b) => b[1] - a[1]);
  const displayedRankedPredictions = sortedProbEntries.slice(0, topNCount).map(([code, prob]) => ({
    class_label: code,
    disease_name: CLASS_METADATA[code]?.name || code,
    probability: prob,
    risk_level: CLASS_METADATA[code]?.risk || "Low Risk"
  }));

  const currentCaseId = caseId || prediction.case_id;
  const isEnsemble = model_name?.includes("Ensemble") || model_name?.includes("SoftVoting");
  const isCnn = (!isEnsemble && (model_name?.includes("MobileNetV2") || model_name?.includes("EfficientNetB0")));
  const isVit = model_name?.includes("VisionTransformer") || model_name?.includes("ViT");
  const supportsGradcam = isCnn || isEnsemble;
  const supportsSaliency = isCnn || isVit || isEnsemble;

  const baseOverlayUrl = supportsGradcam ? (gradcam_overlay_url || (currentCaseId ? `/api/v1/cases/${currentCaseId}/gradcam/overlay?prediction_id=${prediction.id}` : null)) : null;
  const baseHeatmapUrl = supportsGradcam ? (gradcam_heatmap_url || (currentCaseId ? `/api/v1/cases/${currentCaseId}/gradcam/heatmap?prediction_id=${prediction.id}` : null)) : null;
  const baseGcppUrl = supportsGradcam ? (gradcam_pp_overlay_url || (currentCaseId ? `/api/v1/cases/${currentCaseId}/gradcam-plus-plus/overlay?prediction_id=${prediction.id}` : null)) : null;

  const overlayUrl = (isEnsemble && ensembleConstituent !== "ensemble" && baseOverlayUrl)
    ? `${baseOverlayUrl}&constituent=${ensembleConstituent}`
    : baseOverlayUrl;

  const heatmapUrl = (isEnsemble && ensembleConstituent !== "ensemble" && baseHeatmapUrl)
    ? `${baseHeatmapUrl}&constituent=${ensembleConstituent}`
    : baseHeatmapUrl;

  const gcppOverlayUrl = (isEnsemble && ensembleConstituent !== "ensemble" && baseGcppUrl)
    ? `${baseGcppUrl}&constituent=${ensembleConstituent}`
    : baseGcppUrl;

  const saliencyUrl = supportsSaliency ? (saliency_overlay_url || (currentCaseId ? `/api/v1/cases/${currentCaseId}/saliency/overlay?prediction_id=${prediction.id}` : null)) : null;
  const vitAttentionUrl = isVit ? (vit_rollout_overlay_url || vit_attention_overlay_url || (currentCaseId ? `/api/v1/cases/${currentCaseId}/vit-attention/overlay?prediction_id=${prediction.id}` : null)) : null;
  const roiUrl = prediction.lesion_analysis_url || (currentCaseId ? `/api/v1/cases/${currentCaseId}/lesion-analysis/overlay` : null);

  const getActiveImageUrl = () => {
    switch (visTab) {
      case "overlay":
        return overlayUrl;
      case "heatmap":
        return heatmapUrl;
      case "gcpp":
        return gcppOverlayUrl;
      case "saliency":
        return saliencyUrl;
      case "vit_attn":
        return vitAttentionUrl;
      case "roi":
        return roiUrl;
      case "preprocessed":
        return caseImage?.preprocessed_url;
      case "original":
        return caseImage?.original_url;
      default:
        return overlayUrl || saliencyUrl || vitAttentionUrl || roiUrl;
    }
  };

  const activeImageUrl = getActiveImageUrl();

  const getRiskBadge = (risk) => {
    switch (risk) {
      case "High Risk":
        return <span className="badge badge-high">High Risk</span>;
      case "Moderate Risk":
        return <span className="badge badge-moderate">Moderate Risk</span>;
      default:
        return <span className="badge badge-low">Low Risk / Benign</span>;
    }
  };

  const getReviewBadge = (rStatus) => {
    switch (rStatus) {
      case "reviewed":
        return <span className="badge badge-low">Reviewed</span>;
      case "under_review":
        return <span className="badge badge-teal">Under Review</span>;
      case "follow_up_required":
        return <span className="badge badge-high">Follow-up Required</span>;
      default:
        return <span className="badge badge-awaiting">Pending Review</span>;
    }
  };

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
      {/* Suboptimal Image Warning */}
      {is_suitable_for_ai === false && (
        <div
          style={{
            padding: "12px 16px",
            background: "var(--warning-bg)",
            border: "1px solid var(--warning-border)",
            borderRadius: "var(--radius-md)",
            color: "var(--warning-text)",
            display: "flex",
            alignItems: "flex-start",
            gap: 10,
            fontSize: "0.82rem",
            lineHeight: 1.45,
          }}
        >
          <AlertTriangle size={17} style={{ flexShrink: 0, marginTop: 1 }} />
          <div>
            <strong>Image Quality Advisory: </strong>
            Photograph was flagged as <em>{image_quality_status || "suboptimal"}</em> during quality verification. Model predictions carry higher diagnostic uncertainty.
          </div>
        </div>
      )}

      {/* Primary Finding Card */}
      <div
        className="medical-card"
        style={{
          padding: 20,
          borderLeft: risk_level === "High Risk" ? "4px solid var(--error)" : risk_level === "Moderate Risk" ? "4px solid var(--warning)" : "4px solid var(--success)",
        }}
      >
        <div
          style={{
            display: "flex",
            justifyContent: "space-between",
            alignItems: "flex-start",
            flexWrap: "wrap",
            gap: 12,
            marginBottom: 16,
            paddingBottom: 14,
            borderBottom: "1px solid var(--border-subtle)",
          }}
        >
          <div>
            <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 4, flexWrap: "wrap" }}>
              <span className="badge badge-teal" style={{ textTransform: "uppercase" }}>
                Deep Learning Screening Finding
              </span>
              <span style={{ fontSize: "0.75rem", color: "var(--text-muted)" }}>
                Model: <strong>{model_name || "DermAI_MobileNetV2"}</strong>
              </span>
              {isLatest && <span className="badge badge-low" style={{ fontSize: "0.68rem" }}>Latest Inference</span>}
              {getReviewBadge(reviewStatus)}
            </div>
            <h2 style={{ fontSize: "1.4rem", fontWeight: 800, color: "var(--text-main)" }}>
              {predicted_disease_name || CLASS_METADATA[predicted_class]?.fullName || predicted_class}
            </h2>
            <div style={{ fontSize: "0.8rem", color: "var(--text-secondary)", marginTop: 2 }}>
              Classification Taxonomy: <strong style={{ fontFamily: "var(--font-mono)", color: "var(--text-main)" }}>{predicted_class}</strong>
            </div>
          </div>

          <div style={{ display: "flex", flexDirection: "column", alignItems: "flex-end", gap: 4 }}>
            {getRiskBadge(risk_level)}
            <span style={{ fontSize: "0.72rem", color: "var(--text-muted)", display: "flex", alignItems: "center", gap: 4 }}>
              <Clock size={12} />
              {prediction_timestamp ? new Date(prediction_timestamp).toLocaleDateString() : ""}
            </span>
          </div>
        </div>

        {/* Primary Metrics Grid */}
        <div
          style={{
            display: "grid",
            gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))",
            gap: 12,
          }}
        >
          <div
            style={{
              background: "var(--bg-surface-subtle)",
              padding: "12px 14px",
              borderRadius: "var(--radius-sm)",
              border: "1px solid var(--border-color)",
            }}
          >
            <span style={{ fontSize: "0.68rem", color: "var(--text-muted)", textTransform: "uppercase", fontWeight: 700 }}>
              Model Softmax Score
            </span>
            <div style={{ display: "flex", alignItems: "baseline", gap: 6, marginTop: 2 }}>
              <span style={{ fontSize: "1.6rem", fontWeight: 800, color: "var(--text-main)" }}>
                {formattedConfidence}%
              </span>
              <span style={{ fontSize: "0.75rem", color: "var(--text-secondary)" }}>
                ({confidence_score?.toFixed(4)})
              </span>
            </div>
            <p style={{ fontSize: "0.7rem", color: "var(--text-muted)", marginTop: 2 }}>
              Relative model confidence
            </p>
          </div>

          <div
            style={{
              background: "var(--bg-surface-subtle)",
              padding: "12px 14px",
              borderRadius: "var(--radius-sm)",
              border: "1px solid var(--border-color)",
            }}
          >
            <span style={{ fontSize: "0.68rem", color: "var(--text-muted)", textTransform: "uppercase", fontWeight: 700 }}>
              Clinical Risk Level
            </span>
            <div style={{ fontSize: "1.2rem", fontWeight: 800, marginTop: 4, color: risk_level === "High Risk" ? "var(--error-text)" : risk_level === "Moderate Risk" ? "var(--warning-text)" : "var(--success-text)" }}>
              {risk_level || "Standard Priority"}
            </div>
            <p style={{ fontSize: "0.7rem", color: "var(--text-muted)", marginTop: 2 }}>
              {risk_level === "High Risk"
                ? "Immediate clinical evaluation & biopsy recommended"
                : risk_level === "Moderate Risk"
                ? "Formal clinical examination advised"
                : "Benign morphology; clinical surveillance"}
            </p>
          </div>

          <div
            style={{
              background: "var(--bg-surface-subtle)",
              padding: "12px 14px",
              borderRadius: "var(--radius-sm)",
              border: "1px solid var(--border-color)",
            }}
          >
            <span style={{ fontSize: "0.68rem", color: "var(--text-muted)", textTransform: "uppercase", fontWeight: 700 }}>
              Input Standardization
            </span>
            <div style={{ fontSize: "1.2rem", fontWeight: 800, color: is_suitable_for_ai === false ? "var(--error-text)" : "var(--success-text)", marginTop: 4 }}>
              {is_suitable_for_ai === false ? "Suboptimal" : "Validated (224×224)"}
            </div>
            <p style={{ fontSize: "0.7rem", color: "var(--text-muted)", marginTop: 2 }}>
              Status: {image_quality_status || "Standardized"}
            </p>
          </div>
        </div>
      </div>

      {/* Explainable AI Visualizer Stage */}
      <div className="medical-card" style={{ padding: 20 }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", flexWrap: "wrap", gap: 12, marginBottom: 14 }}>
          <div>
            <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
              <BrainCircuit size={18} color="var(--primary-hover)" />
              <h3 style={{ fontSize: "1.05rem", fontWeight: 700, color: "var(--text-main)" }}>
                Explainable AI (XAI) Visualizer
              </h3>
            </div>
            <p style={{ fontSize: "0.8rem", color: "var(--text-secondary)", marginTop: 2 }}>
              Neural feature attribution mapping revealing lesion regions influencing prediction
            </p>
          </div>

          {/* Visualization Tabs */}
          <div
            style={{
              display: "flex",
              flexWrap: "wrap",
              gap: 4,
              background: "var(--bg-surface-subtle)",
              padding: 3,
              borderRadius: "var(--radius-md)",
              border: "1px solid var(--border-color)",
            }}
          >
            {overlayUrl && (
              <button
                type="button"
                onClick={() => setVisTab("overlay")}
                className={`btn btn-sm ${visTab === "overlay" ? "btn-primary" : "btn-ghost"}`}
                style={{ fontSize: "0.74rem", padding: "4px 8px", height: 28 }}
              >
                Grad-CAM
              </button>
            )}
            {heatmapUrl && (
              <button
                type="button"
                onClick={() => setVisTab("heatmap")}
                className={`btn btn-sm ${visTab === "heatmap" ? "btn-primary" : "btn-ghost"}`}
                style={{ fontSize: "0.74rem", padding: "4px 8px", height: 28 }}
              >
                Heatmap
              </button>
            )}
            {gcppOverlayUrl && (
              <button
                type="button"
                onClick={() => setVisTab("gcpp")}
                className={`btn btn-sm ${visTab === "gcpp" ? "btn-primary" : "btn-ghost"}`}
                style={{ fontSize: "0.74rem", padding: "4px 8px", height: 28 }}
              >
                Grad-CAM++
              </button>
            )}
            {saliencyUrl && (
              <button
                type="button"
                onClick={() => setVisTab("saliency")}
                className={`btn btn-sm ${visTab === "saliency" ? "btn-primary" : "btn-ghost"}`}
                style={{ fontSize: "0.74rem", padding: "4px 8px", height: 28 }}
              >
                Saliency
              </button>
            )}
            {vitAttentionUrl && (
              <button
                type="button"
                onClick={() => setVisTab("vit_attn")}
                className={`btn btn-sm ${visTab === "vit_attn" ? "btn-primary" : "btn-ghost"}`}
                style={{ fontSize: "0.74rem", padding: "4px 8px", height: 28 }}
              >
                Attention Rollout
              </button>
            )}
            {roiUrl && (
              <button
                type="button"
                onClick={() => setVisTab("roi")}
                className={`btn btn-sm ${visTab === "roi" ? "btn-primary" : "btn-ghost"}`}
                style={{ fontSize: "0.74rem", padding: "4px 8px", height: 28 }}
              >
                ROI Candidate
              </button>
            )}
            {caseImage?.preprocessed_url && (
              <button
                type="button"
                onClick={() => setVisTab("preprocessed")}
                className={`btn btn-sm ${visTab === "preprocessed" ? "btn-primary" : "btn-ghost"}`}
                style={{ fontSize: "0.74rem", padding: "4px 8px", height: 28 }}
              >
                224px Tensor
              </button>
            )}
            {caseImage?.original_url && (
              <button
                type="button"
                onClick={() => setVisTab("original")}
                className={`btn btn-sm ${visTab === "original" ? "btn-primary" : "btn-ghost"}`}
                style={{ fontSize: "0.74rem", padding: "4px 8px", height: 28 }}
              >
                Original Photo
              </button>
            )}
          </div>
        </div>

        {/* Visualizer Display Stage */}
        <div
          style={{
            display: "grid",
            gridTemplateColumns: "minmax(260px, 340px) 1fr",
            gap: 16,
            alignItems: "center",
            background: "var(--bg-surface-subtle)",
            padding: 16,
            borderRadius: "var(--radius-md)",
            border: "1px solid var(--border-color)",
          }}
          className="xai-stage-grid"
        >
          <div
            style={{
              width: "100%",
              aspectRatio: "1/1",
              maxHeight: 320,
              borderRadius: "var(--radius-sm)",
              overflow: "hidden",
              background: "#0F172A",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              border: "1px solid var(--border-color)",
              position: "relative",
            }}
          >
            {activeImageUrl ? (
              <img
                key={activeImageUrl}
                src={api.getImageUrl(activeImageUrl)}
                alt={visTab}
                onLoad={() => setImageLoading(false)}
                onError={() => {
                  setImageLoading(false);
                  setImageError(true);
                }}
                style={{
                  maxWidth: "100%",
                  maxHeight: "100%",
                  objectFit: "contain",
                  display: imageError ? "none" : "block",
                }}
              />
            ) : null}

            {imageLoading && !imageError && (
              <div
                style={{
                  position: "absolute",
                  color: "#94A3B8",
                  fontSize: "0.75rem",
                  display: "flex",
                  alignItems: "center",
                  gap: 6,
                }}
              >
                <RefreshCw size={14} className="spin" />
                <span>Loading visualization...</span>
              </div>
            )}

            {imageError && (
              <div
                style={{
                  color: "var(--error-text)",
                  fontSize: "0.78rem",
                  textAlign: "center",
                  padding: 16,
                  display: "flex",
                  flexDirection: "column",
                  alignItems: "center",
                  gap: 6,
                }}
              >
                <AlertTriangle size={18} />
                <span>Visualization unavailable for this selection.</span>
              </div>
            )}

            {!activeImageUrl && !imageError && (
              <div style={{ color: "#64748B", fontSize: "0.78rem" }}>
                No explanation available
              </div>
            )}
          </div>

          <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
            <span className="badge badge-teal" style={{ alignSelf: "flex-start", fontSize: "0.72rem" }}>
              {visTab === "overlay" && (
                isEnsemble ? (
                  ensembleConstituent === "mobilenet" ? "Constituent Grad-CAM: MobileNetV2 (out_relu)" :
                  ensembleConstituent === "efficientnet" ? "Constituent Grad-CAM: EfficientNetB0 (top_activation)" :
                  "Soft-Voting Ensemble Grad-CAM (Combined Consensus)"
                ) : (
                  model_name?.includes("EfficientNetB0")
                    ? "Grad-CAM Convolutional Heatmap (EfficientNetB0 - top_activation)"
                    : "Grad-CAM Convolutional Heatmap (MobileNetV2 - out_relu)"
                )
              )}
              {visTab === "heatmap" && (
                isEnsemble
                  ? `Raw Gradient Heatmap (${ensembleConstituent === "mobilenet" ? "MobileNetV2" : ensembleConstituent === "efficientnet" ? "EfficientNetB0" : "Ensemble Consensus"})`
                  : "Raw Gradient Heatmap"
              )}
              {visTab === "gcpp" && (
                isEnsemble
                  ? `Grad-CAM++ Second-Order Attribution (${ensembleConstituent === "mobilenet" ? "MobileNetV2" : ensembleConstituent === "efficientnet" ? "EfficientNetB0" : "Ensemble Consensus"})`
                  : "Grad-CAM++ Second-Order Attribution"
              )}
              {visTab === "saliency" && (
                isEnsemble ? "Ensemble Gradient Saliency Map (Analytical Backpropagation)" : "Gradient Saliency Map"
              )}
              {visTab === "vit_attn" && "ViT Attention Rollout"}
              {visTab === "roi" && "Classical CV Color-Contrast ROI Candidate"}
              {visTab === "preprocessed" && "CLAHE Standardized 224×224 Tensor"}
              {visTab === "original" && "Original Unprocessed Clinical Photograph"}
            </span>

            {/* Ensemble Constituent Selector */}
            {isEnsemble && (visTab === "overlay" || visTab === "heatmap" || visTab === "gcpp") && (
              <div
                style={{
                  display: "flex",
                  alignItems: "center",
                  flexWrap: "wrap",
                  gap: 4,
                  background: "var(--bg-surface-subtle)",
                  padding: "4px 8px",
                  borderRadius: "var(--radius-sm)",
                  border: "1px solid var(--border-color)",
                }}
              >
                <span style={{ fontSize: "0.68rem", color: "var(--text-muted)", fontWeight: 700, marginRight: 2, display: "flex", alignItems: "center", gap: 3 }}>
                  <Layers size={11} /> Constituent:
                </span>
                <button
                  type="button"
                  onClick={() => {
                    setEnsembleConstituent("ensemble");
                    setImageError(false);
                  }}
                  className={`btn btn-sm ${ensembleConstituent === "ensemble" ? "btn-secondary" : "btn-ghost"}`}
                  style={{ fontSize: "0.68rem", padding: "2px 7px", height: 22 }}
                >
                  Ensemble Consensus
                </button>
                <button
                  type="button"
                  onClick={() => {
                    setEnsembleConstituent("mobilenet");
                    setImageError(false);
                  }}
                  className={`btn btn-sm ${ensembleConstituent === "mobilenet" ? "btn-secondary" : "btn-ghost"}`}
                  style={{ fontSize: "0.68rem", padding: "2px 7px", height: 22 }}
                >
                  MobileNetV2 (out_relu)
                </button>
                <button
                  type="button"
                  onClick={() => {
                    setEnsembleConstituent("efficientnet");
                    setImageError(false);
                  }}
                  className={`btn btn-sm ${ensembleConstituent === "efficientnet" ? "btn-secondary" : "btn-ghost"}`}
                  style={{ fontSize: "0.68rem", padding: "2px 7px", height: 22 }}
                >
                  EfficientNetB0 (top_activation)
                </button>
              </div>
            )}

            <p style={{ fontSize: "0.82rem", color: "var(--text-secondary)", lineHeight: 1.5 }}>
              {visTab === "overlay" && (
                isEnsemble ? (
                  ensembleConstituent === "mobilenet"
                    ? "Gradient-weighted activation on constituent MobileNetV2 features at convolutional layer 'out_relu'."
                    : ensembleConstituent === "efficientnet"
                    ? "Gradient-weighted activation on constituent EfficientNetB0 features at convolutional layer 'top_activation'."
                    : "Soft-Voting Ensemble Grad-CAM combines spatial feature activations from constituent CNN models: MobileNetV2 (layer 'out_relu') and EfficientNetB0 (layer 'top_activation'), weighted by soft-voting component consensus."
                ) : (
                  model_name?.includes("EfficientNetB0")
                    ? "Gradient-weighted activation highlights dermoscopic structures that drove neural feature extraction in layer 'top_activation'."
                    : "Gradient-weighted activation highlights dermoscopic structures that drove neural feature extraction in layer 'out_relu'."
                )
              )}
              {visTab === "heatmap" && (
                isEnsemble
                  ? "Thermal gradient activation map reflecting spatial consensus across constituent convolutional layers."
                  : "Thermal gradient map showing normalized positive gradient flow across the spatial feature map."
              )}
              {visTab === "gcpp" && (
                isEnsemble
                  ? "Ensemble Grad-CAM++ aggregates second-order gradient attributions across constituent MobileNetV2 and EfficientNetB0 backbones."
                  : "Second-order pixel gradients weighting multiple overlapping lesion clusters."
              )}
              {visTab === "saliency" && (
                isEnsemble
                  ? "Analytical pixel-level gradient attribution calculated end-to-end through the soft-voting ensemble differentiation with respect to input pixels."
                  : "Direct pixel-level gradients with respect to predicted lesion score."
              )}
              {visTab === "vit_attn" && (
                "Attention Rollout recursively accumulates multi-head self-attention across all 3 Transformer blocks, accounting for residual skip connections and global token pooling across the 14×14 patch grid."
              )}
              {visTab === "preprocessed" && (
                "Standardized 224×224 tensor input after Contrast-Limited Adaptive Histogram Equalization and bilateral filtering."
              )}
              {visTab === "original" && (
                "Original clinical photograph retained for longitudinal comparison and physical verification."
              )}
              {visTab === "roi" && (
                "Classical computer vision color-contrast candidate region. Not a trained segmentation boundary."
              )}
            </p>

            <div
              style={{
                background: "#FFFFFF",
                padding: "10px 12px",
                borderRadius: "var(--radius-sm)",
                border: "1px solid var(--border-color)",
                fontSize: "0.78rem",
              }}
            >
              <div style={{ color: "var(--text-muted)" }}>Target Classification:</div>
              <strong style={{ color: "var(--text-main)" }}>
                {predicted_disease_name} ({predicted_class})
              </strong>
              <div style={{ color: "var(--text-secondary)", marginTop: 2 }}>
                Model: <strong>{model_name || "Deep Learning Classifier"}</strong>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Top Ranked Hypotheses */}
      <div className="medical-card" style={{ padding: 20 }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 14 }}>
          <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
            <Sparkles size={16} color="var(--primary-hover)" />
            <h3 style={{ fontSize: "1.05rem", fontWeight: 700, color: "var(--text-main)" }}>
              Top-{topNCount} Ranked Hypotheses
            </h3>
          </div>
          <div
            style={{
              display: "flex",
              gap: 4,
              background: "var(--bg-surface-subtle)",
              padding: 2,
              borderRadius: "var(--radius-sm)",
              border: "1px solid var(--border-color)",
            }}
          >
            <button
              type="button"
              onClick={() => setTopNCount(3)}
              className={`btn btn-sm ${topNCount === 3 ? "btn-secondary" : "btn-ghost"}`}
              style={{ padding: "3px 8px", height: 26, fontSize: "0.72rem" }}
            >
              Top-3
            </button>
            <button
              type="button"
              onClick={() => setTopNCount(5)}
              className={`btn btn-sm ${topNCount === 5 ? "btn-secondary" : "btn-ghost"}`}
              style={{ padding: "3px 8px", height: 26, fontSize: "0.72rem" }}
            >
              Top-5
            </button>
          </div>
        </div>

        <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
          {displayedRankedPredictions.map((item, idx) => {
            const pct = (item.probability * 100).toFixed(1);
            return (
              <div
                key={item.class_label}
                style={{
                  background: idx === 0 ? "var(--primary-light)" : "var(--bg-surface-subtle)",
                  border: idx === 0 ? "1px solid var(--primary-light-border)" : "1px solid var(--border-color)",
                  borderRadius: "var(--radius-sm)",
                  padding: "10px 14px",
                }}
              >
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 6 }}>
                  <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                    <span className="badge badge-teal" style={{ fontSize: "0.68rem" }}>
                      #{idx + 1}
                    </span>
                    <strong style={{ fontSize: "0.9rem", color: "var(--text-main)" }}>
                      {item.disease_name}
                    </strong>
                    <span style={{ fontSize: "0.75rem", fontFamily: "var(--font-mono)", color: "var(--text-muted)" }}>
                      ({item.class_label})
                    </span>
                  </div>

                  <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                    {getRiskBadge(item.risk_level)}
                    <span style={{ fontFamily: "var(--font-mono)", fontSize: "0.85rem", fontWeight: 700, color: "var(--text-main)" }}>
                      {pct}%
                    </span>
                  </div>
                </div>

                {/* Progress bar */}
                <div
                  style={{
                    height: 6,
                    background: "#FFFFFF",
                    borderRadius: 3,
                    overflow: "hidden",
                    border: "1px solid var(--border-color)",
                  }}
                >
                  <div
                    style={{
                      height: "100%",
                      width: `${pct}%`,
                      background: idx === 0 ? "var(--primary-hover)" : "var(--text-muted)",
                      borderRadius: 3,
                    }}
                  />
                </div>
              </div>
            );
          })}
        </div>
      </div>

      <style>{`
        @media (max-width: 768px) {
          .xai-stage-grid {
            grid-template-columns: 1fr !important;
          }
        }
      `}</style>
    </div>
  );
}
