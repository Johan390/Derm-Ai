import React, { useState } from "react";
import {
  Camera,
  Sun,
  AlertTriangle,
  BookOpen,
  Layers,
  Sliders
} from "lucide-react";

export default function ClinicalGuidelines() {
  const [activeSection, setActiveSection] = useState("protocol");

  const conditions = [
    {
      code: "MEL",
      name: "Melanoma",
      risk: "High Risk",
      riskBadgeClass: "badge-high",
      summary: "Malignant neoplasm arising from epidermal melanocytes. Deadliest form of cutaneous cancer requiring prompt dermatologic excision and histological staging.",
      clinicalFeatures: "Asymmetry, irregular or notched borders, color variegation (dark brown, black, blue, red, white), diameter typically >6mm, and evolutionary changes.",
    },
    {
      code: "BCC",
      name: "Basal Cell Carcinoma",
      risk: "High Risk",
      riskBadgeClass: "badge-high",
      summary: "Most common cutaneous malignancy worldwide. Slow-growing, locally invasive epidermal tumor originating from basal keratinocytes.",
      clinicalFeatures: "Pearly translucent papule or plaque, arborizing telangiectasia (visible fine vessels), rolled borders, central ulceration, or chronic non-healing erosion.",
    },
    {
      code: "AKIEC",
      name: "Actinic Keratosis / Bowen's Disease",
      risk: "Moderate Risk",
      riskBadgeClass: "badge-moderate",
      summary: "Premalignant dysplastic epidermal lesions induced by cumulative ultraviolet radiation with potential for progression to invasive squamous cell carcinoma (SCC).",
      clinicalFeatures: "Rough, erythematous, sandpaper-textured macules or plaques on chronically sun-exposed areas (scalp, face, dorsal hands, forearms).",
    },
    {
      code: "NV",
      name: "Melanocytic Nevus",
      risk: "Low Risk",
      riskBadgeClass: "badge-low",
      summary: "Common benign proliferations of pigment-producing melanocytes ('moles'). Accounts for approximately 67% of benign baseline dermatological presentations.",
      clinicalFeatures: "Geometric symmetry, regular circumscribed borders, uniform light-to-medium brown pigmentation, and stability across serial observations.",
    },
    {
      code: "BKL",
      name: "Benign Keratosis",
      risk: "Low Risk",
      riskBadgeClass: "badge-low",
      summary: "Common benign non-melanocytic epidermal lesions occurring in mature adults. Includes seborrheic keratosis, solar lentigo, and lichen-planus-like keratosis.",
      clinicalFeatures: "Waxy, verrucous, 'stuck-on' appearance, horn pseudocysts under magnification, ranging from light tan to dark brown or black.",
    },
    {
      code: "DF",
      name: "Dermatofibroma",
      risk: "Low Risk",
      riskBadgeClass: "badge-low",
      summary: "Benign fibrohistiocytic dermal nodule commonly developing following minor local trauma or insect bites on the lower extremities.",
      clinicalFeatures: "Firm, hyperpigmented, pink-to-tan dermal nodule exhibiting positive 'dimple sign' (central indentation upon lateral pinching).",
    },
    {
      code: "VASC",
      name: "Vascular Lesion",
      risk: "Low Risk",
      riskBadgeClass: "badge-low",
      summary: "Benign vascular proliferations including cherry angiomas, pyogenic granulomas, and angiokeratomas.",
      clinicalFeatures: "Well-demarcated bright red, violaceous, or black papules that typically blanch under diascopy pressure.",
    },
  ];

  const photographyRules = [
    {
      title: "Distance & Framing",
      icon: Camera,
      do: "Hold camera 10–15 cm from lesion. Ensure lesion occupies at least 30% of the frame without digital zoom distortion.",
      dont: "Do not take extreme wide-angle body shots or digital zooms that induce pixelation.",
    },
    {
      title: "Lighting & Diffuser",
      icon: Sun,
      do: "Use uniform, indirect daylight or diffuse LED illumination. Position lighting perpendicular to reduce skin glare.",
      dont: "Avoid direct camera flash directly over the lesion, which washes out pigmented borders and triggers IQA flags.",
    },
    {
      title: "Focus & Stability",
      icon: Sliders,
      do: "Tap the screen to lock focal point directly on the center of the lesion. Rest hand on an adjacent surface to prevent motion blur.",
      dont: "Never capture images while moving or with smudged camera optics (Laplacian variance < 100 will fail).",
    },
    {
      title: "Scale & Cleanliness",
      icon: Layers,
      do: "Clean the skin surface of makeup, oil, or topical ointments. If using a ruler, position it at the margin, not over the lesion.",
      dont: "Do not obscure lesion margins with surgical markers, bandages, or hair strands.",
    },
  ];

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>
      {/* Header */}
      <div
        className="medical-card"
        style={{
          padding: "24px 28px",
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          flexWrap: "wrap",
          gap: 16,
        }}
      >
        <div>
          <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 4 }}>
            <span className="badge badge-teal" style={{ textTransform: "uppercase" }}>
              Clinical Reference & Guidelines
            </span>
          </div>
          <h1 style={{ fontSize: "1.6rem", fontWeight: 800, color: "var(--text-main)" }}>
            Dermatological Screening Guidelines
          </h1>
          <p style={{ fontSize: "0.88rem", color: "var(--text-secondary)", marginTop: 2 }}>
            Diagnostic criteria, HAM10000 taxonomy summaries, and standardized photography protocols
          </p>
        </div>

        {/* Section Tabs */}
        <div
          style={{
            display: "flex",
            background: "var(--bg-surface-subtle)",
            padding: 3,
            borderRadius: "var(--radius-md)",
            border: "1px solid var(--border-color)",
          }}
        >
          <button
            onClick={() => setActiveSection("protocol")}
            className={`btn btn-sm ${activeSection === "protocol" ? "btn-secondary" : "btn-ghost"}`}
          >
            <Camera size={14} />
            <span>Photography SOP</span>
          </button>

          <button
            onClick={() => setActiveSection("abcde")}
            className={`btn btn-sm ${activeSection === "abcde" ? "btn-secondary" : "btn-ghost"}`}
          >
            <AlertTriangle size={14} />
            <span>ABCDE Criteria</span>
          </button>

          <button
            onClick={() => setActiveSection("conditions")}
            className={`btn btn-sm ${activeSection === "conditions" ? "btn-secondary" : "btn-ghost"}`}
          >
            <BookOpen size={14} />
            <span>7 Neoplasm Types</span>
          </button>
        </div>
      </div>

      {/* SECTION: PHOTOGRAPHY SOP */}
      {activeSection === "protocol" && (
        <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
          <div className="medical-card" style={{ padding: 20 }}>
            <h2 style={{ fontSize: "1.1rem", fontWeight: 700, color: "var(--text-main)", marginBottom: 4 }}>
              Clinical Photography Standard Operating Procedure (SOP)
            </h2>
            <p style={{ fontSize: "0.82rem", color: "var(--text-secondary)" }}>
              Strict adherence to these optical parameters guarantees that uploaded photographs pass server-side Image Quality Assessment (IQA) and preprocessing.
            </p>
          </div>

          <div
            style={{
              display: "grid",
              gridTemplateColumns: "repeat(auto-fit, minmax(240px, 1fr))",
              gap: 16,
            }}
          >
            {photographyRules.map((rule) => {
              const Icon = rule.icon;
              return (
                <div key={rule.title} className="medical-card" style={{ padding: 20 }}>
                  <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 12 }}>
                    <div
                      style={{
                        width: 32,
                        height: 32,
                        borderRadius: "var(--radius-sm)",
                        background: "var(--primary-light)",
                        color: "var(--text-main)",
                        display: "flex",
                        alignItems: "center",
                        justifyContent: "center",
                      }}
                    >
                      <Icon size={16} />
                    </div>
                    <strong style={{ fontSize: "0.95rem", color: "var(--text-main)" }}>
                      {rule.title}
                    </strong>
                  </div>

                  <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
                    <div
                      style={{
                        background: "var(--success-bg)",
                        border: "1px solid var(--success-border)",
                        borderRadius: "var(--radius-xs)",
                        padding: "8px 10px",
                        fontSize: "0.78rem",
                        color: "var(--success-text)",
                        lineHeight: 1.45,
                      }}
                    >
                      <strong>DO: </strong>{rule.do}
                    </div>

                    <div
                      style={{
                        background: "var(--error-bg)",
                        border: "1px solid var(--error-border)",
                        borderRadius: "var(--radius-xs)",
                        padding: "8px 10px",
                        fontSize: "0.78rem",
                        color: "var(--error-text)",
                        lineHeight: 1.45,
                      }}
                    >
                      <strong>AVOID: </strong>{rule.dont}
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* SECTION: ABCDE CRITERIA */}
      {activeSection === "abcde" && (
        <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
          <div className="medical-card" style={{ padding: 20 }}>
            <h2 style={{ fontSize: "1.1rem", fontWeight: 700, color: "var(--text-main)", marginBottom: 4 }}>
              The ABCDE Melanoma Screening Criteria
            </h2>
            <p style={{ fontSize: "0.82rem", color: "var(--text-secondary)" }}>
              The standardized clinical criteria used by dermatologists to distinguish early melanoma from benign melanocytic nevi
            </p>
          </div>

          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(200px, 1fr))", gap: 14 }}>
            {[
              { letter: "A", title: "Asymmetry", desc: "One half of the lesion does not match the other half in contour, shape, or thickness." },
              { letter: "B", title: "Border Irregularity", desc: "The edges are notched, scalloped, ragged, or poorly circumscribed." },
              { letter: "C", title: "Color Variegation", desc: "Multiple shades of tan, brown, black, red, white, or blue across the lesion." },
              { letter: "D", title: "Diameter > 6mm", desc: "Lesion width typically exceeds 6mm (approx size of pencil eraser), though early melanomas can be smaller." },
              { letter: "E", title: "Evolution / Change", desc: "Any change in size, shape, surface elevation, itching, bleeding, or color variegation over time." },
            ].map((item) => (
              <div key={item.letter} className="medical-card" style={{ padding: 18 }}>
                <div
                  style={{
                    width: 36,
                    height: 36,
                    borderRadius: "var(--radius-sm)",
                    background: "var(--primary-accent)",
                    color: "var(--text-main)",
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                    fontWeight: 800,
                    fontSize: "1.1rem",
                    marginBottom: 10,
                  }}
                >
                  {item.letter}
                </div>
                <h3 style={{ fontSize: "0.95rem", fontWeight: 700, color: "var(--text-main)", marginBottom: 4 }}>
                  {item.title}
                </h3>
                <p style={{ fontSize: "0.8rem", color: "var(--text-secondary)", lineHeight: 1.45 }}>
                  {item.desc}
                </p>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* SECTION: 7 NEOPLASM CONDITIONS */}
      {activeSection === "conditions" && (
        <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
          {conditions.map((c) => (
            <div key={c.code} className="medical-card" style={{ padding: 20 }}>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: 8, flexWrap: "wrap", gap: 8 }}>
                <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                  <span style={{ fontFamily: "var(--font-mono)", fontSize: "0.88rem", fontWeight: 700, color: "var(--text-main)" }}>
                    {c.code}
                  </span>
                  <h3 style={{ fontSize: "1.05rem", fontWeight: 700, color: "var(--text-main)" }}>
                    {c.name}
                  </h3>
                </div>
                <span className={`badge ${c.riskBadgeClass}`}>
                  {c.risk}
                </span>
              </div>

              <p style={{ fontSize: "0.84rem", color: "var(--text-secondary)", lineHeight: 1.5, marginBottom: 8 }}>
                {c.summary}
              </p>

              <div style={{ fontSize: "0.8rem", background: "var(--bg-surface-subtle)", padding: "8px 12px", borderRadius: "var(--radius-xs)", border: "1px solid var(--border-color)" }}>
                <strong style={{ color: "var(--text-main)" }}>Clinical Features: </strong>
                <span style={{ color: "var(--text-secondary)" }}>{c.clinicalFeatures}</span>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
