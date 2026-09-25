
import os
import base64
import tempfile
import math
import pickle
import warnings
import textwrap

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import geopandas as gpd
import torch
import torch.nn as nn

from ultralytics import YOLO

warnings.filterwarnings("ignore")


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Elephant Prediction AI",
    page_icon="🐘",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

YOLO_MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "best.pt"
)

TFT_MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "multi_video_tft_trajectory_model_best.pt"
)

FEATURE_SCALER_PATH = os.path.join(
    BASE_DIR,
    "scalers",
    "multi_video_tft_feature_scaler.pkl"
)

TARGET_SCALER_PATH = os.path.join(
    BASE_DIR,
    "scalers",
    "multi_video_tft_target_scaler.pkl"
)

RISK_GPKG = os.path.join(
    BASE_DIR,
    "GIS",
    "combined_risk_zones.gpkg"
)

STUDY_AREA_GPKG = os.path.join(
    BASE_DIR,
    "GIS",
    "koundinya_study_area_75km.gpkg"
)

FOREST_GPKG = os.path.join(
    BASE_DIR,
    "GIS",
    "forest_only.gpkg"
)


# ============================================================
# SESSION STATE
# ============================================================

if "analysis_done" not in st.session_state:
    st.session_state.analysis_done = False

if "analysis_data" not in st.session_state:
    st.session_state.analysis_data = None

if "prediction_data" not in st.session_state:
    st.session_state.prediction_data = None

if "risk_data" not in st.session_state:
    st.session_state.risk_data = None

if "video_path" not in st.session_state:
    st.session_state.video_path = None

if "annotated_video" not in st.session_state:
    st.session_state.annotated_video = None

if "uploader_version" not in st.session_state:
    st.session_state.uploader_version = 0


# ============================================================
# HELPER FUNCTIONS
# ============================================================

@st.cache_resource
def load_yolo_model():
    return YOLO(YOLO_MODEL_PATH)


@st.cache_resource
def load_tft_model():

    checkpoint = torch.load(
        TFT_MODEL_PATH,
        map_location="cpu",
        weights_only=False
    )

    state_dict = checkpoint["model_state_dict"]

    input_size = checkpoint["input_size"]
    hidden_size = checkpoint["hidden_size"]
    num_heads = checkpoint["num_heads"]
    dropout = checkpoint["dropout"]

    class TrajectoryTFT(nn.Module):

        def __init__(
            self,
            input_size,
            hidden_size,
            num_heads,
            dropout
        ):
            super().__init__()

            self.input_projection = nn.Linear(
                input_size,
                hidden_size
            )

            self.lstm = nn.LSTM(
                hidden_size,
                hidden_size,
                batch_first=True
            )

            self.attention = nn.MultiheadAttention(
                embed_dim=hidden_size,
                num_heads=num_heads,
                batch_first=True
            )

            self.norm = nn.LayerNorm(
                hidden_size
            )

            self.dropout = nn.Dropout(
                dropout
            )

            self.output_layer = nn.Linear(
                hidden_size,
                2
            )

        def forward(self, x):

            x = self.input_projection(x)

            lstm_output, _ = self.lstm(x)

            attention_output, _ = self.attention(
                lstm_output,
                lstm_output,
                lstm_output
            )

            x = self.norm(
                lstm_output + attention_output
            )

            x = x[:, -1, :]

            x = self.dropout(x)

            output = self.output_layer(x)

            return output

    model = TrajectoryTFT(
        input_size=input_size,
        hidden_size=hidden_size,
        num_heads=num_heads,
        dropout=dropout
    )

    model.load_state_dict(
        state_dict,
        strict=True
    )

    model.eval()

    return model


@st.cache_resource
def load_scalers():

    with open(FEATURE_SCALER_PATH, "rb") as f:
        feature_scaler = pickle.load(f)

    with open(TARGET_SCALER_PATH, "rb") as f:
        target_scaler = pickle.load(f)

    return feature_scaler, target_scaler


def calculate_movement(df):

    df = df.sort_values(
        ["track_id", "frame"]
    ).copy()

    df["dx"] = (
        df.groupby("track_id")["x_center"]
        .diff()
    )

    df["dy"] = (
        df.groupby("track_id")["y_center"]
        .diff()
    )

    df["dt"] = (
        df.groupby("track_id")["timestamp"]
        .diff()
    )

    df["distance"] = np.sqrt(
        df["dx"].fillna(0) ** 2 +
        df["dy"].fillna(0) ** 2
    )

    df["speed"] = (
        df["distance"] /
        df["dt"].replace(0, np.nan)
    )

    df["speed"] = (
        df["speed"]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
    )

    df["direction"] = np.degrees(
        np.arctan2(
            df["dy"].fillna(0),
            df["dx"].fillna(0)
        )
    )

    return df


def smooth_movement(df):

    df = df.sort_values(
        ["track_id", "frame"]
    ).copy()

    for column in ["x_center", "y_center"]:

        df[f"{column}_smooth"] = (
            df.groupby("track_id")[column]
            .transform(
                lambda s: s.rolling(
                    window=5,
                    center=True,
                    min_periods=1
                ).median()
            )
        )

    df["dx_smooth"] = (
        df.groupby("track_id")["x_center_smooth"]
        .diff()
        .fillna(0)
    )

    df["dy_smooth"] = (
        df.groupby("track_id")["y_center_smooth"]
        .diff()
        .fillna(0)
    )

    df["distance_smooth"] = np.sqrt(
        df["dx_smooth"] ** 2 +
        df["dy_smooth"] ** 2
    )

    df["speed_smooth"] = (
        df["distance_smooth"] /
        df["dt"].replace(0, np.nan)
    )

    df["speed_smooth"] = (
        df["speed_smooth"]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
    )

    df["direction_smooth"] = np.degrees(
        np.arctan2(
            df["dy_smooth"],
            df["dx_smooth"]
        )
    )

    df = df.replace(
        [np.inf, -np.inf],
        np.nan
    )

    df = df.fillna(0)

    return df


def generate_tft_predictions(
    movement_df,
    model,
    feature_scaler,
    target_scaler
):

    features = [
        "x_center_smooth",
        "y_center_smooth",
        "dx_smooth",
        "dy_smooth",
        "distance_smooth",
        "speed_smooth",
        "direction_smooth"
    ]

    sequence_length = 20

    predictions = []

    for track_id, group in movement_df.groupby(
        "track_id"
    ):

        group = group.sort_values(
            "frame"
        ).reset_index(drop=True)

        if len(group) <= sequence_length:
            continue

        feature_values = group[features].values

        scaled_features = feature_scaler.transform(
            feature_values
        )

        X = []
        metadata = []

        for i in range(
            sequence_length,
            len(group)
        ):

            X.append(
                scaled_features[
                    i - sequence_length:i
                ]
            )

            metadata.append(
                (
                    group.loc[i, "track_id"],
                    group.loc[i, "frame"],
                    group.loc[i, "x_center_smooth"],
                    group.loc[i, "y_center_smooth"]
                )
            )

        if not X:
            continue

        X = np.asarray(
            X,
            dtype=np.float32
        )

        tensor_x = torch.tensor(X)

        with torch.no_grad():
            pred_scaled = model(
                tensor_x
            ).cpu().numpy()

        pred_pixels = target_scaler.inverse_transform(
            pred_scaled
        )

        for j, values in enumerate(
            pred_pixels
        ):

            track_id_value = metadata[j][0]
            frame_value = metadata[j][1]
            actual_x = metadata[j][2]
            actual_y = metadata[j][3]

            predicted_x = float(values[0])
            predicted_y = float(values[1])

            error_x = (
                predicted_x -
                actual_x
            )

            error_y = (
                predicted_y -
                actual_y
            )

            position_error = math.sqrt(
                error_x ** 2 +
                error_y ** 2
            )

            predictions.append(
                {
                    "track_id": track_id_value,
                    "target_frame": frame_value,
                    "actual_x": actual_x,
                    "actual_y": actual_y,
                    "predicted_x": predicted_x,
                    "predicted_y": predicted_y,
                    "error_x": error_x,
                    "error_y": error_y,
                    "position_error": position_error
                }
            )

    return pd.DataFrame(
        predictions
    )


def calculate_prediction_metrics(predictions):

    if predictions.empty:
        return {}

    actual_x = predictions["actual_x"].values
    actual_y = predictions["actual_y"].values

    pred_x = predictions["predicted_x"].values
    pred_y = predictions["predicted_y"].values

    error_x = pred_x - actual_x
    error_y = pred_y - actual_y

    mae_x = np.mean(
        np.abs(error_x)
    )

    mae_y = np.mean(
        np.abs(error_y)
    )

    rmse_x = np.sqrt(
        np.mean(error_x ** 2)
    )

    rmse_y = np.sqrt(
        np.mean(error_y ** 2)
    )

    position_mae = np.mean(
        predictions["position_error"]
    )

    position_rmse = np.sqrt(
        np.mean(
            predictions["position_error"] ** 2
        )
    )

    def r2_score_manual(
        actual,
        predicted
    ):

        denominator = np.sum(
            (actual - np.mean(actual)) ** 2
        )

        if denominator == 0:
            return 0.0

        numerator = np.sum(
            (actual - predicted) ** 2
        )

        return 1 - (
            numerator / denominator
        )

    return {
        "X MAE": mae_x,
        "Y MAE": mae_y,
        "X RMSE": rmse_x,
        "Y RMSE": rmse_y,
        "Position MAE": position_mae,
        "Position RMSE": position_rmse,
        "X R²": r2_score_manual(
            actual_x,
            pred_x
        ),
        "Y R²": r2_score_manual(
            actual_y,
            pred_y
        )
    }


def pixel_to_utm(
    predictions,
    video_width,
    video_height
):

    predictions = predictions.copy()

    try:

        study_area = gpd.read_file(
            STUDY_AREA_GPKG
        )

        study_area = study_area.to_crs(
            "EPSG:32644"
        )

        minx, miny, maxx, maxy = (
            study_area.total_bounds
        )

        predictions["predicted_utm_x"] = (
            minx +
            (
                predictions["predicted_x"] /
                video_width
            ) *
            (maxx - minx)
        )

        predictions["predicted_utm_y"] = (
            maxy -
            (
                predictions["predicted_y"] /
                video_height
            ) *
            (maxy - miny)
        )

        predictions["mapping_type"] = (
            "Prototype pixel-to-UTM mapping"
        )

        return predictions

    except Exception:

        predictions["predicted_utm_x"] = np.nan
        predictions["predicted_utm_y"] = np.nan

        predictions["mapping_type"] = (
            "Unavailable"
        )

        return predictions


def assign_gis_risk(
    predictions
):

    if predictions.empty:
        return predictions

    if not os.path.exists(RISK_GPKG):
        predictions["risk_score"] = 0
        predictions["risk_level"] = "LOW"
        return predictions

    try:

        risk = gpd.read_file(
            RISK_GPKG
        )

        risk = risk.to_crs(
            "EPSG:32644"
        )

        points = gpd.GeoDataFrame(
            predictions.copy(),
            geometry=gpd.points_from_xy(
                predictions["predicted_utm_x"],
                predictions["predicted_utm_y"]
            ),
            crs="EPSG:32644"
        )

        joined = gpd.sjoin(
            points,
            risk,
            how="left",
            predicate="within"
        )

        score_column = None

        for column in [
            "risk_score",
            "max_risk_score",
            "score"
        ]:

            if column in joined.columns:
                score_column = column
                break

        if score_column:

            joined["risk_score_final"] = (
                joined[score_column]
                .fillna(0)
            )

        else:

            joined["risk_score_final"] = 0

        def risk_level(score):

            score = float(score)

            if score >= 7:
                return "CRITICAL"

            if score >= 5:
                return "HIGH"

            if score >= 3:
                return "MEDIUM"

            return "LOW"

        joined["risk_level"] = (
            joined["risk_score_final"]
            .apply(risk_level)
        )

        result = pd.DataFrame(
            joined.drop(
                columns=["geometry"],
                errors="ignore"
            )
        )

        return result

    except Exception:

        predictions = predictions.copy()

        predictions["risk_score_final"] = 0
        predictions["risk_level"] = "LOW"

        return predictions


def create_trajectory_plot(
    movement_df,
    prediction_df
):

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    tracks = movement_df[
        "track_id"
    ].unique()

    for track_id in tracks:

        track = movement_df[
            movement_df["track_id"] == track_id
        ]

        ax.plot(
            track["x_center_smooth"],
            track["y_center_smooth"],
            linewidth=1.5,
            label=f"Track {track_id}"
        )

    if not prediction_df.empty:

        ax.scatter(
            prediction_df["predicted_x"],
            prediction_df["predicted_y"],
            marker="x",
            s=25,
            label="TFT Prediction"
        )

    ax.set_title(
        "Elephant Movement and Predicted Trajectory"
    )

    ax.set_xlabel(
        "Image X Coordinate"
    )

    ax.set_ylabel(
        "Image Y Coordinate"
    )

    ax.invert_yaxis()

    if len(tracks) <= 10:
        ax.legend()

    fig.tight_layout()

    return fig


def create_speed_plot(
    movement_df
):

    fig, ax = plt.subplots(
        figsize=(10, 5)
    )

    for track_id, group in movement_df.groupby(
        "track_id"
    ):

        ax.plot(
            group["frame"],
            group["speed_smooth"],
            linewidth=1,
            label=f"Track {track_id}"
        )

    ax.set_title(
        "Elephant Movement Speed"
    )

    ax.set_xlabel(
        "Frame"
    )

    ax.set_ylabel(
        "Pixel Speed"
    )

    if movement_df["track_id"].nunique() <= 10:
        ax.legend()

    fig.tight_layout()

    return fig


def create_risk_plot(
    risk_df
):

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    if risk_df.empty:

        ax.text(
            0.5,
            0.5,
            "No GIS risk predictions available",
            ha="center",
            va="center"
        )

        ax.axis("off")

        return fig

    counts = (
        risk_df["risk_level"]
        .value_counts()
        .reindex(
            [
                "LOW",
                "MEDIUM",
                "HIGH",
                "CRITICAL"
            ],
            fill_value=0
        )
    )

    ax.bar(
        counts.index,
        counts.values
    )

    ax.set_title(
        "GIS Risk Distribution"
    )

    ax.set_xlabel(
        "Risk Level"
    )

    ax.set_ylabel(
        "Predictions"
    )

    fig.tight_layout()

    return fig


def create_gis_visualization_plot(risk_df):

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    try:

        study_area = gpd.read_file(
            STUDY_AREA_GPKG
        )

        study_area = study_area.to_crs(
            "EPSG:32644"
        )

        study_area.boundary.plot(
            ax=ax,
            linewidth=1
        )

    except Exception:

        pass

    if not risk_df.empty and {
        "predicted_utm_x",
        "predicted_utm_y",
        "risk_level"
    }.issubset(risk_df.columns):

        valid_points = risk_df.dropna(
            subset=[
                "predicted_utm_x",
                "predicted_utm_y"
            ]
        )

        for level in [
            "LOW",
            "MEDIUM",
            "HIGH",
            "CRITICAL"
        ]:

            subset = valid_points[
                valid_points["risk_level"] == level
            ]

            if not subset.empty:

                ax.scatter(
                    subset["predicted_utm_x"],
                    subset["predicted_utm_y"],
                    s=20,
                    label=level
                )

    ax.set_title(
        "GIS Visualization of Predicted Positions"
    )

    ax.set_xlabel(
        "UTM Easting"
    )

    ax.set_ylabel(
        "UTM Northing"
    )

    if not risk_df.empty:
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend()

    fig.tight_layout()

    return fig


def create_pdf_report(
    movement_df,
    prediction_df,
    risk_df,
    metrics,
    video_name
):

    from io import BytesIO

    from reportlab.lib.pagesizes import A4
    from reportlab.platypus import (
        SimpleDocTemplate,
        Paragraph,
        Spacer,
        Table,
        TableStyle,
        Image
    )
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet

    buffer = BytesIO()

    document = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40
    )

    styles = getSampleStyleSheet()
    story = []

    story.append(
        Paragraph(
            "Elephant Prediction AI",
            styles["Title"]
        )
    )

    story.append(
        Spacer(1, 8)
    )

    story.append(
        Paragraph(
            "AI-Based Elephant Trajectory Prediction "
            "and GIS-Enabled Risk Assessment System",
            styles["Heading2"]
        )
    )

    story.append(
        Spacer(1, 15)
    )

    story.append(
        Paragraph(
            f"<b>Analyzed Video:</b> {video_name}",
            styles["BodyText"]
        )
    )

    story.append(
        Spacer(1, 15)
    )

    story.append(
        Paragraph(
            "1. Overall Summary",
            styles["Heading2"]
        )
    )

    risk_counts = (
        risk_df["risk_level"].value_counts()
        if not risk_df.empty and "risk_level" in risk_df.columns
        else pd.Series(dtype=int)
    )

    most_frequent = (
        risk_counts.idxmax()
        if not risk_counts.empty
        else "LOW"
    )

    most_frequent_count = (
        int(risk_counts.max())
        if not risk_counts.empty
        else 0
    )

    severity_order = [
        "LOW",
        "MEDIUM",
        "HIGH",
        "CRITICAL"
    ]

    severity_present = [
        level
        for level in severity_order
        if risk_counts.get(level, 0) > 0
    ]

    highest_severity = (
        severity_present[-1]
        if severity_present
        else "LOW"
    )

    highest_severity_count = int(
        risk_counts.get(
            highest_severity,
            0
        )
    )

    overall_rows = [
        ["Metric", "Value"],
        [
            "Elephant Tracks",
            str(movement_df["track_id"].nunique())
        ],
        [
            "Frames Analyzed",
            str(movement_df["frame"].nunique())
        ],
        [
            "Trajectory Predictions",
            str(len(prediction_df))
        ],
        [
            "Most Frequent Risk",
            f"{most_frequent} ({most_frequent_count})"
        ],
        [
            "Highest Severity Reached",
            f"{highest_severity} ({highest_severity_count})"
        ]
    ]

    overall_table = Table(
        overall_rows,
        colWidths=[260, 180]
    )

    overall_table.setStyle(
        TableStyle(
            [
                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, 0),
                    colors.lightgrey
                ),
                (
                    "GRID",
                    (0, 0),
                    (-1, -1),
                    0.5,
                    colors.grey
                ),
                (
                    "PADDING",
                    (0, 0),
                    (-1, -1),
                    6
                )
            ]
        )
    )

    story.append(
        overall_table
    )

    story.append(
        Spacer(1, 15)
    )

    story.append(
        Paragraph(
            "2. System Pipeline",
            styles["Heading2"]
        )
    )

    story.append(
        Paragraph(
            "Video Input → YOLOv10 Detection → ByteTrack "
            "Tracking → Movement Feature Extraction → "
            "TFT-style Trajectory Prediction → "
            "GIS Risk Assessment",
            styles["BodyText"]
        )
    )

    story.append(
        Spacer(1, 15)
    )

    story.append(
        Paragraph(
            "3. Detection and Tracking",
            styles["Heading2"]
        )
    )

    summary_rows = [
        ["Metric", "Value"],
        [
            "Frames analyzed",
            str(movement_df["frame"].nunique())
        ],
        [
            "Elephant tracks",
            str(movement_df["track_id"].nunique())
        ],
        [
            "Tracking records",
            str(len(movement_df))
        ],
        [
            "Trajectory predictions",
            str(len(prediction_df))
        ]
    ]

    summary_table = Table(
        summary_rows,
        colWidths=[260, 180]
    )

    summary_table.setStyle(
        TableStyle(
            [
                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, 0),
                    colors.lightgrey
                ),
                (
                    "GRID",
                    (0, 0),
                    (-1, -1),
                    0.5,
                    colors.grey
                ),
                (
                    "PADDING",
                    (0, 0),
                    (-1, -1),
                    6
                )
            ]
        )
    )

    story.append(
        summary_table
    )

    story.append(
        Spacer(1, 15)
    )

    story.append(
        Paragraph(
            "4. TFT-style Trajectory Prediction",
            styles["Heading2"]
        )
    )

    if metrics:

        metric_rows = [
            ["Metric", "Value"]
        ]

        for key, value in metrics.items():

            metric_rows.append(
                [
                    key,
                    f"{value:.4f}"
                ]
            )

        metric_table = Table(
            metric_rows,
            colWidths=[260, 180]
        )

        metric_table.setStyle(
            TableStyle(
                [
                    (
                        "BACKGROUND",
                        (0, 0),
                        (-1, 0),
                        colors.lightgrey
                    ),
                    (
                        "GRID",
                        (0, 0),
                        (-1, -1),
                        0.5,
                        colors.grey
                    ),
                    (
                        "PADDING",
                        (0, 0),
                        (-1, -1),
                        6
                    )
                ]
            )
        )

        story.append(
            metric_table
        )

    story.append(
        Spacer(1, 15)
    )

    story.append(
        Paragraph(
            "5. GIS Risk Assessment",
            styles["Heading2"]
        )
    )

    if not risk_df.empty:

        risk_rows = [
            ["Risk Level", "Predictions"]
        ]

        for level in severity_order:

            risk_rows.append(
                [
                    level,
                    str(
                        int(
                            risk_counts.get(
                                level,
                                0
                            )
                        )
                    )
                ]
            )

        risk_table = Table(
            risk_rows,
            colWidths=[260, 180]
        )

        risk_table.setStyle(
            TableStyle(
                [
                    (
                        "BACKGROUND",
                        (0, 0),
                        (-1, 0),
                        colors.lightgrey
                    ),
                    (
                        "GRID",
                        (0, 0),
                        (-1, -1),
                        0.5,
                        colors.grey
                    ),
                    (
                        "PADDING",
                        (0, 0),
                        (-1, -1),
                        6
                    )
                ]
            )
        )

        story.append(
            risk_table
        )

        story.append(
            Spacer(1, 10)
        )

        story.append(
            Paragraph(
                f"<b>Most Frequent Risk:</b> "
                f"{most_frequent} "
                f"({most_frequent_count} predictions)",
                styles["BodyText"]
            )
        )

        story.append(
            Paragraph(
                f"<b>Highest Severity Reached:</b> "
                f"{highest_severity} "
                f"({highest_severity_count} predictions)",
                styles["BodyText"]
            )
        )

    story.append(
        Spacer(1, 15)
    )

    story.append(
        Paragraph(
            "6. GIS Visualization of Predicted Positions",
            styles["Heading2"]
        )
    )

    gis_fig = create_gis_visualization_plot(
        risk_df
    )

    gis_buffer = BytesIO()

    gis_fig.savefig(
        gis_buffer,
        format="png",
        dpi=160,
        bbox_inches="tight"
    )

    plt.close(
        gis_fig
    )

    gis_buffer.seek(0)

    story.append(
        Image(
            gis_buffer,
            width=500,
            height=300
        )
    )

    document.build(
        story
    )

    buffer.seek(0)

    return buffer.getvalue()


# ============================================================
# ============================================================
# HEADER AND NAVIGATION
# ============================================================

hero_image_path = os.path.join(
    BASE_DIR,
    "assets",
    "elephant_hero.jpg"
)

if "active_page" not in st.session_state:
    st.session_state.active_page = "HOME"


# ============================================================
# TOP NAVIGATION
# ============================================================

st.markdown(
    """
    <style>

    .top-title {
        color: #064e2b;
        font-size: 23px;
        font-weight: 800;
        white-space: nowrap;
        text-align: left;
        padding-top: 9px;
        letter-spacing: 0.2px;
    }

    div.stButton > button {
        min-height: 42px;
        border-radius: 8px;
        font-weight: 700;
    }

    div.stButton > button[kind="primary"] {
        background-color: #064e2b !important;
        color: #ffffff !important;
        border: 1px solid #064e2b !important;
    }

    div.stButton > button[kind="primary"]:hover {
        background-color: #053d22 !important;
        color: #ffffff !important;
        border-color: #053d22 !important;
    }

    div.stButton > button[kind="secondary"] {
        background-color: #ffffff !important;
        color: #064e2b !important;
        border: 1px solid #064e2b !important;
    }

    div.stButton > button[kind="secondary"]:hover {
        background-color: #eaf5ef !important;
        color: #064e2b !important;
        border-color: #064e2b !important;
    }

    </style>
    """,
    unsafe_allow_html=True
)

nav_items = [
    "HOME",
    "DETECTION",
    "GIS RISK",
    "TRAJECTORY",
    "ANALYTICS",
    "REPORTS"
]

nav_columns = st.columns(
    [2.35, 1, 1, 1, 1, 1, 1]
)

with nav_columns[0]:

    st.markdown(
        """
        <div class="top-title">
            Elephant Prediction AI
        </div>
        """,
        unsafe_allow_html=True
    )

for column, page in zip(
    nav_columns[1:],
    nav_items
):

    with column:

        button_type = (
            "primary"
            if page == st.session_state.active_page
            else "secondary"
        )

        if st.button(
            page,
            key=f"nav_{page.replace(' ', '_')}",
            use_container_width=True,
            type=button_type
        ):
            st.session_state.active_page = page
            st.rerun()


# ============================================================
# HOME HERO SECTION
# ============================================================

if st.session_state.active_page == "HOME":

    if os.path.exists(hero_image_path):

        with open(hero_image_path, "rb") as image_file:
            hero_b64 = base64.b64encode(
                image_file.read()
            ).decode("utf-8")

        hero_html = f"""
<style>
.full-hero {{
    width: 100vw;
    min-height: 560px;
    margin-left: calc(50% - 50vw);
    background-image:
        linear-gradient(
            rgba(0, 25, 12, 0.38),
            rgba(0, 15, 8, 0.52)
        ),
        url("data:image/jpeg;base64,{hero_b64}");
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
    display: flex;
    align-items: center;
    justify-content: center;
    text-align: center;
}}

.hero-content {{
    max-width: 1100px;
    padding: 40px;
}}

.hero-title {{
    color: #ffffff;
    font-size: 58px;
    font-weight: 900;
    margin: 0;
    text-shadow: 0 4px 18px rgba(0,0,0,0.82);
}}

.hero-subtitle {{
    color: #ffffff;
    font-size: 24px;
    font-weight: 600;
    margin-top: 18px;
    line-height: 1.45;
    text-shadow: 0 3px 12px rgba(0,0,0,0.82);
}}
</style>

<div class="full-hero">
    <div class="hero-content">
        <div class="hero-title">
            Elephant Prediction AI
        </div>
        <div class="hero-subtitle">
            AI-Based Elephant Trajectory Prediction and
            GIS-Enabled Risk Assessment System
        </div>
    </div>
</div>
"""

        st.markdown(
            hero_html,
            unsafe_allow_html=True
        )

    else:

        st.warning(
            "Hero image not found."
        )

st.divider()


# UPLOAD FIRST
# ============================================================

if (
    not st.session_state.analysis_done
    or st.session_state.active_page == "HOME"
):

    st.header("🎥 Upload Elephant Video")

    uploaded_file = st.file_uploader(
        "Choose a video",
        type=[
            "mp4",
            "avi",
            "mov",
            "mkv"
        ],
        key=f"video_uploader_{st.session_state.uploader_version}"
    )

    if uploaded_file is not None:

        st.video(
            uploaded_file
        )

        st.write(
            f"Selected video: {uploaded_file.name}"
        )

        analyze_button = st.button(
            "🚀 Analyze Video",
            type="primary",
            use_container_width=True
        )

        if analyze_button:

            st.session_state.uploaded_video_name = (
                uploaded_file.name
            )

            with st.spinner(
                "Running YOLOv10 detection and ByteTrack tracking..."
            ):

                suffix = os.path.splitext(
                    uploaded_file.name
                )[1]

                temp_input = tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=suffix
                )

                temp_input.write(
                    uploaded_file.getbuffer()
                )

                temp_input.close()

                st.session_state.video_path = (
                    temp_input.name
                )

                cap = cv2.VideoCapture(
                    temp_input.name
                )

                fps = cap.get(
                    cv2.CAP_PROP_FPS
                )

                if fps <= 0:
                    fps = 30.0

                frame_width = int(
                    cap.get(
                        cv2.CAP_PROP_FRAME_WIDTH
                    )
                )

                frame_height = int(
                    cap.get(
                        cv2.CAP_PROP_FRAME_HEIGHT
                    )
                )

                fourcc = cv2.VideoWriter_fourcc(
                    *"mp4v"
                )

                output_path = tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=".mp4"
                ).name

                writer = cv2.VideoWriter(
                    output_path,
                    fourcc,
                    fps,
                    (
                        frame_width,
                        frame_height
                    )
                )

                model = load_yolo_model()

                yolo_device = (
                    0 if torch.cuda.is_available()
                    else "cpu"
                )

                records = []
                frame_number = 0

                total_frames = int(
                    cap.get(cv2.CAP_PROP_FRAME_COUNT)
                )

                progress_bar = st.progress(
                    0,
                    text="Analyzing video..."
                )

                progress_text = st.empty()


                while True:

                    ret, frame = cap.read()

                    if not ret:
                        break

                    results = model.track(
                        frame,
                        tracker="bytetrack.yaml",
                        conf=0.25,
                        iou=0.5,
                        persist=True,
                        verbose=False,
                        device=yolo_device
                    )

                    annotated = frame.copy()

                    if results:

                        result = results[0]

                        if result.boxes is not None:

                            boxes = result.boxes

                            if boxes.xyxy is not None:

                                xyxy = boxes.xyxy.cpu().numpy()

                                confs = (
                                    boxes.conf.cpu().numpy()
                                    if boxes.conf is not None
                                    else np.ones(
                                        len(xyxy)
                                    )
                                )

                                if boxes.id is not None:

                                    track_ids = (
                                        boxes.id
                                        .cpu()
                                        .numpy()
                                        .astype(int)
                                    )

                                else:

                                    track_ids = [
                                        -1
                                    ] * len(xyxy)

                                for box, conf, track_id in zip(
                                    xyxy,
                                    confs,
                                    track_ids
                                ):

                                    x1, y1, x2, y2 = box

                                    x_center = (
                                        x1 + x2
                                    ) / 2

                                    y_center = (
                                        y1 + y2
                                    ) / 2

                                    width = (
                                        x2 - x1
                                    )

                                    height = (
                                        y2 - y1
                                    )

                                    records.append(
                                        {
                                            "frame": frame_number,
                                            "track_id": int(track_id),
                                            "x_center": float(x_center),
                                            "y_center": float(y_center),
                                            "width": float(width),
                                            "height": float(height),
                                            "confidence": float(conf),
                                            "timestamp": (
                                                frame_number /
                                                fps
                                            )
                                        }
                                    )

                                    cv2.rectangle(
                                        annotated,
                                        (
                                            int(x1),
                                            int(y1)
                                        ),
                                        (
                                            int(x2),
                                            int(y2)
                                        ),
                                        (0, 255, 0),
                                        2
                                    )

                                    cv2.putText(
                                        annotated,
                                        f"Elephant ID {track_id}",
                                        (
                                            int(x1),
                                            max(
                                                20,
                                                int(y1) - 10
                                            )
                                        ),
                                        cv2.FONT_HERSHEY_SIMPLEX,
                                        0.6,
                                        (0, 255, 0),
                                        2
                                    )

                    writer.write(
                        annotated
                    )

                    frame_number += 1

                    if total_frames > 0:
                        progress = min(
                            frame_number / total_frames,
                            1.0
                        )

                        progress_bar.progress(
                            progress,
                            text=f"Analyzing video... {progress * 100:.1f}%"
                        )

                        progress_text.write(
                            f"Frames processed: {frame_number:,} / {total_frames:,}"
                        )

                cap.release()
                writer.release()

                progress_bar.progress(
                    1.0,
                    text="✅ Video analysis completed"
                )

                progress_text.write(
                    f"Frames processed: {total_frames:,} / {total_frames:,}"
                )

                browser_video_path = tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix="_annotated_h264.mp4"
                ).name

                import subprocess

                ffmpeg_command = [
                    "ffmpeg",
                    "-y",
                    "-i",
                    output_path,
                    "-c:v",
                    "libx264",
                    "-preset",
                    "veryfast",
                    "-crf",
                    "23",
                    "-pix_fmt",
                    "yuv420p",
                    "-movflags",
                    "+faststart",
                    "-an",
                    browser_video_path
                ]

                conversion = subprocess.run(
                    ffmpeg_command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )

                if conversion.returncode == 0:
                    st.session_state.annotated_video = (
                        browser_video_path
                    )
                else:
                    st.session_state.annotated_video = (
                        output_path
                    )

                if records:

                    tracking_df = pd.DataFrame(
                        records
                    )

                    tracking_df = tracking_df[
                        tracking_df["track_id"] >= 0
                    ].copy()

                else:

                    tracking_df = pd.DataFrame()

                if tracking_df.empty:

                    st.error(
                        "No elephants were detected in the uploaded video."
                    )

                    st.stop()

                movement_df = calculate_movement(
                    tracking_df
                )

                movement_df = smooth_movement(
                    movement_df
                )

                st.session_state.analysis_data = (
                    movement_df
                )

                with st.spinner(
                    "Running TFT-style trajectory prediction..."
                ):

                    tft_model = load_tft_model()

                    feature_scaler, target_scaler = (
                        load_scalers()
                    )

                    prediction_df = generate_tft_predictions(
                        movement_df,
                        tft_model,
                        feature_scaler,
                        target_scaler
                    )

                st.session_state.prediction_data = (
                    prediction_df
                )

                if not prediction_df.empty:

                    prediction_df = pixel_to_utm(
                        prediction_df,
                        frame_width,
                        frame_height
                    )

                    with st.spinner(
                        "Running GIS risk assessment..."
                    ):

                        prediction_df = assign_gis_risk(
                            prediction_df
                        )

                    st.session_state.risk_data = (
                        prediction_df
                    )

                else:

                    st.session_state.risk_data = (
                        pd.DataFrame()
                    )

                st.session_state.analysis_done = True
                st.session_state.active_page = "HOME"
                st.rerun()


# ============================================================
# RESULTS
# ============================================================

if st.session_state.analysis_done:

    movement_df = st.session_state.analysis_data
    prediction_df = st.session_state.prediction_data
    risk_df = st.session_state.risk_data

    metrics = calculate_prediction_metrics(
        prediction_df
    )

    active_page = st.session_state.active_page

    # ========================================================
    # HOME
    # ========================================================

    if active_page == "HOME":

        st.header("📊 Analysis Overview")

        overview_row1 = st.columns(3)

        with overview_row1[0]:
            st.write("🐘 **Elephant Tracks**")
            st.subheader(
                str(
                    movement_df["track_id"].nunique()
                )
            )

        with overview_row1[1]:
            st.write("🎞️ **Frames Analyzed**")
            st.subheader(
                str(
                    movement_df["frame"].nunique()
                )
            )

        with overview_row1[2]:
            st.write("🎯 **Trajectory Predictions**")
            st.subheader(
                str(
                    len(prediction_df)
                )
            )

        st.write("")

        risk_counts = (
            risk_df["risk_level"].value_counts()
            if not risk_df.empty and "risk_level" in risk_df.columns
            else pd.Series(dtype=int)
        )

        most_frequent_risk = (
            risk_counts.idxmax()
            if not risk_counts.empty
            else "LOW"
        )

        most_frequent_count = (
            int(risk_counts.max())
            if not risk_counts.empty
            else 0
        )

        severity_order = [
            "LOW",
            "MEDIUM",
            "HIGH",
            "CRITICAL"
        ]

        severity_present = [
            level
            for level in severity_order
            if risk_counts.get(level, 0) > 0
        ]

        highest_severity = (
            severity_present[-1]
            if severity_present
            else "LOW"
        )

        highest_severity_count = int(
            risk_counts.get(
                highest_severity,
                0
            )
        )

        overview_row2 = st.columns(2)

        with overview_row2[0]:
            st.write("🟢 **Most Frequent Risk**")
            st.subheader(
                most_frequent_risk
            )
            st.caption(
                f"{most_frequent_count} predictions"
            )

        with overview_row2[1]:
            st.write("⚠️ **Highest Severity Reached**")
            st.subheader(
                highest_severity
            )
            st.caption(
                f"{highest_severity_count} predictions"
            )

    # ========================================================
    # DETECTION
    # ========================================================

    elif active_page == "DETECTION":

        st.header("🎥 Detection & Tracking")

        video_col, details_col = st.columns(
            [2.2, 1]
        )

        with video_col:

            st.subheader(
                "🎬 Annotated Detection Video"
            )

            if st.session_state.annotated_video:

                video_path = st.session_state.annotated_video

                if os.path.exists(video_path):

                    with open(
                        video_path,
                        "rb"
                    ) as video_file:

                        video_bytes = video_file.read()

                    st.video(
                        video_bytes,
                        format="video/mp4"
                    )

                else:

                    st.warning(
                        "Annotated video file was not found."
                    )

            else:

                st.warning(
                    "Annotated video is not available."
                )

        with details_col:

            st.subheader(
                "Detection Summary"
            )

            st.write(
                f"**Elephant tracks:** "
                f"{movement_df['track_id'].nunique()}"
            )

            st.write(
                f"**Frames:** "
                f"{movement_df['frame'].nunique()}"
            )

            st.write(
                f"**Tracking records:** "
                f"{len(movement_df)}"
            )

            st.success(
                "✓ YOLOv10 Detection"
            )

            st.success(
                "✓ ByteTrack Tracking"
            )

    # ========================================================
    # GIS RISK
    # ========================================================

    elif active_page == "GIS RISK":

        st.header("🗺️ GIS Risk Assessment")

        if not risk_df.empty:

            risk_counts = (
                risk_df["risk_level"]
                .value_counts()
                .reindex(
                    [
                        "LOW",
                        "MEDIUM",
                        "HIGH",
                        "CRITICAL"
                    ],
                    fill_value=0
                )
            )

            risk_cols = st.columns(4)

            for column, level in zip(
                risk_cols,
                [
                    "LOW",
                    "MEDIUM",
                    "HIGH",
                    "CRITICAL"
                ]
            ):

                with column:

                    st.write(
                        f"**{level}**"
                    )

                    st.subheader(
                        str(
                            int(
                                risk_counts[level]
                            )
                        )
                    )

                    st.caption(
                        "Predictions"
                    )

            most_frequent_risk = risk_counts.idxmax()
            most_frequent_count = int(
                risk_counts.max()
            )

            severity_present = [
                level
                for level in [
                    "LOW",
                    "MEDIUM",
                    "HIGH",
                    "CRITICAL"
                ]
                if risk_counts[level] > 0
            ]

            highest_risk = (
                severity_present[-1]
                if severity_present
                else "LOW"
            )

            highest_risk_count = int(
                risk_counts[highest_risk]
            )

            summary_cols = st.columns(2)

            with summary_cols[0]:
                st.write("🟢 **Most Frequent Risk**")
                st.subheader(
                    most_frequent_risk
                )
                st.caption(
                    f"{most_frequent_count} predictions"
                )

            with summary_cols[1]:
                st.write("⚠️ **Highest Severity Reached**")
                st.subheader(
                    highest_risk
                )
                st.caption(
                    f"{highest_risk_count} predictions"
                )

            if highest_risk in ["HIGH", "CRITICAL"]:
                st.warning(
                    f"{highest_risk_count} trajectory predictions fall in "
                    f"{highest_risk}-risk zones."
                )

            risk_fig = create_risk_plot(
                risk_df
            )

            st.pyplot(
                risk_fig,
                clear_figure=True
            )

            plt.close(
                risk_fig
            )

            st.subheader(
                "GIS Visualization of Predicted Positions"
            )

            gis_fig = create_gis_visualization_plot(
                risk_df
            )

            st.pyplot(
                gis_fig,
                clear_figure=True
            )

            plt.close(
                gis_fig
            )

        else:

            st.info(
                "No GIS risk predictions are available."
            )

    # ========================================================
    # TRAJECTORY
    # ========================================================

    elif active_page == "TRAJECTORY":

        st.header("📈 Trajectory Prediction")

        trajectory_fig = create_trajectory_plot(
            movement_df,
            prediction_df
        )

        st.pyplot(
            trajectory_fig,
            clear_figure=True
        )

        plt.close(
            trajectory_fig
        )

        if metrics:

            st.subheader(
                "🧠 TFT-style Prediction Performance"
            )

            metric_cols = st.columns(4)

            metric_values = [
                (
                    "Position MAE",
                    metrics["Position MAE"]
                ),
                (
                    "Position RMSE",
                    metrics["Position RMSE"]
                ),
                (
                    "X R²",
                    metrics["X R²"]
                ),
                (
                    "Y R²",
                    metrics["Y R²"]
                )
            ]

            for column, item in zip(
                metric_cols,
                metric_values
            ):

                with column:

                    st.write(
                        f"**{item[0]}**"
                    )

                    st.subheader(
                        f"{item[1]:.4f}"
                    )

    # ========================================================
    # ANALYTICS
    # ========================================================

    elif active_page == "ANALYTICS":

        st.header("📊 Movement Analytics")

        speed_fig = create_speed_plot(
            movement_df
        )

        st.pyplot(
            speed_fig,
            clear_figure=True
        )

        plt.close(
            speed_fig
        )

    # ========================================================
    # REPORTS
    # ========================================================

    elif active_page == "REPORTS":

        st.header("📄 Reports")

        try:

            pdf_bytes = create_pdf_report(
                movement_df,
                prediction_df,
                risk_df,
                metrics,
                st.session_state.uploaded_video_name
            )

            st.success(
                "✅ Full project report generated successfully."
            )

            st.download_button(
                label="📄 Download Full Project Report",
                data=pdf_bytes,
                file_name=(
                    "Elephant_Prediction_AI_Analysis_Report.pdf"
                ),
                mime="application/pdf",
                type="primary",
                use_container_width=True
            )

        except Exception as e:

            st.error(
                "PDF report generation failed."
            )

            st.exception(e)


# ============================================================
# INITIAL SCREEN
# ============================================================

if not st.session_state.analysis_done:

    if st.session_state.active_page != "HOME":

        st.info(
            "Upload and analyze a video from the HOME section first."
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.markdown(
    "<div style='text-align:center; padding:12px 0; color:#777;'>"
    "© 2026 Elephant Prediction AI · All Rights Reserved · Developed by Our Team"
    "</div>",
    unsafe_allow_html=True
)
