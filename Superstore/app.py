import streamlit as st
import pandas as pd
import torch
import torch.nn as nn
import json
from pathlib import Path


# --------------------------------------------------
# Discount range finder
# --------------------------------------------------

def get_discount_ranges(
    model,
    sample,
    discount_index,
    min_discount=0.0,
    max_discount=0.8,
    step=0.001
):
    model.eval()

    class_names = {
        0: "Loss",
        1: "Medium Profit",
        2: "High Profit"
    }

    predictions = []

    with torch.no_grad():

        discount = min_discount

        while discount <= max_discount + 1e-9:

            x = sample.clone()

            # Replace discount value
            x[discount_index] = discount

            output = model(x.unsqueeze(0))

            predicted_class = output.argmax(dim=1).item()

            predictions.append(
                (round(discount, 3), predicted_class)
            )

            discount += step


    # --------------------------------------------------
    # Convert predictions into ranges
    # --------------------------------------------------

    ranges = []

    start_discount = predictions[0][0]
    current_class = predictions[0][1]

    for i in range(1, len(predictions)):

        discount, predicted_class = predictions[i]

        if predicted_class != current_class:

            end_discount = predictions[i - 1][0]

            ranges.append({
                "start": start_discount,
                "end": end_discount,
                "class": current_class
            })

            start_discount = discount
            current_class = predicted_class


    # Final range
    ranges.append({
        "start": start_discount,
        "end": predictions[-1][0],
        "class": current_class
    })

    return ranges, class_names


# --------------------------------------------------
# Paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent


# --------------------------------------------------
# Load category values
# --------------------------------------------------

with open(
    BASE_DIR / "notebooks" / "categories.json",
    "r"
) as f:

    categories = json.load(f)



# --------------------------------------------------

with open(
    BASE_DIR / "notebooks" / "feature_columns.json",
    "r"
) as f:

    feature_columns = json.load(f)


# --------------------------------------------------
# Model
# --------------------------------------------------

model = nn.Sequential(

    nn.Linear(95, 128),
    nn.BatchNorm1d(128),
    nn.ReLU(),
    nn.Dropout(0.25),

    nn.Linear(128, 64),
    nn.BatchNorm1d(64),
    nn.ReLU(),
    nn.Dropout(0.20),

    nn.Linear(64, 32),
    nn.ReLU(),

    nn.Linear(32, 3)
)


# --------------------------------------------------
# Load trained weights
# --------------------------------------------------

state_dict = torch.load(
    BASE_DIR / "notebooks" / "best_model.pth",
    map_location="cpu"
)

model.load_state_dict(state_dict)

model.eval()


# --------------------------------------------------
# Streamlit page
# --------------------------------------------------

st.title("Superstore Discount Optimizer")

st.write(
    "Enter the order information and choose the desired "
    "profit level."
)


# --------------------------------------------------
# Categorical inputs
# --------------------------------------------------

ship_mode = st.selectbox(
    "Ship Mode",
    categories["Ship Mode"]
)

segment = st.selectbox(
    "Segment",
    categories["Segment"]
)

state = st.selectbox(
    "State",
    categories["State"]
)

region = st.selectbox(
    "Region",
    categories["Region"]
)

category = st.selectbox(
    "Category",
    categories["Category"]
)

sub_category = st.selectbox(
    "Sub-Category",
    categories["Sub-Category"]
)

order_month = st.selectbox(
    "Order Month",
    list(range(1, 13))
)


# --------------------------------------------------
# Numerical inputs
# --------------------------------------------------

quantity = st.number_input(
    "Quantity",
    min_value=1,
    step=1,
    value=1
)

sales = st.number_input(
    "Sales",
    min_value=0.0,
    step=1.0,
    value=100.0
)


# --------------------------------------------------
# Desired profit class
#
# IMPORTANT:
# This is NOT sent into the model.
# It only tells the app which predicted discount
# range the user wants.
# --------------------------------------------------

desired_profit = st.selectbox(
    "Desired Profit Level",
    [0, 1, 2],
    format_func=lambda x: {
        0: "Loss",
        1: "Medium Profit",
        2: "High Profit"
    }[x]
)


# --------------------------------------------------
# Find discount
# --------------------------------------------------

if st.button("Calculate Discount"):

    # ----------------------------------------------
    # Create one raw sample
    # ----------------------------------------------

    sample_df = pd.DataFrame([{

        "Ship Mode": ship_mode,
        "Segment": segment,
        "State": state,
        "Region": region,
        "Category": category,
        "Sub-Category": sub_category,

        "Sales": sales,
        "Quantity": quantity,

        # Temporary value.
        # get_discount_ranges() replaces this
        # from 0.000 → 0.800
        "Discount": 0.0,

        "Order Month": order_month
    }])


    # ----------------------------------------------
    # One-hot encode categorical columns
    # ----------------------------------------------

    sample_encoded = pd.get_dummies(sample_df)


    # ----------------------------------------------
    # Match EXACT training features
    #
    # Missing dummy columns become 0.
    # Extra columns are removed.
    # Column order becomes identical to training.
    # ----------------------------------------------

    sample_encoded = sample_encoded.reindex(
        columns=feature_columns,
        fill_value=0
    )


    sample_encoded = sample_encoded.astype(float)


    # ----------------------------------------------
    # Safety check
    # ----------------------------------------------

    if sample_encoded.shape[1] != 95:

        st.error(
            f"Model expects 95 features, but received "
            f"{sample_encoded.shape[1]}."
        )

        st.stop()


    # ----------------------------------------------
    # Convert to tensor
    # ----------------------------------------------

    sample_tensor = torch.tensor(
        sample_encoded.iloc[0].values,
        dtype=torch.float32
    )


    # ----------------------------------------------
    # Find Discount's exact position in the
    # final model input
    # ----------------------------------------------

    discount_index = feature_columns.index("Discount")


    # ----------------------------------------------
    # Calculate all discount ranges
    # ----------------------------------------------

    ranges, class_names = get_discount_ranges(
        model=model,
        sample=sample_tensor,
        discount_index=discount_index
    )


    # ----------------------------------------------
    # Display all discovered ranges
    # ----------------------------------------------

    st.subheader("Predicted Discount Ranges")

    for r in ranges:

        start_percent = r["start"] * 100
        end_percent = r["end"] * 100

        st.write(
            f"{start_percent:.1f}% – "
            f"{end_percent:.1f}% → "
            f"{class_names[r['class']]}"
        )


    # ----------------------------------------------
    # Find ranges matching user's desired class
    # ----------------------------------------------

    matching_ranges = [
        r
        for r in ranges
        if r["class"] == desired_profit
    ]


    if len(matching_ranges) == 0:

        st.warning(
            f"No discount between 0% and 80% "
            f"was predicted as "
            f"{class_names[desired_profit]}."
        )

    else:

        # ------------------------------------------
        # If the desired class appears multiple
        # times, use the widest continuous range
        # ------------------------------------------

        selected_range = max(
            matching_ranges,
            key=lambda r: r["end"] - r["start"]
        )


        # ------------------------------------------
        # Take the middle of the range
        # ------------------------------------------

        recommended_discount = (
            selected_range["start"]
            + selected_range["end"]
        ) / 2


        # ------------------------------------------
        # Display recommendation
        # ------------------------------------------

        st.subheader("Recommended Discount")

        st.success(
            f"{recommended_discount * 100:.1f}%"
        )

        st.write(
            f"Selected range: "
            f"{selected_range['start'] * 100:.1f}% – "
            f"{selected_range['end'] * 100:.1f}%"
        )

        st.write(
            f"Target: "
            f"{class_names[desired_profit]}"
        )