import streamlit as st
import pandas as pd
import numpy as np
import json
from pathlib import Path
from xgboost import XGBClassifier


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

    class_names = {
        0: "Loss",
        1: "Medium Profit",
        2: "High Profit"
    }

    # Every discount we want to try
    # (rounded so the values don't drift)
    discounts = np.round(
        np.arange(min_discount, max_discount + step / 2, step),
        3
    )

    # One row per discount:
    # copy the sample and only change the discount
    sample = np.asarray(sample, dtype=np.float32)

    X = np.tile(sample, (len(discounts), 1))

    X[:, discount_index] = discounts

    predicted = model.predict(X)


    # --------------------------------------------------
    # Convert predictions into ranges
    # --------------------------------------------------

    ranges = []

    start_discount = discounts[0]
    current_class = int(predicted[0])

    for i in range(1, len(discounts)):

        if predicted[i] != current_class:

            ranges.append({
                "start": float(start_discount),
                "end": float(discounts[i - 1]),
                "class": current_class
            })

            start_discount = discounts[i]
            current_class = int(predicted[i])


    # Final range
    ranges.append({
        "start": float(start_discount),
        "end": float(discounts[-1]),
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

@st.cache_resource
def load_model():

    model = XGBClassifier()

    model.load_model(
        BASE_DIR / "notebooks" / "xgb_model.json"
    )

    return model


model = load_model()


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

    # Order Month is a number, so it has to be listed
    # here or get_dummies skips it
    sample_encoded = pd.get_dummies(
        sample_df,
        columns=[
            "Ship Mode",
            "Segment",
            "State",
            "Region",
            "Category",
            "Sub-Category",
            "Order Month"
        ]
    )


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
    # Convert to a numpy row
    # ----------------------------------------------

    sample_array = sample_encoded.iloc[0].values.astype(np.float32)


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
        sample=sample_array,
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
            f"No discount gives {class_names[desired_profit]} "
            f"for this order. Try another profit level "
            f"or change the Sales amount."
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