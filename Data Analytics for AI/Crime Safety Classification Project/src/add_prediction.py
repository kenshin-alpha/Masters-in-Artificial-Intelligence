import json
import os

notebook_path = r"c:\Users\Kenshin\Documents\Data Analytics for AI\Crime_Safety_Classification_Project\notebooks\Safety_Classification - Copy.ipynb"

# Define the new cells
new_cells = [
    {
        "cell_type": "markdown",
        "id": "pred_intro",
        "metadata": {},
        "source": [
            "## 5. Future Prediction (Next Year's Safety)\n",
            "\n",
            "In this final section, we use the trained Neural Network model to predict the **Next Year's Safety Level** for every Garda Station.\n",
            "We use the *latest available year's crime data* for each station as the input features."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "id": "pred_code",
        "metadata": {},
        "outputs": [],
        "source": [
            "# 1. Retrain Best Model on FULL Dataset (Neural Network)\n",
            "print(\"Training final model on full historical dataset...\")\n",
            "final_model = Pipeline([\n",
            "    ('scaler', RobustScaler()),\n",
            "    ('classifier', MLPClassifier(hidden_layer_sizes=(100,64,64,32), max_iter=500, random_state=42))\n",
            "])\n",
            "final_model.fit(X, y_encoded)\n",
            "print(\"Model trained successfully.\")\n",
            "\n",
            "# 2. Prepare Input Data for Prediction (Latest Year)\n",
            "# We go back to 'df_pivot' which includes the last year (where Next_Year_Crime was NaN)\n",
            "latest_data = df_pivot.sort_values('Year').groupby('Garda Station', as_index=False).last()\n",
            "print(f\"\\nPredicting for {len(latest_data)} stations using data from year(s): {latest_data['Year'].unique()}\")\n",
            "\n",
            "X_future = latest_data[offence_cols]\n",
            "\n",
            "# 3. Make Predictions\n",
            "future_preds_encoded = final_model.predict(X_future)\n",
            "future_preds_labels = le.inverse_transform(future_preds_encoded)\n",
            "\n",
            "# 4. Create Results DataFrame\n",
            "future_safety_report = pd.DataFrame({\n",
            "    'Garda Station': latest_data['Garda Station'],\n",
            "    'Division': latest_data.get('Division', 'Unknown'),\n",
            "    'Current_Year_Crime': latest_data['Total_Crime'],\n",
            "    'Predicted_Next_Year_Safety': future_preds_labels\n",
            "})\n",
            "\n",
            "# Display Results\n",
            "print(\"\\n================================================\")\n",
            "print(\"PREDICTED SAFETY LEVELS FOR NEXT YEAR\")\n",
            "print(\"================================================\")\n",
            "print(future_safety_report['Predicted_Next_Year_Safety'].value_counts())\n",
            "\n",
            "print(\"\\nSample Predictions (Top Unsafe Predictions by Current Volume):\")\n",
            "print(future_safety_report[future_safety_report['Predicted_Next_Year_Safety'] == 'Unsafe']\n",
            "      .sort_values('Current_Year_Crime', ascending=False).head(10).to_string(index=False))\n",
            "\n",
            "print(\"\\nSample Predictions (Safe Stations):\")\n",
            "print(future_safety_report[future_safety_report['Predicted_Next_Year_Safety'] == 'Safe']\n",
            "      .head(5).to_string(index=False))"
        ]
    }
]

# Read the notebook
with open(notebook_path, 'r', encoding='utf-8') as f:
    nb_data = json.load(f)

# Append new cells
nb_data['cells'].extend(new_cells)

# Write back
with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb_data, f, indent=1)

print("Notebook updated successfully.")
