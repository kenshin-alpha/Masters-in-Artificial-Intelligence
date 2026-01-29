import json
import os

notebook_path = r"c:\Users\Kenshin\Documents\Data Analytics for AI\Crime_Safety_Classification_Project\notebooks\Safety_Classification - Copy.ipynb"

# The corrected source code for the prediction cell
corrected_source = [
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
    "# FIX: Calculate Division for latest_data because it might be missing in df_pivot\n",
    "latest_data['Division'] = latest_data['Garda Station'].apply(\n",
    "    lambda x: x.split(', ')[-1].replace(' Division', '') if ', ' in str(x) else 'Unknown'\n",
    ")\n",
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
    "    'Division': latest_data['Division'],\n",
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

if os.path.exists(notebook_path):
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb_data = json.load(f)

    # Find the cell to update (the one with id="pred_code")
    cell_found = False
    for cell in nb_data['cells']:
        if cell.get('id') == "pred_code":
            cell['source'] = corrected_source
            cell_found = True
            break
    
    if cell_found:
        with open(notebook_path, 'w', encoding='utf-8') as f:
            json.dump(nb_data, f, indent=1)
        print("Notebook updated successfully: Division column logic added.")
    else:
        print("Error: Target prediction cell not found.")

else:
    print(f"Error: Notebook not found at {notebook_path}")
