import json
import os

notebook_path = r"c:\Users\Kenshin\Documents\Data Analytics for AI\Crime_Safety_Classification_Project\notebooks\Safety_Classification - Copy.ipynb"

# Define the new cell for Excel export
new_cell = {
    "cell_type": "code",
    "execution_count": None,
    "id": "export_excel",
    "metadata": {},
    "outputs": [],
    "source": [
        "# 5. Export Predictions to Excel\n",
        "import os\n",
        "\n",
        "# Ensure reports directory exists\n",
        "output_dir = '../reports'\n",
        "if not os.path.exists(output_dir):\n",
        "    os.makedirs(output_dir)\n",
        "\n",
        "output_path = os.path.join(output_dir, 'future_safety_predictions.xlsx')\n",
        "future_safety_report.to_excel(output_path, index=False)\n",
        "print(f\"Predictions successfully saved to: {os.path.abspath(output_path)}\")"
    ]
}

# Read the notebook
if os.path.exists(notebook_path):
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb_data = json.load(f)

    # Append the new cell using the same logic as before (before metadata, effectively extending cells list)
    # nb_data['cells'] is a list, just append
    nb_data['cells'].append(new_cell)

    # Write back
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(nb_data, f, indent=1)

    print("Notebook updated with Excel export code.")
else:
    print(f"Error: Notebook not found at {notebook_path}")
