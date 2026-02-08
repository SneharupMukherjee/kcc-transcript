import tkinter as tk
import customtkinter as ctk
from tkinter import filedialog, Listbox, Scrollbar, messagebox
import pandas as pd
import geopandas as gpd
from fuzzywuzzy import process, fuzz
from bokeh.io import output_file, show
from bokeh.models import HoverTool, ColorBar, LinearColorMapper, PrintfTickFormatter
from bokeh.plotting import figure
from bokeh.palettes import OrRd
from bokeh.models import GeoJSONDataSource
import json
import os
import warnings

warnings.filterwarnings("ignore", category=UserWarning,
                        message="GeoDataFrame's CRS is not representable in URN OGC format")

selected_file_path = None
selected_sheet_name = None
corrections = {}
unmatched_districts = []
shapefile_districts = []
matched_districts = set()
continue_button_clicked = False

# Initialize the main window
ctk.set_appearance_mode("dark")  # Set the theme to light
root = ctk.CTk()
root.title("MapPlotter")
# Increased width to accommodate wider correction frame
root.geometry("1200x800")
root.state('zoomed')
root.resizable(False, False)
root.iconbitmap("Map_Icon.ico")

# Define custom font and color
custom_font = ("Lato", 14)

bold_italic_font = ctk.CTkFont(
    family="Lato", size=15, weight="bold", slant="italic")
fluorescent_blue = "#ffff8e"  # Fluorescent blue color


# Load existing corrections from a file (e.g., corrections.json)


def load_corrections(correction_file="corrections.json"):
    if os.path.exists(correction_file):
        with open(correction_file, 'r') as file:
            return json.load(file)
    return {}

# Save corrections to a file (e.g., corrections.json)


def save_corrections(corrections, correction_file="corrections.json"):
    with open(correction_file, 'w') as file:
        json.dump(corrections, file)


# Load the corrections at the start of the program
corrections = load_corrections()


def update_generate_button_state(*args):
    """Enable the Generate Map button only if a sheet is selected."""
    if sheet_listbox.curselection():
        generate_map_button.configure(state=tk.NORMAL)
    else:
        generate_map_button.configure(state=tk.DISABLED)


def open_file():
    global selected_file_path
    status_textbox.delete(1.0, tk.END)
    file_path = filedialog.askopenfilename(
        filetypes=[("Excel files", "*.xlsx *.xls")])
    if file_path:
        try:
            selected_file_path = file_path
            excel_file = pd.ExcelFile(file_path)
            sheet_listbox.delete(0, tk.END)  # Clear previous list
            for sheet in excel_file.sheet_names:
                sheet_listbox.insert(tk.END, sheet)
            status_textbox.insert(
                tk.END, f"Loaded {len(excel_file.sheet_names)} sheets from the file.\n")
        except Exception as e:
            status_textbox.insert(tk.END, f"Error loading file: {e}\n")
    status_textbox.update()


def combine_directions(district_name, next_column_value):
    directions = ["east", "west", "south", "north"]
    if district_name in directions:
        return district_name + next_column_value
    return district_name

# Function to update the district matching interface


def update_district_matching_interface():
    unmatched_dropdown.set_completion_list(unmatched_districts)
    unmatched_dropdown.set(
        unmatched_districts[0] if unmatched_districts else "")
    update_possible_matches()  # Filter the second combo box
    continue_button.configure(
        state=tk.NORMAL if unmatched_districts else tk.DISABLED)


def update_possible_matches(*args):
    """Update the possible matches based on unmatched districts."""
    remaining_shapefile_districts = [
        district for district in shapefile_districts if district not in matched_districts]
    possible_dropdown.set_completion_list(remaining_shapefile_districts)
    possible_dropdown.set("")  # Clear the second combo box for user input


def match_district():
    global unmatched_districts

    selected_unmatched = unmatched_dropdown.get()
    selected_possible = possible_dropdown.get()

    if selected_unmatched and selected_possible and selected_unmatched in unmatched_districts:
        corrections[selected_unmatched] = selected_possible
        # Remove matched district from the list
        unmatched_districts.remove(selected_unmatched)
        # Mark the district as matched
        matched_districts.add(selected_possible)
        update_district_matching_interface()  # Refresh the combo boxes


def create_fuzzy_mapping(choices, targets, threshold=70):
    global unmatched_districts, shapefile_districts, matched_districts
    mapping = {}
    for target in targets:
        if target in corrections:
            mapping[target] = corrections[target]
            matched_districts.add(corrections[target])
            continue

        match_tuple = process.extractOne(target, choices, scorer=fuzz.ratio)
        if match_tuple is not None and match_tuple[1] >= threshold:
            mapping[target] = match_tuple[0]
            matched_districts.add(match_tuple[0])
        else:
            unmatched_districts.append(target)
            mapping[target] = None  # Will be corrected later

    # Populate all shapefile districts in alphabetical order
    shapefile_districts = sorted(choices)

    return mapping


def continue_map_generation():
    global continue_button_clicked, df, india_districts, data_districts
    continue_button_clicked = True

    save_corrections(corrections)

    status_textbox.insert(
        tk.END, "\nContinuing map generation with corrections...\nPlease Wait========This will Take 20-30 SECONDS===============\n")
    status_textbox.update()

    df['Matched_District'] = df['District'].map(
        lambda x: corrections.get(x, x))

    generate_map_step_2()


def generate_map_step_1():
    global selected_sheet_name, unmatched_districts, df, india_districts, data_districts, india_states
    unmatched_districts.clear()
    matched_districts.clear()
    try:
        selected_sheet = sheet_listbox.get(tk.ACTIVE)
        selected_sheet_name = sheet_listbox.get(tk.ACTIVE)
        if selected_sheet:
            status_textbox.insert(
                tk.END, f"Generating map for {selected_sheet}...\n")

            file_path = selected_file_path
            sheet_name = selected_sheet_name
            df = pd.read_excel(file_path, sheet_name=sheet_name)

            df = df.rename(
                columns={df.columns[0]: 'District', df.columns[1]: 'consumer_spend'})
            status_textbox.insert(
                tk.END, "Normalizing District Names from Excel\n")
            status_textbox.update()

            df['District'] = df['District'].str.strip().str.lower(
            ).str.replace(' ', '').str.replace(r'-', '', regex=False)

            status_textbox.insert(
                tk.END, "Shape File for Districts Loaded... PLEASE WAIT.....\n")
            status_textbox.update()

            district_shapefile_path = "district/DISTRICT_BOUNDARY.shp"
            india_districts = gpd.read_file(district_shapefile_path)

            # Load the shapefile for India states
            state_shapefile_path = "state/STATE_BOUNDARY.shp"
            india_states = gpd.read_file(state_shapefile_path)

            columns_to_normalize = ['District', 'STATE']

            india_districts[columns_to_normalize] = india_districts[columns_to_normalize].apply(
                lambda col: col.str.strip().str.lower().str.replace(
                    ' ', '').str.replace(r'-', '', regex=False).str.replace(r'|', 'i').str.replace(r'@', 'u').str.replace(r'>', 'a')
            )

            india_districts['District'] = india_districts.apply(
                lambda row: combine_directions(row['District'], row['STATE']), axis=1)

            mapping = create_fuzzy_mapping(
                india_districts['District'], df['District'], threshold=80)

            df['Matched_District'] = df['District'].map(mapping)

            update_district_matching_interface()

            status_textbox.insert(
                tk.END, "Please correct the unmatched districts and press Match.\nIf no entries to Match Please press FINALISE and CONTINUE")
        else:
            status_textbox.insert(tk.END, "Please select a sheet first.\n")
    except Exception as e:
        status_textbox.insert(tk.END, f"Error: {e}\n")
    status_textbox.update()


def generate_map_step_2():
    global df, india_districts, india_states
    try:
        merged = india_districts.set_index('District').join(
            df.set_index('Matched_District'))

        palette = list(OrRd[9])
        reversed_palette = palette[::-1]

        color_mapper = LinearColorMapper(palette=reversed_palette, low=df['consumer_spend'].min(
        ), high=df['consumer_spend'].max(), nan_color='#efeeec')

        merged_json = json.loads(merged.to_json())
        geosource = GeoJSONDataSource(geojson=json.dumps(merged_json))

        states_json = json.loads(india_states.to_json())
        states_source = GeoJSONDataSource(geojson=json.dumps(states_json))

        p = figure(title=title_entry.get(), height=850,
                   width=850, toolbar_location="below", tools="pan, wheel_zoom, box_zoom, reset, save")

        district_renderer = p.patches('xs', 'ys', source=geosource, fill_color={'field': 'consumer_spend', 'transform': color_mapper},
                                      line_color='grey', line_width=0.25, fill_alpha=1)

        p.patches('xs', 'ys', source=states_source, fill_alpha=0,
                  line_color='black', line_width=0.3)

        hover = HoverTool(renderers=[district_renderer])
        hover.tooltips = [
            ("District", "@District"),
            ("Consumer Spend", "@consumer_spend{0.00%}")
        ]
        p.add_tools(hover)

        color_bar = ColorBar(color_mapper=color_mapper,
                             width=8, location=(0, 0))
        color_bar.formatter = PrintfTickFormatter(format="%.1f%%")

        p.add_layout(color_bar, 'right')

        output_file("district_spending_map.html",
                    title=title_entry.get())

        show(p)

        status_textbox.insert(
            tk.END, "Map generation completed.\n")
    except Exception as e:
        status_textbox.insert(tk.END, f"Error: {e}\n")
    status_textbox.update()


class AutocompleteCombobox(ctk.CTkComboBox):
    def set_completion_list(self, completion_list):
        self._completion_list = sorted(completion_list)
        self._hits = []
        self._hit_index = 0
        self.position = 0
        self.configure(values=self._completion_list)
        self.bind('<KeyRelease>', self._on_keyrelease)

    def _on_keyrelease(self, event):
        if event.keysym in ('BackSpace', 'Left', 'Right', 'Up', 'Down'):
            return
        self._autocomplete()

    def _autocomplete(self):
        pattern = self.get().lower()
        self._hits = [
            item for item in self._completion_list if pattern in item.lower()]
        if self._hits:
            self.configure(values=self._hits)
            self.event_generate('<Down>')  # Open the dropdown


try:
    main_frame = ctk.CTkFrame(root)
    main_frame.pack(fill=tk.BOTH, expand=True)

    left_frame = ctk.CTkFrame(main_frame)
    left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)

    # Increase the width of the correction frame
    correction_frame = ctk.CTkFrame(main_frame)
    # Allow the frame to expand
    correction_frame.pack(side=tk.RIGHT, fill=tk.BOTH,
                          expand=True, padx=10, pady=10)

    open_file_button = ctk.CTkButton(
        left_frame, text="Open Base File", command=open_file, font=custom_font)
    open_file_button.pack(pady=10)

    title_label = ctk.CTkLabel(
        left_frame, text="Select The Relevant Data Sheet Name from list below")
    title_label.pack(pady=5)

    listbox_frame = ctk.CTkFrame(left_frame)
    listbox_frame.pack(pady=10)

    scrollbar = Scrollbar(listbox_frame)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    sheet_listbox = Listbox(listbox_frame, height=5,
                            width=50, yscrollcommand=scrollbar.set, font=custom_font)
    sheet_listbox.pack(side=tk.LEFT, fill=tk.BOTH)

    scrollbar.config(command=sheet_listbox.yview)
    sheet_listbox.bind('<<ListboxSelect>>', update_generate_button_state)

    title_label = ctk.CTkLabel(left_frame, text="Map Title:")
    title_label.pack(pady=5)
    title_entry = ctk.CTkEntry(
        left_frame, width=600, font=custom_font, fg_color="lightblue", text_color="black")
    title_entry.pack(pady=5)
    title_entry.insert(0, "Heat Map of Consumer Spending in Indian Districts")

    generate_map_button = ctk.CTkButton(
        left_frame, text="Generate Map", command=generate_map_step_1, font=custom_font, fg_color='green', state=tk.DISABLED)
    generate_map_button.pack(pady=10)

    status_textbox = ctk.CTkTextbox(left_frame, height=250, width=700)
    status_textbox.pack(pady=10)
    status_textbox.configure(state=tk.NORMAL, wrap=tk.WORD,
                             fg_color="black", text_color="white", font=("Consolas", 15))

    # Create the disclaimer label with the specified formatting
    disclaimer_label = ctk.CTkLabel(
        left_frame,
        text="NOTE: This application is designed exclusively for Shailesh Jha for a specific set of data provided. This application does not manipulate / update / change any existing data of user. The Excel data to be proided must be in a specfic format (2 colums - 1 with district name and other with the required data to plot as HeatMap)",
        font=bold_italic_font,
        text_color="white",
        wraplength=700  # Adjust wraplength as needed
    )
    disclaimer_label.pack(padx=2, pady=2)

    # Add the designed by label
    designer_label = ctk.CTkLabel(
        left_frame,
        text="Designed and Developed by: Pankaj Sonawane - (pankaj.sonawane@kotak.com)",
        font=bold_italic_font,
        text_color=fluorescent_blue,
        wraplength=700
    )
    designer_label.pack(padx=5, pady=2)

    # Add the libraries used label
    libraries_label = ctk.CTkLabel(
        left_frame,
        text="Python Libraries used: CustomTkinter, Pandas, GeoPandas, fuzzywuzzy and Bokeh",
        font=bold_italic_font,
        text_color=fluorescent_blue,
        wraplength=700
    )
    libraries_label.pack(padx=5, pady=2)

    # New code right side
    possible_var = tk.StringVar()
    unmatched_var = tk.StringVar()
    unmatched_dropdown = AutocompleteCombobox(
        correction_frame, variable=unmatched_var, font=custom_font)
    # Make the dropdown fill the width of the frame
    unmatched_dropdown.pack(fill=tk.X, padx=5, pady=5)

    possible_dropdown = AutocompleteCombobox(
        correction_frame, variable=possible_var, font=custom_font)
    # Make the dropdown fill the width of the frame
    possible_dropdown.pack(fill=tk.X, padx=5, pady=5)

    continue_button = ctk.CTkButton(
        correction_frame, text="Match", command=match_district, font=custom_font, fg_color='blue', state=tk.DISABLED)
    # Make the button fill the width of the frame
    continue_button.pack(fill=tk.X, padx=5, pady=10)

    finalize_button = ctk.CTkButton(
        correction_frame, text="Finalize and Continue", command=continue_map_generation, font=custom_font, fg_color='green')
    # Make the button fill the width of the frame
    finalize_button.pack(fill=tk.X, padx=5, pady=10)

    description_label = ctk.CTkLabel(
        correction_frame, text="Check the corrections.json file stored at the location where the program is installed for details of matched districts", font=custom_font, text_color='white', wraplength=450)
    # Make the button fill the width of the frame
    description_label.pack(padx=5, pady=10)

    # New code ends right side


except Exception as e:
    messagebox.showerror("Error", f"An error occurred: {e}")
root.mainloop()
