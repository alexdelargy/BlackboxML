import pandas as pd
import tkinter as tk
from tkinter import ttk

class ModelApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Model Designer")
        
        # Load data
        self.df = pd.read_csv('car_price_prediction.csv')
        self.X = self.df.drop(columns=['Price'])
        self.y = self.df['Price']
        
        # Create widgets
        self.create_widgets()
    
    def create_widgets(self):
        # Model type selection
        self.model_type_label = tk.Label(self.root, text="Select Model Type:")
        self.model_type_label.grid(row=0, column=0, padx=10, pady=10)
        
        self.model_type = tk.StringVar()
        self.model_type_combobox = ttk.Combobox(self.root, textvariable=self.model_type)
        self.model_type_combobox['values'] = ('Linear', 'RandomForest')
        self.model_type_combobox.grid(row=0, column=1, padx=10, pady=10)
        
        # Train button
        self.train_button = tk.Button(self.root, text="Train Model", command=self.train_model)
        self.train_button.grid(row=1, column=0, columnspan=2, padx=10, pady=10)
        
        # Output text
        self.output_text = tk.Text(self.root, height=10, width=50)
        self.output_text.grid(row=2, column=0, columnspan=2, padx=10, pady=10)