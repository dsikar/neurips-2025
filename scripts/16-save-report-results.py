"""
Simple Results Manager for Self-Driving Model Performance
Streamlined for 15 models with basic training metrics.
"""

import json
from datetime import datetime
from typing import Dict, Any, Optional
from pathlib import Path


class SimpleResultsManager:
    """Simple manager for basic model results - easily extensible later."""
    
    def __init__(self, results_file: str = "model_results.json"):
        self.results_file = Path(results_file)
        self.results = self._load_results()
    
    def _load_results(self) -> Dict[str, Any]:
        """Load existing results or create new structure."""
        if self.results_file.exists():
            with open(self.results_file, 'r') as f:
                return json.load(f)
        else:
            return {"models": {}}
    
    def save_results(self):
        """Save results to file."""
        with open(self.results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
    
    def add_classification_result(self, 
                                 model_name: str,
                                 dataset: str,
                                 train_accuracy: float,
                                 train_loss: float,
                                 val_accuracy: float,
                                 val_loss: float):
        """Add classification model results."""
        self.results["models"][model_name] = {
            "task_type": "classification",
            "dataset": dataset,
            "train_accuracy": train_accuracy,
            "train_loss": train_loss,
            "val_accuracy": val_accuracy,
            "val_loss": val_loss
        }
        self.save_results()
    
    def add_regression_result(self, 
                             model_name: str,
                             dataset: str,
                             train_loss: float,
                             eval_loss: float,
                             mae: float):
        """Add regression model results."""
        self.results["models"][model_name] = {
            "task_type": "regression", 
            "dataset": dataset,
            "train_loss": train_loss,
            "eval_loss": eval_loss,
            "mae": mae
        }
        self.save_results()
    
    def _format_value(self, value, is_accuracy=False, for_latex=False):
        """Format values with proper decimal places and handle None."""
        if value is None:
            return "N/A"
        if is_accuracy:
            percent_symbol = "\\%" if for_latex else "%"
            return f"{value * 100:.2f}{percent_symbol}"
        else:
            return f"{value:.4f}"
    
    def generate_obsidian_tables(self) -> str:
        """Generate simple Obsidian tables."""
        markdown = "# Model Performance Summary\n\n"
        
        # Classification models
        cls_models = {k: v for k, v in self.results["models"].items() 
                     if v["task_type"] == "classification"}
        
        if cls_models:
            markdown += "## Classification Models\n\n"
            markdown += "| Model Name | Dataset | Train Acc | Train Loss | Val Acc | Val Loss |\n"
            markdown += "|------------|---------|-----------|------------|---------|----------|\n"
            
            for name, data in cls_models.items():
                train_acc = self._format_value(data['train_accuracy'], is_accuracy=True)
                train_loss = self._format_value(data['train_loss'])
                val_acc = self._format_value(data['val_accuracy'], is_accuracy=True)
                val_loss = self._format_value(data['val_loss'])
                
                markdown += f"| {name} | {data['dataset']} | {train_acc} | {train_loss} | {val_acc} | {val_loss} |\n"
        
        # Regression models
        reg_models = {k: v for k, v in self.results["models"].items() 
                     if v["task_type"] == "regression"}
        
        if reg_models:
            markdown += "\n## Regression Models\n\n"
            markdown += "| Model Name | Dataset | Train Loss | Eval Loss | MAE |\n"
            markdown += "|------------|---------|------------|-----------|-----|\n"
            
            for name, data in reg_models.items():
                train_loss = self._format_value(data['train_loss'])
                eval_loss = self._format_value(data['eval_loss'])
                mae = self._format_value(data['mae'])
                
                markdown += f"| {name} | {data['dataset']} | {train_loss} | {eval_loss} | {mae} |\n"
        
        return markdown
    
    def export_to_latex(self) -> str:
        """Generate LaTeX tables with proper formatting."""
        latex = "\\documentclass{article}\n"
        latex += "\\usepackage{booktabs}\n"
        latex += "\\usepackage{array}\n"
        latex += "\\usepackage{longtable}\n"
        latex += "\\begin{document}\n\n"
        
        # Classification models
        cls_models = {k: v for k, v in self.results["models"].items() 
                     if v["task_type"] == "classification"}
        
        if cls_models:
            latex += "\\section{Classification Models}\n"
            latex += "\\begin{longtable}{@{}llllll@{}}\n"
            latex += "\\toprule\n"
            latex += "Model Name & Dataset & Train Acc & Train Loss & Val Acc & Val Loss \\\\\n"
            latex += "\\midrule\n"
            latex += "\\endfirsthead\n"
            latex += "\\toprule\n"
            latex += "Model Name & Dataset & Train Acc & Train Loss & Val Acc & Val Loss \\\\\n"
            latex += "\\midrule\n"
            latex += "\\endhead\n"
            
            for name, data in cls_models.items():
                train_acc = self._format_value(data['train_accuracy'], is_accuracy=True, for_latex=True)
                train_loss = self._format_value(data['train_loss'], for_latex=True)
                val_acc = self._format_value(data['val_accuracy'], is_accuracy=True, for_latex=True)
                val_loss = self._format_value(data['val_loss'], for_latex=True)
                
                # Escape underscores for LaTeX
                name_escaped = name.replace('_', '\\_')
                dataset_escaped = data['dataset'].replace('_', '\\_')
                
                latex += f"{name_escaped} & {dataset_escaped} & {train_acc} & {train_loss} & {val_acc} & {val_loss} \\\\\n"
            
            latex += "\\bottomrule\n"
            latex += "\\caption{Classification Model Performance}\n"
            latex += "\\end{longtable}\n\n"
        
        # Regression models
        reg_models = {k: v for k, v in self.results["models"].items() 
                     if v["task_type"] == "regression"}
        
        if reg_models:
            latex += "\\section{Regression Models}\n"
            latex += "\\begin{longtable}{@{}lllll@{}}\n"
            latex += "\\toprule\n"
            latex += "Model Name & Dataset & Train Loss & Eval Loss & MAE \\\\\n"
            latex += "\\midrule\n"
            latex += "\\endfirsthead\n"
            latex += "\\toprule\n"
            latex += "Model Name & Dataset & Train Loss & Eval Loss & MAE \\\\\n"
            latex += "\\midrule\n"
            latex += "\\endhead\n"
            
            for name, data in reg_models.items():
                train_loss = self._format_value(data['train_loss'])
                eval_loss = self._format_value(data['eval_loss'])
                mae = self._format_value(data['mae'])
                
                # Escape underscores for LaTeX
                name_escaped = name.replace('_', '\\_')
                dataset_escaped = data['dataset'].replace('_', '\\_')
                
                latex += f"{name_escaped} & {dataset_escaped} & {train_loss} & {eval_loss} & {mae} \\\\\n"
            
            latex += "\\bottomrule\n"
            latex += "\\caption{Regression Model Performance}\n"
            latex += "\\end{longtable}\n\n"
        
        latex += "\\end{document}"
        return latex
    
# Example usage
if __name__ == "__main__":
    manager = SimpleResultsManager("self_driving_results.json")
    
    ##############################
    # Add classification results #
    ##############################   
    # CNN models with different binning and balancing
    manager.add_classification_result(
        "ClsCNN3bB", "3-bin Balanced", 
        train_accuracy=0.8316, train_loss=0.3797,
        val_accuracy=0.8302, val_loss=0.3808
    )
    manager.add_classification_result(
        "ClsCNN3bU", "3-bin Unbalanced", 
        train_accuracy=0.8697, train_loss=0.3129,
        val_accuracy=0.8714, val_loss=0.3110
    )       
    manager.add_classification_result(
        "ClsCNN5bB", "5-bin Balanced", 
        train_accuracy=0.8745, train_loss=0.3001,
        val_accuracy=0.8692, val_loss=0.2975
    )
    manager.add_classification_result(
        "ClsCNN5bU", "5-bin Unbalanced", 
        train_accuracy=0.9287, train_loss=0.1993,
        val_accuracy=0.9303, val_loss=0.1928
    )
    manager.add_classification_result(
        "ClsCNN15bB", "15-bin Balanced", 
        train_accuracy=0.4801, train_loss=1.0744,
        val_accuracy=0.4781, val_loss=1.0598
    )
    manager.add_classification_result(
        "ClsCNN15bU", "15-bin Unbalanced", 
        train_accuracy=0.7376, train_loss=0.7241,
        val_accuracy=0.7534, val_loss=0.6839
    ) 
    # ViT models with different binning and balancing
    manager.add_classification_result(
        "ClsViT3bB", "3-bin Bal", 
        train_accuracy=None, train_loss=0.18745221195837497,
        val_accuracy=0.9715048975957258, val_loss=0.09603918343782425
    )
    manager.add_classification_result(
        "ClsViT3bU", "3-bin Unbal", 
        train_accuracy=None, train_loss=0.26249011515332965,
        val_accuracy=0.9192800148450547, val_loss=0.20747409760951996
    )
    manager.add_classification_result(
        "ClsViT5bB", "5-bin Bal", 
        train_accuracy=None, train_loss=0.13174371687250105,
        val_accuracy=0.9754403687101458, val_loss=0.07662060111761093
    )
    manager.add_classification_result(
        "ClsViT5bU", "5-bin Unbal", 
        train_accuracy=None, train_loss=0.19849784551300945,
        val_accuracy=0.9405445452861642, val_loss=0.15902680158615112
    )
    manager.add_classification_result(
        "ClsViT15bB", "15-bin Bal", 
        train_accuracy=None, train_loss=0.4012311922046025,
        val_accuracy=0.9292087459554859, val_loss=0.19837264716625214
    )
    manager.add_classification_result(
        "ClsViT15bU", "15-bin Unbal", 
        train_accuracy=None, train_loss=None,
        val_accuracy=None, val_loss=None
    )
    ##############################
    # Add regression results     #
    ##############################
    manager.add_regression_result(
        "RegCNNCUFid", "Continuous Unbal w/ Fiducials",
        train_loss=0.0000, eval_loss=0.0000, mae=None
    )
    manager.add_regression_result(
        "RegCNNCU", "Continuous Unbal",
        train_loss=None, eval_loss=None, mae=None
    )
    manager.add_regression_result(
        "RgrViTCU", "Continuous Unbal",
        train_loss=0.00017209326360099003, eval_loss=8.113816875265911e-05, mae=0.007625640835613012
    )    
    # Generate tables obsidian format
    tables = manager.generate_obsidian_tables()
    print(tables)
    
    # Save to file
    with open("Model-Performance-Summary.md", "w") as f:
        f.write(tables)

    # Generate LaTeX tables
    latex_content = manager.export_to_latex()
    with open("model_results.tex", "w") as f:
        f.write(latex_content)

    print(latex_content)    