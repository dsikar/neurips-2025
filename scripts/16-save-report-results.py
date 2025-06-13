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
                                 val_loss: float,
                                 overall_accuracy: Optional[float] = None,
                                 distance_mae: Optional[float] = None):
        """Add classification model results."""
        self.results["models"][model_name] = {
            "task_type": "classification",
            "dataset": dataset,
            "train_accuracy": train_accuracy,
            "train_loss": train_loss,
            "val_accuracy": val_accuracy,
            "val_loss": val_loss,
            "overall_accuracy": overall_accuracy,
            "distance_mae": distance_mae
        }
        self.save_results()
    
    def add_regression_result(self, 
                             model_name: str,
                             dataset: str,
                             train_loss: float,
                             eval_loss: float,
                             mae: float,
                             overall_mae: Optional[float] = None,
                             distance_mae: Optional[float] = None):
        """Add regression model results."""
        self.results["models"][model_name] = {
            "task_type": "regression", 
            "dataset": dataset,
            "train_loss": train_loss,
            "eval_loss": eval_loss,
            "mae": mae,
            "overall_mae": overall_mae,
            "distance_mae": distance_mae
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
            markdown += "| Model Name | Dataset | Train Acc | Train Loss | Val Acc | Val Loss | O Acc | D MAE |\n"
            markdown += "|------------|---------|-----------|------------|---------|----------|-------|-------|\n"
            
            for name, data in cls_models.items():
                train_acc = self._format_value(data['train_accuracy'], is_accuracy=True)
                train_loss = self._format_value(data['train_loss'])
                val_acc = self._format_value(data['val_accuracy'], is_accuracy=True)
                val_loss = self._format_value(data['val_loss'])
                overall_acc = self._format_value(data.get('overall_accuracy'), is_accuracy=True)
                distance_mae = self._format_value(data.get('distance_mae'))
                
                markdown += f"| {name} | {data['dataset']} | {train_acc} | {train_loss} | {val_acc} | {val_loss} | {overall_acc} | {distance_mae} |\n"
        
        # Regression models
        reg_models = {k: v for k, v in self.results["models"].items() 
                     if v["task_type"] == "regression"}
        
        if reg_models:
            markdown += "\n## Regression Models\n\n"
            markdown += "| Model Name | Dataset | Train Loss | Eval Loss | MAE | O MAE | D MAE |\n"
            markdown += "|------------|---------|------------|-----------|-----|-------|-------|\n"
            
            for name, data in reg_models.items():
                train_loss = self._format_value(data['train_loss'])
                eval_loss = self._format_value(data['eval_loss'])
                mae = self._format_value(data['mae'])
                overall_mae = self._format_value(data.get('overall_mae'))
                distance_mae = self._format_value(data.get('distance_mae'))
                
                markdown += f"| {name} | {data['dataset']} | {train_loss} | {eval_loss} | {mae} | {overall_mae} | {distance_mae} |\n"
        
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
            latex += "\\begin{longtable}{@{}llllllll@{}}\n"
            latex += "\\toprule\n"
            latex += "Model Name & Dataset & T Acc & T Loss & V Acc & V Loss & O Acc & D MAE \\\\\n"
            latex += "\\midrule\n"
            latex += "\\endfirsthead\n"
            latex += "\\toprule\n"
            latex += "Model Name & Dataset & T Acc & T Loss & V Acc & V Loss & O Acc & D MAE \\\\\n"
            latex += "\\midrule\n"
            latex += "\\endhead\n"
            
            for name, data in cls_models.items():
                train_acc = self._format_value(data['train_accuracy'], is_accuracy=True, for_latex=True)
                train_loss = self._format_value(data['train_loss'], for_latex=True)
                val_acc = self._format_value(data['val_accuracy'], is_accuracy=True, for_latex=True)
                val_loss = self._format_value(data['val_loss'], for_latex=True)
                overall_acc = self._format_value(data.get('overall_accuracy'), is_accuracy=True, for_latex=True)
                distance_mae = self._format_value(data.get('distance_mae'), for_latex=True)
                
                # Escape underscores for LaTeX
                name_escaped = name.replace('_', '\\_')
                dataset_escaped = data['dataset'].replace('_', '\\_')
                
                latex += f"{name_escaped} & {dataset_escaped} & {train_acc} & {train_loss} & {val_acc} & {val_loss} & {overall_acc} & {distance_mae} \\\\\n"
            
            latex += "\\bottomrule\n"
            latex += "\\caption{Classification Model Performance}\n"
            latex += "\\end{longtable}\n\n"
        
        # Regression models
        reg_models = {k: v for k, v in self.results["models"].items() 
                     if v["task_type"] == "regression"}
        
        if reg_models:
            latex += "\\section{Regression Models}\n"
            latex += "\\begin{longtable}{@{}lllllll@{}}\n"
            latex += "\\toprule\n"
            latex += "Model Name & Dataset & Train Loss & Eval Loss & MAE & O MAE & D MAE \\\\\n"
            latex += "\\midrule\n"
            latex += "\\endfirsthead\n"
            latex += "\\toprule\n"
            latex += "Model Name & Dataset & Train Loss & Eval Loss & MAE & O MAE & D MAE \\\\\n"
            latex += "\\midrule\n"
            latex += "\\endhead\n"
            
            for name, data in reg_models.items():
                train_loss = self._format_value(data['train_loss'])
                eval_loss = self._format_value(data['eval_loss'])
                mae = self._format_value(data['mae'])
                overall_mae = self._format_value(data.get('overall_mae'))
                distance_mae = self._format_value(data.get('distance_mae'))
                
                # Escape underscores for LaTeX
                name_escaped = name.replace('_', '\\_')
                dataset_escaped = data['dataset'].replace('_', '\\_')
                
                latex += f"{name_escaped} & {dataset_escaped} & {train_loss} & {eval_loss} & {mae} & {overall_mae} & {distance_mae} \\\\\n"
            
            latex += "\\bottomrule\n"
            latex += "\\caption{Regression Model Performance}\n"
            latex += "\\end{longtable}\n\n"
        
        latex += "\\end{document}"
        return latex
    
# Example usage
if __name__ == "__main__":
    # DELETE BEFORE RUNNING
    # rm self_driving_results.json
    # TODO
    # 1. Incorporate Overall Accuracy for classification models - O Acc, 1 script for classifiers
    # 2. Incorporate Overall MAE for regression models - O MAE, 1 script for regressors
    # 3. Incorporate Distance MAE for all models - D MAE, scripts 1 and 2 should also compute this, hopefully

    manager = SimpleResultsManager("self_driving_results.json")
    
    ##############################
    # Add classification results #
    ##############################   
    # CNN models with different binning and balancing
    manager.add_classification_result(
        "ClsCNN3bB", "3-bin Bal", 
        train_accuracy=0.8316, train_loss=0.3797,
        val_accuracy=0.8302, val_loss=0.3808,
        distance_mae = 0.0365, overall_accuracy=0.8336
    )
    manager.add_classification_result(
        "ClsCNN3bU", "3-bin Unbal", 
        train_accuracy=0.8697, train_loss=0.3129,
        val_accuracy=0.8714, val_loss=0.3110,
        distance_mae = 0.0466, overall_accuracy=0.8736
    )       
    manager.add_classification_result(
        "ClsCNN5bB", "5-bin Bal", 
        train_accuracy=0.8745, train_loss=0.3001,
        val_accuracy=0.8692, val_loss=0.2975,
        distance_mae = 0.0491, overall_accuracy=0.8770
    )
    manager.add_classification_result(
        "ClsCNN5bU", "5-bin Unbal", 
        train_accuracy=0.9287, train_loss=0.1993,
        val_accuracy=0.9303, val_loss=0.1928,
        distance_mae = 0.0753, overall_accuracy=0.9318
    )
    manager.add_classification_result(
        "ClsCNN15bB", "15-bin Bal", 
        train_accuracy=0.4801, train_loss=1.0744,
        val_accuracy=0.4781, val_loss=1.0598,
        distance_mae = 0.0162, overall_accuracy=0.4801
    )
    manager.add_classification_result(
        "ClsCNN15bU", "15-bin Unbal", 
        train_accuracy=0.7376, train_loss=0.7241,
        val_accuracy=0.7534, val_loss=0.6839,
        distance_mae = 0.0194, overall_accuracy=0.7511
    )
    # ViT models with different binning and balancing
    manager.add_classification_result(
        "ClsViT3bB", "3-bin Bal", 
        train_accuracy=None, train_loss=0.18745221195837497,
        val_accuracy=0.9715048975957258, val_loss=0.09603918343782425,
        distance_mae = 0.0462, overall_accuracy=0.979653
    )
    manager.add_classification_result(
        "ClsViT3bU", "3-bin Unbal", 
        train_accuracy=None, train_loss=0.26249011515332965,
        val_accuracy=0.9192800148450547, val_loss=0.20747409760951996,
        distance_mae = 0.0397, overall_accuracy=0.924984
    )
    manager.add_classification_result(
        "ClsViT5bB", "5-bin Bal", 
        train_accuracy=None, train_loss=0.13174371687250105,
        val_accuracy=0.9754403687101458, val_loss=0.07662060111761093,
        distance_mae = 0.0454, overall_accuracy=0.977675
    )
    manager.add_classification_result(
        "ClsViT5bU", "5-bin Unbal", 
        train_accuracy=None, train_loss=0.19849784551300945,
        val_accuracy=0.9405445452861642, val_loss=0.15902680158615112,
        distance_mae = 0.0666, overall_accuracy=0.949687
    )
    manager.add_classification_result(
        "ClsViT15bB", "15-bin Bal", 
        train_accuracy=None, train_loss=0.4012311922046025,
        val_accuracy=0.9292087459554859, val_loss=0.19837264716625214,
        distance_mae = 0.0925, overall_accuracy=0.938563
    )
    manager.add_classification_result(
        "ClsViT15bU", "15-bin Unbal", 
        train_accuracy=None, train_loss=None,
        val_accuracy=None, val_loss=None,
        distance_mae = 0.0844, overall_accuracy=0.765513
    )
    ##############################
    # Add regression results     #
    ##############################
    manager.add_regression_result(
        "RegCNNCUFid", "Cont Unbal w/ Fids",
        train_loss=0.0000, eval_loss=0.0000, mae=None,
        distance_mae = 0.0461, overall_mae=0.003588
    )
    manager.add_regression_result(
        "RegCNNCU", "Continuous Unbal",
        train_loss=None, eval_loss=None, mae=None,
        distance_mae = 0.0588, overall_mae=0.003315
    )
    manager.add_regression_result(
        "RgrViTCU", "Continuous Unbal",
        train_loss=0.00017209326360099003, eval_loss=8.113816875265911e-05, mae=0.007625640835613012,
        distance_mae = 0.0399, overall_mae=0.007563
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